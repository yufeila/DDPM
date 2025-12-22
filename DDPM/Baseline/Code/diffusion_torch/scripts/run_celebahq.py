"""
CelebA-HQ 256x256 training and sampling script for DDPM.

Migrated from: diffusion_tf/scripts/run_celebahq.py
Migration notes:
- TF training loop -> PyTorch training loop
- TPU -> DDP multi-GPU training  
- NHWC -> NCHW data format
- block_size (space_to_depth) for memory optimization

Usage:
    # Training
    python scripts/run_celebahq.py train \
        --log_dir ./logs/celebahq \
        --data_dir ./data/celebahq \
        --batch_size 32 \
        --lr 2e-5 \
        --num_steps 500000

        CUDA_VISIBLE_DEVICES=1 \
python scripts/run_celebahq.py train \
  --log_dir ./logs/celebahq \
  --data_dir ./data/celebahq \
  --batch_size 32 \
  --lr 2e-5 \
  --num_steps 500000


    # Multi-GPU Training
    torchrun --nproc_per_node=4 scripts/run_celebahq.py train \
        --log_dir ./logs/celebahq_ddp \
        --data_dir ./data/celebahq \
        --batch_size 64

    # Sampling
    python scripts/run_celebahq.py sample \
        --checkpoint ./logs/celebahq/checkpoints/ema_500000.pt \
        --output_dir ./samples/celebahq \
        --num_samples 1000

Expected results:
    logs/celebahq/
    ├── checkpoints/
    │   ├── checkpoint_010000.pt
    │   ├── ema_010000.pt
    │   └── ...
    ├── samples/
    │   ├── samples_010000.png
    │   └── ...
    └── tensorboard/
"""

import argparse
import os
import time
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image

# Import from diffusion_torch package
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from diffusion_torch.models.unet import UNet
from diffusion_torch.diffusion_utils import GaussianDiffusion, get_beta_schedule
from diffusion_torch.data_utils.dist_utils import (
    setup_distributed, cleanup_distributed, is_main_process, 
    get_rank, get_world_size, wrap_model_ddp, seed_all, barrier
)
from diffusion_torch.data_utils.summaries import DistributedSummaryWriter, ScalarTracker
from diffusion_torch.utils import tile_imgs


# ===== CelebA-HQ Dataset =====

class CelebAHQDataset(Dataset):
    """
    CelebA-HQ dataset.
    
    Expects images in data_dir with format: xxxxx.png or xxxxx.jpg
    Images should be 256x256 or will be resized.
    """
    
    def __init__(self, data_dir: str, image_size: int = 256, randflip: bool = True):
        self.data_dir = data_dir
        self.image_size = image_size
        self.randflip = randflip
        
        # Find all image files
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            import glob
            self.image_files.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
        
        self.image_files = sorted(self.image_files)
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        print(f"Found {len(self.image_files)} images in {data_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Resize if needed
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Convert to numpy array [H, W, C] in [0, 255]
        img = np.array(img, dtype=np.float32)
        
        # Random horizontal flip
        if self.randflip and np.random.rand() > 0.5:
            img = img[:, ::-1, :].copy()
        
        # Convert to tensor [C, H, W]
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        return {'image': img, 'label': 0}  # label is dummy for unconditional


def normalize_data(x):
    """Normalize from [0, 255] to [-1, 1]."""
    return x / 127.5 - 1.0


def unnormalize_data(x):
    """Unnormalize from [-1, 1] to [0, 255]."""
    return (x + 1.0) * 127.5


# ===== Default Configuration =====

def get_default_config():
    """
    Get default configuration for CelebA-HQ training.
    
    Matches TF config: ch=128, ch_mult=(1, 1, 2, 2, 4, 4), attn_resolutions=(16,)
    """
    return {
        # Model - matches unet2d16b2c112244 with 114M params
        'ch': 128,
        'ch_mult': (1, 1, 2, 2, 4, 4),
        'num_res_blocks': 2,
        'attn_resolutions': (16,),
        'dropout': 0.0,
        'resamp_with_conv': True,
        'block_size': 1,  # space_to_depth block size for memory reduction
        
        # Diffusion
        'num_diffusion_timesteps': 1000,
        'beta_schedule': 'linear',
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'loss_type': 'noisepred',
        
        # Training
        'learning_rate': 2e-5,
        'batch_size': 64,
        'num_train_steps': 500000,
        'warmup_steps': 5000,
        'grad_clip': 1.0,
        'ema_decay': 0.9999,
        
        # Logging
        'log_interval': 100,
        'save_interval': 10000,
        'sample_interval': 10000,
        'num_samples': 16,
        
        # Data
        'data_dir': './data/celebahq',
        'image_size': 256,
        'num_channels': 3,
        'randflip': True,
        
        # Misc
        'seed': 42,
    }


def create_model(config: dict, device: torch.device) -> nn.Module:
    """
    Create UNet model for CelebA-HQ.
    
    Corresponds to TF's unet2d16b2c112244 model (114M params).
    """
    block_size = config.get('block_size', 1)
    in_ch = config['num_channels']
    out_ch = config['num_channels']
    
    # If using block_size > 1, adjust channels
    if block_size != 1:
        in_ch *= block_size ** 2
        out_ch *= block_size ** 2
    
    model = UNet(
        in_ch=in_ch,
        ch=config['ch'],
        out_ch=out_ch,
        num_res_blocks=config['num_res_blocks'],
        attn_resolutions=config['attn_resolutions'],
        dropout=config['dropout'],
        ch_mult=config['ch_mult'],
    )
    return model.to(device)


def create_diffusion(config: dict, device: torch.device) -> GaussianDiffusion:
    """Create GaussianDiffusion instance."""
    betas = get_beta_schedule(
        beta_schedule=config['beta_schedule'],
        beta_start=config['beta_start'],
        beta_end=config['beta_end'],
        num_diffusion_timesteps=config['num_diffusion_timesteps'],
    )
    return GaussianDiffusion(betas=betas, loss_type=config['loss_type'], device=device)


# ===== EMA Helper =====

class EMAHelper:
    """Exponential Moving Average of model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        return self.shadow.copy()
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict.copy()


def get_lr_with_warmup(step: int, warmup_steps: int, base_lr: float) -> float:
    """Get learning rate with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr


def space_to_depth(x, block_size):
    """PyTorch equivalent of tf.nn.space_to_depth. [B, C, H, W] -> [B, C*block^2, H/block, W/block]"""
    if block_size == 1:
        return x
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // block_size, block_size, W // block_size, block_size)
    x = x.permute(0, 1, 3, 5, 2, 4).reshape(B, C * block_size * block_size, H // block_size, W // block_size)
    return x


def depth_to_space(x, block_size):
    """PyTorch equivalent of tf.nn.depth_to_space. [B, C*block^2, H, W] -> [B, C, H*block, W*block]"""
    if block_size == 1:
        return x
    B, C, H, W = x.shape
    out_ch = C // (block_size * block_size)
    x = x.reshape(B, out_ch, block_size, block_size, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, out_ch, H * block_size, W * block_size)
    return x


# ===== Training Function =====

def train(
    config: dict,
    log_dir: str,
    resume_checkpoint: Optional[str] = None,
    device: torch.device = None,
):
    """Train DDPM on CelebA-HQ."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    distributed = setup_distributed()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    
    seed_all(config['seed'] + get_rank())
    
    if is_main_process():
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'samples'), exist_ok=True)
    barrier()
    
    model = create_model(config, device)
    diffusion = create_diffusion(config, device)
    
    if is_main_process():
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")
    
    if distributed:
        model = wrap_model_ddp(model, device_ids=[local_rank])
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    ema = EMAHelper(model, decay=config['ema_decay'])
    
    start_step = 0
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        if distributed:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ema.load_state_dict(checkpoint['ema_state_dict'])
        start_step = checkpoint['step']
        if is_main_process():
            print(f"Resumed from step {start_step}")
    
    # Create dataset
    dataset = CelebAHQDataset(
        data_dir=config['data_dir'],
        image_size=config['image_size'],
        randflip=config['randflip'],
    )
    
    batch_size = config['batch_size'] // get_world_size()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if distributed else None
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )
    
    writer = DistributedSummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))
    tracker = ScalarTracker()
    
    block_size = config.get('block_size', 1)
    model.train()
    step = start_step
    data_iter = iter(dataloader)
    
    if is_main_process():
        print(f"Starting training from step {start_step}")
        print(f"Total steps: {config['num_train_steps']}")
    
    pbar = tqdm(range(start_step, config['num_train_steps']), 
                disable=not is_main_process(), desc="Training")
    
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            if distributed:
                sampler.set_epoch(step // len(dataloader))
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        images = batch['image'].to(device)
        images = normalize_data(images)
        
        # Apply space_to_depth for memory reduction
        if block_size != 1:
            images = space_to_depth(images, block_size)
        
        B = images.shape[0]
        t = torch.randint(0, config['num_diffusion_timesteps'], (B,), device=device)
        
        def denoise_fn(x, t):
            return model(x, t)
        
        losses = diffusion.p_losses(denoise_fn, images, t)
        loss = losses.mean()
        
        optimizer.zero_grad()
        loss.backward()
        
        if config['grad_clip'] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
        lr = get_lr_with_warmup(step, config['warmup_steps'], config['learning_rate'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.step()
        
        if distributed:
            ema.update(model.module)
        else:
            ema.update(model)
        
        tracker.add('loss', loss.item())
        tracker.add('lr', lr)
        
        if step % config['log_interval'] == 0 and step > 0:
            avg_loss = tracker.mean('loss')
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{lr:.2e}'})
            if is_main_process():
                writer.add_scalar('train/loss', avg_loss, step)
                writer.add_scalar('train/lr', lr, step)
            tracker.reset()
        
        if step % config['save_interval'] == 0 and step > 0:
            if is_main_process():
                save_checkpoint(model, optimizer, ema, step, config, log_dir, distributed)
        
        if step % config['sample_interval'] == 0 and step > 0:
            if is_main_process():
                generate_and_save_samples(model, ema, diffusion, config, step, log_dir, device, distributed)
            barrier()
    
    if is_main_process():
        save_checkpoint(model, optimizer, ema, step, config, log_dir, distributed)
    
    writer.close()
    cleanup_distributed()
    
    if is_main_process():
        print("Training completed!")


# ===== Checkpoint and Sampling Functions =====

def save_checkpoint(model, optimizer, ema, step, config, log_dir, distributed=False):
    """Save training checkpoint."""
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    
    model_state = model.module.state_dict() if distributed else model.state_dict()
    
    checkpoint = {
        'step': step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'config': config,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{step:06d}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    ema_path = os.path.join(checkpoint_dir, f'ema_{step:06d}.pt')
    torch.save(ema.state_dict(), ema_path)
    print(f"Saved EMA to {ema_path}")


def load_checkpoint_for_sampling(checkpoint_path, config, device):
    """Load checkpoint for sampling (EMA weights)."""
    model = create_model(config, device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    if 'ema_state_dict' in state_dict:
        ema_state = state_dict['ema_state_dict']
    elif 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
        return model
    else:
        ema_state = state_dict
    
    for name, param in model.named_parameters():
        if name in ema_state:
            param.data = ema_state[name]
    
    return model


@torch.no_grad()
def generate_and_save_samples(model, ema, diffusion, config, step, log_dir, device, distributed=False):
    """Generate and save sample images during training."""
    model.eval()
    
    if distributed:
        ema.apply_shadow(model.module)
    else:
        ema.apply_shadow(model)
    
    block_size = config.get('block_size', 1)
    num_samples = config['num_samples']
    
    if block_size != 1:
        H = W = config['image_size'] // block_size
        C = config['num_channels'] * block_size * block_size
    else:
        H = W = config['image_size']
        C = config['num_channels']
    
    shape = (num_samples, C, H, W)
    
    def denoise_fn(x, t):
        return model(x, t)
    
    samples = diffusion.p_sample_loop(denoise_fn=denoise_fn, shape=shape)
    
    if distributed:
        ema.restore(model.module)
    else:
        ema.restore(model)
    
    model.train()
    
    # Apply depth_to_space if needed
    if block_size != 1:
        samples = depth_to_space(samples, block_size)
    
    samples = unnormalize_data(samples)
    samples = samples.clamp(0, 255).to(torch.uint8)
    
    samples_np = samples.cpu().numpy()
    samples_nhwc = samples_np.transpose(0, 2, 3, 1)
    
    grid = tile_imgs(samples_nhwc)
    
    img = Image.fromarray(grid)
    sample_path = os.path.join(log_dir, 'samples', f'samples_{step:06d}.png')
    img.save(sample_path)
    print(f"Saved samples to {sample_path}")
    
    npz_path = os.path.join(log_dir, 'samples', f'samples_{step:06d}.npz')
    np.savez_compressed(npz_path, samples=samples_np)


def sample(
    checkpoint_path: str,
    output_dir: str,
    num_samples: int = 1000,
    batch_size: int = 16,
    config: Optional[dict] = None,
    device: torch.device = None,
):
    """Generate samples from a trained model."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = config or get_default_config()
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading checkpoint from {checkpoint_path}")
    model = load_checkpoint_for_sampling(checkpoint_path, config, device)
    model.eval()
    
    diffusion = create_diffusion(config, device)
    block_size = config.get('block_size', 1)
    
    print(f"Generating {num_samples} samples with batch size {batch_size}")
    
    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Sampling"):
        current_batch = min(batch_size, num_samples - i * batch_size)
        
        if block_size != 1:
            H = W = config['image_size'] // block_size
            C = config['num_channels'] * block_size * block_size
        else:
            H = W = config['image_size']
            C = config['num_channels']
        
        shape = (current_batch, C, H, W)
        
        def denoise_fn(x, t):
            return model(x, t)
        
        samples = diffusion.p_sample_loop(denoise_fn=denoise_fn, shape=shape)
        
        if block_size != 1:
            samples = depth_to_space(samples, block_size)
        
        samples = unnormalize_data(samples)
        samples = samples.clamp(0, 255).to(torch.uint8)
        
        all_samples.append(samples.cpu().numpy())
    
    all_samples = np.concatenate(all_samples, axis=0)[:num_samples]
    
    output_path = os.path.join(output_dir, 'samples.npz')
    np.savez_compressed(output_path, samples=all_samples)
    print(f"Saved {num_samples} samples to {output_path}")
    
    preview_samples = all_samples[:16]
    preview_nhwc = preview_samples.transpose(0, 2, 3, 1)
    grid = tile_imgs(preview_nhwc)
    
    img = Image.fromarray(grid)
    preview_path = os.path.join(output_dir, 'preview.png')
    img.save(preview_path)
    print(f"Saved preview to {preview_path}")
    
    return all_samples


# ===== Main Entry Point =====

def parse_args():
    parser = argparse.ArgumentParser(description='DDPM CelebA-HQ Training/Sampling')
    subparsers = parser.add_subparsers(dest='mode', help='Mode: train or sample')
    
    train_parser = subparsers.add_parser('train', help='Train DDPM')
    train_parser.add_argument('--log_dir', type=str, default='./logs/celebahq')
    train_parser.add_argument('--data_dir', type=str, default='./data/celebahq')
    train_parser.add_argument('--resume', type=str, default=None)
    train_parser.add_argument('--batch_size', type=int, default=64)
    train_parser.add_argument('--lr', type=float, default=2e-5)
    train_parser.add_argument('--num_steps', type=int, default=500000)
    train_parser.add_argument('--block_size', type=int, default=1, help='space_to_depth block size')
    train_parser.add_argument('--seed', type=int, default=42)
    
    sample_parser = subparsers.add_parser('sample', help='Generate samples')
    sample_parser.add_argument('--checkpoint', type=str, required=True)
    sample_parser.add_argument('--output_dir', type=str, default='./samples/celebahq')
    sample_parser.add_argument('--num_samples', type=int, default=1000)
    sample_parser.add_argument('--batch_size', type=int, default=16)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.mode == 'train':
        config = get_default_config()
        config['batch_size'] = args.batch_size
        config['learning_rate'] = args.lr
        config['num_train_steps'] = args.num_steps
        config['data_dir'] = args.data_dir
        config['block_size'] = args.block_size
        config['seed'] = args.seed
        
        train(config=config, log_dir=args.log_dir, resume_checkpoint=args.resume)
    
    elif args.mode == 'sample':
        sample(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
        )
    
    else:
        print("Usage:")
        print("  python run_celebahq.py train --log_dir ./logs/celebahq --data_dir ./data/celebahq")
        print("  python run_celebahq.py sample --checkpoint ./logs/celebahq/checkpoints/ema_500000.pt")


if __name__ == '__main__':
    main()
