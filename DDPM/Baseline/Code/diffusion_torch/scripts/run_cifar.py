"""
CIFAR-10 training and sampling script for DDPM.

Migrated from: diffusion_tf/scripts/run_cifar.py
Migration notes:
- TF training loop -> PyTorch training loop
- TPU -> DDP multi-GPU training
- tf.train.Checkpoint -> torch.save/load
- NHWC -> NCHW data format

    # Sampling
    python scripts/run_cifar.py sample \
        --checkpoint ./logs/cifar10/checkpoints/ema_010000.pt \
        --output_dir ./samples/lsun_bedroom \
        --num_samples 1000
"""

import argparse
import os
import time
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from diffusion_torch package
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from diffusion_torch.models.unet import UNet
from diffusion_torch.diffusion_utils import GaussianDiffusion, get_beta_schedule
from diffusion_torch.data_utils.datasets import get_dataset, normalize_data, unnormalize_data, get_dataloader
from diffusion_torch.data_utils.dist_utils import (
    setup_distributed, cleanup_distributed, is_main_process, 
    get_rank, get_world_size, wrap_model_ddp, seed_all, barrier
)
from diffusion_torch.data_utils.summaries import DistributedSummaryWriter, ScalarTracker
from diffusion_torch.utils import tile_imgs


# ===== Default Configuration =====
# Corresponds to TF's config in run_cifar.py

def get_default_config():
    """Get default configuration for CIFAR-10 training."""
    return {
        # Model
        'ch': 128,
        'ch_mult': (1, 2, 2, 2),
        'num_res_blocks': 2,
        'attn_resolutions': (16,),
        'dropout': 0.1,
        'resamp_with_conv': True,
        
        # Diffusion
        'num_diffusion_timesteps': 1000,
        'beta_schedule': 'linear',
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'loss_type': 'noisepred',
        
        # Training
        'learning_rate': 2e-4,
        'batch_size': 128,
        'num_train_steps': 800000,
        'warmup_steps': 5000,
        'grad_clip': 1.0,
        'ema_decay': 0.9999,
        
        # Logging
        'log_interval': 100,
        'save_interval': 10000,
        'sample_interval': 10000,
        'num_samples': 64,
        
        # Data
        'data_dir': './data',
        'image_size': 32,
        'num_channels': 3,
        
        # Misc
        'seed': 42,
    }


def create_model(config: dict, device: torch.device) -> nn.Module:
    """
    Create UNet model for CIFAR-10.
    
    Corresponds to TF's model creation in run_cifar.py.
    """
    model = UNet(
        in_ch=config['num_channels'],
        ch=config['ch'],
        out_ch=config['num_channels'],
        num_res_blocks=config['num_res_blocks'],
        attn_resolutions=config['attn_resolutions'],
        dropout=config['dropout'],
        ch_mult=config['ch_mult'],
    )
    return model.to(device)


def create_diffusion(config: dict, device: torch.device) -> GaussianDiffusion:
    """
    Create GaussianDiffusion instance.
    
    Corresponds to TF's diffusion creation.
    """
    betas = get_beta_schedule(
        beta_schedule=config['beta_schedule'],
        beta_start=config['beta_start'],
        beta_end=config['beta_end'],
        num_diffusion_timesteps=config['num_diffusion_timesteps'],
    )
    return GaussianDiffusion(betas=betas, loss_type=config['loss_type'], device=device)


class EMAHelper:
    """
    Exponential Moving Average of model parameters.
    
    Corresponds to TF's EMA implementation.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        """Update EMA parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self, model: nn.Module):
        """Apply EMA parameters to model."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        """Restore original parameters."""
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


# ===== Training Function =====

def train(
    config: dict,
    log_dir: str,
    resume_checkpoint: Optional[str] = None,
    device: torch.device = None,
):
    """
    Train DDPM on CIFAR-10.
    
    Corresponds to TF's training loop in run_cifar.py.
    
    Args:
        config: training configuration
        log_dir: directory for logs and checkpoints
        resume_checkpoint: path to checkpoint to resume from
        device: torch device
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup distributed training
    distributed = setup_distributed()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    
    # Set random seed
    seed_all(config['seed'] + get_rank())
    
    # Create directories
    if is_main_process():
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'samples'), exist_ok=True)
    barrier()
    
    # Create model and diffusion
    model = create_model(config, device)
    diffusion = create_diffusion(config, device)
    
    if is_main_process():
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")
    
    # Wrap model with DDP
    if distributed:
        model = wrap_model_ddp(model, device_ids=[local_rank])
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Create EMA
    ema = EMAHelper(model, decay=config['ema_decay'])
    
    # Load checkpoint if resuming
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
    
    # Create dataset and dataloader
    dataset = get_dataset('cifar10', data_dir=config['data_dir'], train=True)
    
    # Adjust batch size for distributed training
    batch_size = config['batch_size'] // get_world_size()
    dataloader = get_dataloader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    
    # Create summary writer
    writer = DistributedSummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))
    tracker = ScalarTracker()
    
    # Training loop
    model.train()
    step = start_step
    data_iter = iter(dataloader)
    
    if is_main_process():
        print(f"Starting training from step {start_step}")
        print(f"Total steps: {config['num_train_steps']}")
    
    pbar = tqdm(range(start_step, config['num_train_steps']), 
                disable=not is_main_process(),
                desc="Training")
    
    for step in pbar:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        # Prepare data: [B, C, H, W] in [0, 255] -> [-1, 1]
        images = batch['image'].to(device)
        images = normalize_data(images)  # [0, 255] -> [-1, 1]
        
        B = images.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, config['num_diffusion_timesteps'], (B,), device=device)
        
        # Compute loss
        def denoise_fn(x, t):
            return model(x, t)
        
        losses = diffusion.p_losses(denoise_fn, images, t)
        loss = losses.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config['grad_clip'] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
        # Update learning rate with warmup
        lr = get_lr_with_warmup(step, config['warmup_steps'], config['learning_rate'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Optimizer step
        optimizer.step()
        
        # Update EMA
        if distributed:
            ema.update(model.module)
        else:
            ema.update(model)
        
        # Logging
        tracker.add('loss', loss.item())
        tracker.add('lr', lr)
        
        if step % config['log_interval'] == 0 and step > 0:
            avg_loss = tracker.mean('loss')
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{lr:.2e}'})
            
            if is_main_process():
                writer.add_scalar('train/loss', avg_loss, step)
                writer.add_scalar('train/lr', lr, step)
            
            tracker.reset()
        
        # Save checkpoint
        if step % config['save_interval'] == 0 and step > 0:
            if is_main_process():
                save_checkpoint(
                    model, optimizer, ema, step, config, log_dir, distributed
                )
        
        # Generate samples
        if step % config['sample_interval'] == 0 and step > 0:
            if is_main_process():
                generate_and_save_samples(
                    model, ema, diffusion, config, step, log_dir, device, distributed
                )
            barrier()
    
    # Final save
    if is_main_process():
        save_checkpoint(model, optimizer, ema, step, config, log_dir, distributed)
    
    writer.close()
    cleanup_distributed()
    
    if is_main_process():
        print("Training completed!")


# ===== Checkpoint Functions =====

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    ema: EMAHelper,
    step: int,
    config: dict,
    log_dir: str,
    distributed: bool = False,
):
    """Save training checkpoint."""
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    
    # Get model state dict
    if distributed:
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    # Save full checkpoint
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
    
    # Also save EMA-only checkpoint for sampling
    ema_path = os.path.join(checkpoint_dir, f'ema_{step:06d}.pt')
    torch.save(ema.state_dict(), ema_path)
    print(f"Saved EMA to {ema_path}")


def load_checkpoint_for_sampling(
    checkpoint_path: str,
    config: dict,
    device: torch.device,
) -> nn.Module:
    """Load checkpoint for sampling (EMA weights)."""
    model = create_model(config, device)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Handle different formats
    if 'ema_state_dict' in state_dict:
        # Full checkpoint
        ema_state = state_dict['ema_state_dict']
    elif 'model_state_dict' in state_dict:
        # Use model weights directly
        model.load_state_dict(state_dict['model_state_dict'])
        return model
    else:
        # Assume it's EMA state dict directly
        ema_state = state_dict
    
    # Load EMA weights
    for name, param in model.named_parameters():
        if name in ema_state:
            param.data = ema_state[name]
    
    return model


# ===== Sampling Functions =====

@torch.no_grad()
def generate_and_save_samples(
    model: nn.Module,
    ema: EMAHelper,
    diffusion: GaussianDiffusion,
    config: dict,
    step: int,
    log_dir: str,
    device: torch.device,
    distributed: bool = False,
):
    """Generate and save sample images during training."""
    model.eval()
    
    # Apply EMA weights
    if distributed:
        ema.apply_shadow(model.module)
    else:
        ema.apply_shadow(model)
    
    # Generate samples
    num_samples = config['num_samples']
    shape = (num_samples, config['num_channels'], config['image_size'], config['image_size'])
    
    def denoise_fn(x, t):
        return model(x, t)
    
    samples = diffusion.p_sample_loop(denoise_fn=denoise_fn, shape=shape)
    
    # Restore original weights
    if distributed:
        ema.restore(model.module)
    else:
        ema.restore(model)
    
    model.train()
    
    # Convert to [0, 255] and save
    samples = unnormalize_data(samples)  # [-1, 1] -> [0, 255]
    samples = samples.clamp(0, 255).to(torch.uint8)
    
    # Save as grid image
    samples_np = samples.cpu().numpy()  # [N, C, H, W]
    # Convert to NHWC for tile_imgs
    samples_nhwc = samples_np.transpose(0, 2, 3, 1)  # [N, H, W, C]
    
    grid = tile_imgs(samples_nhwc)
    
    # Save image
    from PIL import Image
    img = Image.fromarray(grid)
    sample_path = os.path.join(log_dir, 'samples', f'samples_{step:06d}.png')
    img.save(sample_path)
    print(f"Saved samples to {sample_path}")
    
    # Also save as npz
    npz_path = os.path.join(log_dir, 'samples', f'samples_{step:06d}.npz')
    np.savez_compressed(npz_path, samples=samples_np)


def sample(
    checkpoint_path: str,
    output_dir: str,
    num_samples: int = 50000,
    batch_size: int = 256,
    config: Optional[dict] = None,
    device: torch.device = None,
):
    """
    Generate samples from a trained model.
    
    Corresponds to TF's sampling in run_cifar.py.
    
    Args:
        checkpoint_path: path to checkpoint
        output_dir: directory to save samples
        num_samples: total number of samples to generate
        batch_size: batch size for generation
        config: model configuration (uses default if None)
        device: torch device
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = config or get_default_config()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading checkpoint from {checkpoint_path}")
    model = load_checkpoint_for_sampling(checkpoint_path, config, device)
    model.eval()
    
    # Create diffusion
    diffusion = create_diffusion(config, device)
    
    print(f"Generating {num_samples} samples with batch size {batch_size}")
    
    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Sampling"):
        current_batch = min(batch_size, num_samples - i * batch_size)
        shape = (current_batch, config['num_channels'], config['image_size'], config['image_size'])
        
        def denoise_fn(x, t):
            return model(x, t)
        
        samples = diffusion.p_sample_loop(denoise_fn=denoise_fn, shape=shape)
        
        # Convert to [0, 255]
        samples = unnormalize_data(samples)
        samples = samples.clamp(0, 255).to(torch.uint8)
        
        all_samples.append(samples.cpu().numpy())
    
    # Concatenate and save
    all_samples = np.concatenate(all_samples, axis=0)[:num_samples]
    
    output_path = os.path.join(output_dir, 'samples.npz')
    np.savez_compressed(output_path, samples=all_samples)
    print(f"Saved {num_samples} samples to {output_path}")
    
    # Also save a grid preview
    preview_samples = all_samples[:64]
    preview_nhwc = preview_samples.transpose(0, 2, 3, 1)
    grid = tile_imgs(preview_nhwc)
    
    from PIL import Image
    img = Image.fromarray(grid)
    preview_path = os.path.join(output_dir, 'preview.png')
    img.save(preview_path)
    print(f"Saved preview to {preview_path}")
    
    return all_samples


# ===== Main Entry Point =====

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DDPM CIFAR-10 Training/Sampling')
    
    subparsers = parser.add_subparsers(dest='mode', help='Mode: train or sample')
    
    # Training arguments
    train_parser = subparsers.add_parser('train', help='Train DDPM')
    train_parser.add_argument('--log_dir', type=str, default='./logs/cifar10',
                              help='Directory for logs and checkpoints')
    train_parser.add_argument('--resume', type=str, default=None,
                              help='Path to checkpoint to resume from')
    train_parser.add_argument('--batch_size', type=int, default=128,
                              help='Training batch size')
    train_parser.add_argument('--lr', type=float, default=2e-4,
                              help='Learning rate')
    train_parser.add_argument('--num_steps', type=int, default=800000,
                              help='Number of training steps')
    train_parser.add_argument('--data_dir', type=str, default='./data',
                              help='Data directory')
    train_parser.add_argument('--seed', type=int, default=42,
                              help='Random seed')
    
    # Sampling arguments
    sample_parser = subparsers.add_parser('sample', help='Generate samples')
    sample_parser.add_argument('--checkpoint', type=str, required=True,
                               help='Path to model checkpoint')
    sample_parser.add_argument('--output_dir', type=str, default='./samples',
                               help='Output directory for samples')
    sample_parser.add_argument('--num_samples', type=int, default=50000,
                               help='Number of samples to generate')
    sample_parser.add_argument('--batch_size', type=int, default=256,
                               help='Batch size for sampling')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.mode == 'train':
        config = get_default_config()
        config['batch_size'] = args.batch_size
        config['learning_rate'] = args.lr
        config['num_train_steps'] = args.num_steps
        config['data_dir'] = args.data_dir
        config['seed'] = args.seed
        
        train(
            config=config,
            log_dir=args.log_dir,
            resume_checkpoint=args.resume,
        )
    
    elif args.mode == 'sample':
        sample(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
        )
    
    else:
        print("Please specify mode: train or sample")
        print("Example:")
        print("  python run_cifar.py train --log_dir ./logs/cifar10")
        print("  python run_cifar.py sample --checkpoint ./logs/cifar10/checkpoints/ema_100000.pt")


if __name__ == '__main__':
    main()
