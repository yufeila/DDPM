"""
Evaluation worker for checkpoint polling and sample generation.

Migrated from: diffusion_tf/tpu_utils/simple_eval_worker.py
Migration notes:
- TPU-specific evaluation -> GPU-based evaluation
- TF checkpoint loading -> PyTorch checkpoint loading
- Polling loop for new checkpoints
"""

import glob
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .dist_utils import is_main_process, barrier


class CheckpointPoller:
    """
    Poll for new checkpoints in a directory.
    
    Corresponds to TF's checkpoint polling logic in simple_eval_worker.py.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        pattern: str = "*.pt",
        poll_interval: int = 60,
    ):
        """
        Initialize the checkpoint poller.
        
        Args:
            checkpoint_dir: directory containing checkpoints
            pattern: glob pattern for checkpoint files
            poll_interval: seconds between polls
        """
        self.checkpoint_dir = checkpoint_dir
        self.pattern = pattern
        self.poll_interval = poll_interval
        self._seen_checkpoints: set = set()
    
    def get_all_checkpoints(self) -> List[str]:
        """
        Get all checkpoint files sorted by modification time.
        
        Returns:
            List of checkpoint paths, oldest first
        """
        pattern = os.path.join(self.checkpoint_dir, self.pattern)
        checkpoints = glob.glob(pattern)
        # Sort by modification time
        checkpoints.sort(key=os.path.getmtime)
        return checkpoints
    
    def get_new_checkpoints(self) -> List[str]:
        """
        Get checkpoints that haven't been seen yet.
        
        Returns:
            List of new checkpoint paths
        """
        all_ckpts = self.get_all_checkpoints()
        new_ckpts = [c for c in all_ckpts if c not in self._seen_checkpoints]
        return new_ckpts
    
    def mark_seen(self, checkpoint_path: str):
        """Mark a checkpoint as seen."""
        self._seen_checkpoints.add(checkpoint_path)
    
    def poll_once(self) -> Optional[str]:
        """
        Check for new checkpoints once.
        
        Returns:
            Path to newest unseen checkpoint, or None
        """
        new_ckpts = self.get_new_checkpoints()
        if new_ckpts:
            return new_ckpts[-1]  # Return newest
        return None
    
    def poll_forever(self, callback: Callable[[str], None]):
        """
        Poll for new checkpoints forever, calling callback for each.
        
        Args:
            callback: function to call with each new checkpoint path
        """
        print(f"Polling for checkpoints in {self.checkpoint_dir}")
        while True:
            new_ckpt = self.poll_once()
            if new_ckpt is not None:
                print(f"Found new checkpoint: {new_ckpt}")
                callback(new_ckpt)
                self.mark_seen(new_ckpt)
            else:
                time.sleep(self.poll_interval)


def extract_step_from_path(path: str) -> int:
    """
    Extract step number from checkpoint path.
    
    Assumes format like 'model_010000.pt' or 'ema_0.9999_010000.pt'.
    
    Args:
        path: checkpoint file path
    
    Returns:
        Step number, or -1 if not found
    """
    basename = os.path.basename(path)
    name = os.path.splitext(basename)[0]
    
    # Try to find number at the end
    parts = name.split('_')
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    
    return -1


class EvalWorker:
    """
    Worker for evaluating model checkpoints.
    
    Corresponds to TF's SimpleEvalWorker.
    
    Polls for new checkpoints and generates samples for evaluation.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        output_dir: str,
        model_fn: Callable[[], nn.Module],
        sample_fn: Callable[[nn.Module, int], torch.Tensor],
        num_samples: int = 50000,
        batch_size: int = 256,
        checkpoint_pattern: str = "ema_*.pt",
        poll_interval: int = 60,
        device: torch.device = None,
    ):
        """
        Initialize the evaluation worker.
        
        Args:
            checkpoint_dir: directory containing model checkpoints
            output_dir: directory to save generated samples
            model_fn: function that returns a new model instance
            sample_fn: function (model, num_samples) -> samples tensor
            num_samples: number of samples to generate per checkpoint
            batch_size: batch size for sampling
            checkpoint_pattern: glob pattern for checkpoints
            poll_interval: seconds between checkpoint polls
            device: torch device
        """
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.model_fn = model_fn
        self.sample_fn = sample_fn
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.poller = CheckpointPoller(
            checkpoint_dir=checkpoint_dir,
            pattern=checkpoint_pattern,
            poll_interval=poll_interval,
        )
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_checkpoint(self, checkpoint_path: str, model: nn.Module) -> nn.Module:
        """
        Load model weights from checkpoint.
        
        Args:
            checkpoint_path: path to checkpoint file
            model: model instance
        
        Returns:
            Model with loaded weights
        """
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Remove 'module.' prefix if present (from DDP)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        return model
    
    def generate_samples(self, model: nn.Module) -> np.ndarray:
        """
        Generate samples using the model.
        
        Args:
            model: model to use for generation
        
        Returns:
            samples as numpy array [N, C, H, W] in [0, 255]
        """
        model.eval()
        all_samples = []
        
        num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
        
        with torch.no_grad():
            for i in range(num_batches):
                current_batch = min(self.batch_size, self.num_samples - i * self.batch_size)
                samples = self.sample_fn(model, current_batch)
                
                # Convert from [-1, 1] to [0, 255]
                samples = (samples + 1.0) * 127.5
                samples = samples.clamp(0, 255).to(torch.uint8)
                
                all_samples.append(samples.cpu().numpy())
                
                if is_main_process():
                    print(f"Generated batch {i+1}/{num_batches}")
        
        return np.concatenate(all_samples, axis=0)[:self.num_samples]
    
    def save_samples(self, samples: np.ndarray, step: int):
        """
        Save generated samples to disk.
        
        Args:
            samples: numpy array [N, C, H, W]
            step: training step
        """
        output_path = os.path.join(self.output_dir, f"samples_{step:06d}.npz")
        np.savez_compressed(output_path, samples=samples)
        print(f"Saved {len(samples)} samples to {output_path}")
    
    def evaluate_checkpoint(self, checkpoint_path: str):
        """
        Evaluate a single checkpoint.
        
        Args:
            checkpoint_path: path to checkpoint
        """
        step = extract_step_from_path(checkpoint_path)
        print(f"Evaluating checkpoint at step {step}: {checkpoint_path}")
        
        # Create and load model
        model = self.model_fn()
        model = model.to(self.device)
        model = self.load_checkpoint(checkpoint_path, model)
        
        # Generate samples
        samples = self.generate_samples(model)
        
        # Save samples (only on main process)
        if is_main_process():
            self.save_samples(samples, step)
        
        barrier()  # Sync all processes
    
    def run_once(self):
        """
        Check for and evaluate any new checkpoints once.
        """
        new_ckpt = self.poller.poll_once()
        if new_ckpt is not None:
            self.evaluate_checkpoint(new_ckpt)
            self.poller.mark_seen(new_ckpt)
    
    def run_forever(self):
        """
        Poll for new checkpoints forever.
        """
        self.poller.poll_forever(self.evaluate_checkpoint)
    
    def evaluate_all(self):
        """
        Evaluate all existing checkpoints.
        """
        all_ckpts = self.poller.get_all_checkpoints()
        for ckpt in all_ckpts:
            self.evaluate_checkpoint(ckpt)
            self.poller.mark_seen(ckpt)


def load_samples_from_npz(path: str) -> np.ndarray:
    """
    Load samples from an npz file.
    
    Args:
        path: path to npz file
    
    Returns:
        samples array [N, C, H, W]
    """
    data = np.load(path)
    if 'samples' in data:
        return data['samples']
    elif 'arr_0' in data:
        return data['arr_0']
    else:
        raise ValueError(f"Could not find samples in {path}")


def compute_fid_from_samples(
    real_activations: np.ndarray,
    generated_samples: np.ndarray,
    inception_model: nn.Module,
    batch_size: int = 256,
    device: torch.device = None,
) -> float:
    """
    Compute FID between real activations and generated samples.
    
    Args:
        real_activations: precomputed activations for real images [N, D]
        generated_samples: generated images [M, C, H, W] in [0, 255]
        inception_model: Inception model for computing activations
        batch_size: batch size for computing activations
        device: torch device
    
    Returns:
        FID score
    """
    from .metrics_numpy import frechet_classifier_distance_from_activations
    
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception_model = inception_model.to(device)
    inception_model.eval()
    
    # Compute activations for generated samples
    gen_activations = []
    num_batches = (len(generated_samples) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, len(generated_samples))
            batch = generated_samples[start:end]
            
            # Convert to tensor and normalize
            batch = torch.from_numpy(batch).float().to(device)
            batch = batch / 255.0  # [0, 1]
            
            # Get activations (assumes model returns activations)
            acts = inception_model(batch)
            if isinstance(acts, tuple):
                acts = acts[0]
            
            gen_activations.append(acts.cpu().numpy())
    
    gen_activations = np.concatenate(gen_activations, axis=0)
    
    # Compute FID
    fid = frechet_classifier_distance_from_activations(real_activations, gen_activations)
    
    return fid
