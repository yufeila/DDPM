"""
Utility functions for diffusion models.

Migrated from: diffusion_tf/utils.py
Migration notes:
- TF SummaryWriter -> torch.utils.tensorboard.SummaryWriter
- tf.set_random_seed -> torch.manual_seed
- Image tiling logic remains numpy-based
- NHWC image arrays for tile_imgs (numpy), model uses NCHW
"""

import contextlib
import io
import os
import random
import time
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter as TBSummaryWriter


class SummaryWriter:
    """
    TensorBoard summary writer.
    Corresponds to TF's custom SummaryWriter class.
    """
    
    def __init__(self, log_dir: str, write_graph: bool = True):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = TBSummaryWriter(log_dir=log_dir)
        self._log_dir = log_dir
    
    def flush(self):
        self.writer.flush()
    
    def close(self):
        self.writer.close()
    
    def scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        self.writer.add_scalar(tag, float(value), step)
    
    def image(self, tag: str, image: np.ndarray, step: int):
        """
        Log an image.
        
        Args:
            tag: name for the image
            image: numpy array in HWC format, uint8 [0, 255]
            step: global step
        """
        image = np.asarray(image)
        if image.ndim == 2:
            image = image[:, :, None]
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        # TensorBoard expects HWC format for add_image with dataformats='HWC'
        self.writer.add_image(tag, image, step, dataformats='HWC')
    
    def images(self, tag: str, images: np.ndarray, step: int):
        """Log multiple images as a tiled grid."""
        self.image(tag, tile_imgs(images), step=step)


def seed_all(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def tile_imgs(imgs: np.ndarray, *, pad_pixels: int = 1, pad_val: int = 255, num_col: int = 0) -> np.ndarray:
    """
    Tile multiple images into a single grid image.
    
    Args:
        imgs: numpy array of shape [N, H, W, C] in NHWC format, uint8
        pad_pixels: padding between images
        pad_val: padding color value (0-255)
        num_col: number of columns, 0 for auto (square grid)
    
    Returns:
        Tiled image as numpy array [H', W', C]
    """
    assert pad_pixels >= 0 and 0 <= pad_val <= 255
    
    imgs = np.asarray(imgs)
    assert imgs.dtype == np.uint8
    if imgs.ndim == 3:
        imgs = imgs[..., None]
    n, h, w, c = imgs.shape
    assert c == 1 or c == 3, 'Expected 1 or 3 channels'
    
    if num_col <= 0:
        # Make a square grid
        ceil_sqrt_n = int(np.ceil(np.sqrt(float(n))))
        num_row = ceil_sqrt_n
        num_col = ceil_sqrt_n
    else:
        # Make a grid with specified columns
        assert n % num_col == 0
        num_row = int(np.ceil(n / num_col))
    
    imgs = np.pad(
        imgs,
        pad_width=((0, num_row * num_col - n), (pad_pixels, pad_pixels), (pad_pixels, pad_pixels), (0, 0)),
        mode='constant',
        constant_values=pad_val
    )
    h, w = h + 2 * pad_pixels, w + 2 * pad_pixels
    imgs = imgs.reshape(num_row, num_col, h, w, c)
    imgs = imgs.transpose(0, 2, 1, 3, 4)
    imgs = imgs.reshape(num_row * h, num_col * w, c)
    
    if pad_pixels > 0:
        imgs = imgs[pad_pixels:-pad_pixels, pad_pixels:-pad_pixels, :]
    if c == 1:
        imgs = imgs[..., 0]
    return imgs


def save_tiled_imgs(filename: str, imgs: np.ndarray, pad_pixels: int = 1, pad_val: int = 255, num_col: int = 0):
    """Save tiled images to file."""
    Image.fromarray(tile_imgs(imgs, pad_pixels=pad_pixels, pad_val=pad_val, num_col=num_col)).save(filename)


# ===== Math utilities =====

def approx_standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """
    Approximate the CDF of a standard normal distribution.
    Corresponds to TF's approx_standard_normal_cdf().
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(
    x: torch.Tensor, 
    *, 
    means: torch.Tensor, 
    log_scales: torch.Tensor
) -> torch.Tensor:
    """
    Compute log-likelihood for discretized Gaussian.
    Assumes data is integers [0, 255] rescaled to [-1, 1].
    
    Corresponds to TF's discretized_gaussian_log_likelihood().
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(torch.clamp(cdf_plus, min=1e-12))
    log_one_minus_cdf_min = torch.log(torch.clamp(1. - cdf_min, min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999, log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min,
                    torch.log(torch.clamp(cdf_delta, min=1e-12))))
    assert log_probs.shape == x.shape
    return log_probs


# ===== Training utilities =====

def rms(variables: List[torch.Tensor]) -> torch.Tensor:
    """Compute root mean square of a list of tensors."""
    total_sum = sum(torch.sum(v ** 2) for v in variables)
    total_numel = sum(v.numel() for v in variables)
    return torch.sqrt(total_sum / total_numel)


def get_warmed_up_lr(max_lr: float, warmup: int, global_step: int) -> float:
    """
    Compute learning rate with linear warmup.
    
    Corresponds to TF's get_warmed_up_lr().
    """
    if warmup == 0:
        return max_lr
    return max_lr * min(float(global_step) / float(warmup), 1.0)


@contextlib.contextmanager
def ema_scope(model: torch.nn.Module, ema_model: torch.nn.Module):
    """
    Context manager for using EMA parameters.
    Temporarily swaps model parameters with EMA parameters.
    
    Corresponds to TF's ema_scope().
    """
    # Store original parameters
    original_params = {name: param.data.clone() for name, param in model.named_parameters()}
    
    # Copy EMA parameters to model
    for name, param in model.named_parameters():
        if name in dict(ema_model.named_parameters()):
            param.data.copy_(dict(ema_model.named_parameters())[name].data)
    
    try:
        yield
    finally:
        # Restore original parameters
        for name, param in model.named_parameters():
            param.data.copy_(original_params[name])
