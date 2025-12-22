"""
Neural network building blocks for diffusion models.

Migrated from: diffusion_tf/nn.py
Migration notes:
- TF uses NHWC, PyTorch uses NCHW
- tf.variable_scope -> nn.Module with named submodules
- default_init: variance_scaling(fan_avg, uniform) -> kaiming_uniform with adjusted scale
- All conv2d/dense layers now follow PyTorch conventions
"""

import math
import string
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== Neural network building defaults =====
DEFAULT_DTYPE = torch.float32


def default_init(scale: float = 1.0):
    """
    Returns a weight initialization function.
    Corresponds to TF's variance_scaling(scale, fan_avg, uniform).
    """
    def _init(tensor):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        fan_avg = (fan_in + fan_out) / 2.0
        scale_actual = 1e-10 if scale == 0 else scale
        std = math.sqrt(scale_actual / fan_avg)
        bound = math.sqrt(3.0) * std  # uniform bounds
        with torch.no_grad():
            return tensor.uniform_(-bound, bound)
    return _init


# ===== Utilities =====

def debug_print(x: torch.Tensor, name: str) -> torch.Tensor:
    """Print tensor statistics for debugging."""
    print(f"{name}: mean={x.mean().item():.4f}, std={x.std().item():.4f}, "
          f"min={x.min().item():.4f}, max={x.max().item():.4f}")
    return x


def flatten(x: torch.Tensor) -> torch.Tensor:
    """Flatten tensor to [batch_size, -1]."""
    return x.view(x.shape[0], -1)


def sumflat(x: torch.Tensor) -> torch.Tensor:
    """Sum over all dimensions except batch."""
    return x.sum(dim=list(range(1, len(x.shape))))


def meanflat(x: torch.Tensor) -> torch.Tensor:
    """Mean over all dimensions except batch."""
    return x.mean(dim=list(range(1, len(x.shape))))


# ===== Neural network layers =====

class NIN(nn.Module):
    """
    Network-in-Network layer (1x1 convolution generalized to arbitrary dimensions).
    Corresponds to TF's nin() which uses einsum for contraction.
    
    For NCHW input (4D tensor), uses 1x1 conv.
    For NHWC input (4D tensor with last dim as channel), uses tensordot.
    """
    def __init__(self, in_dim: int, num_units: int, init_scale: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.num_units = num_units
        self.W = nn.Parameter(torch.empty(in_dim, num_units))
        self.b = nn.Parameter(torch.zeros(num_units))
        default_init(init_scale)(self.W)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[1] == self.in_dim:
            # NCHW format: use 1x1 convolution via einsum
            # x: [B, C_in, H, W] -> [B, C_out, H, W]
            y = torch.einsum('bchw,cd->bdhw', x, self.W) + self.b.view(1, -1, 1, 1)
        else:
            # General case: contract on last dimension
            # x: [..., in_dim] -> [..., num_units]
            y = torch.tensordot(x, self.W, dims=([-1], [0])) + self.b
        return y


class Dense(nn.Module):
    """
    Dense (fully connected) layer.
    Corresponds to TF's dense().
    """
    def __init__(self, in_dim: int, num_units: int, init_scale: float = 1.0, bias: bool = True):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_dim, num_units))
        self.bias = bias
        if bias:
            self.b = nn.Parameter(torch.zeros(num_units))
        default_init(init_scale)(self.W)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim] -> [B, num_units]
        z = torch.matmul(x, self.W)
        if self.bias:
            z = z + self.b
        return z


class Conv2d(nn.Module):
    """
    2D Convolution layer.
    Corresponds to TF's conv2d().
    
    Note: TF uses NHWC, PyTorch uses NCHW.
    Input shape: [B, C, H, W] (NCHW)
    """
    def __init__(
        self,
        in_channels: int,
        num_units: int,
        filter_size: Union[int, Tuple[int, int]] = 3,
        stride: int = 1,
        dilation: int = 1,
        padding: str = 'same',
        init_scale: float = 1.0,
        bias: bool = True
    ):
        super().__init__()
        if isinstance(filter_size, int):
            filter_size = (filter_size, filter_size)
        
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.filter_size = filter_size
        
        # Weight shape: [out_channels, in_channels, kH, kW]
        self.W = nn.Parameter(torch.empty(num_units, in_channels, *filter_size))
        self.use_bias = bias
        if bias:
            self.b = nn.Parameter(torch.zeros(num_units))
        default_init(init_scale)(self.W)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        if self.padding == 'same' or self.padding == 'SAME':
            # Calculate padding for 'same' mode
            pad_h = (self.filter_size[0] - 1) * self.dilation // 2
            pad_w = (self.filter_size[1] - 1) * self.dilation // 2
            padding = (pad_h, pad_w)
        else:
            padding = 0
        
        z = F.conv2d(x, self.W, bias=None, stride=self.stride, 
                     padding=padding, dilation=self.dilation)
        if self.use_bias:
            z = z + self.b.view(1, -1, 1, 1)
        return z


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Build sinusoidal embeddings (from Fairseq / tensor2tensor).
    
    Corresponds to TF's get_timestep_embedding().
    
    Args:
        timesteps: [B] tensor of timestep indices
        embedding_dim: dimension of the embedding
    
    Returns:
        [B, embedding_dim] tensor of embeddings
    """
    assert len(timesteps.shape) == 1
    
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1))
    
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


# ===== Functional versions for convenience =====

def nin(x: torch.Tensor, layer: NIN) -> torch.Tensor:
    """Functional wrapper for NIN layer."""
    return layer(x)


def dense(x: torch.Tensor, layer: Dense) -> torch.Tensor:
    """Functional wrapper for Dense layer."""
    return layer(x)


def conv2d(x: torch.Tensor, layer: Conv2d) -> torch.Tensor:
    """Functional wrapper for Conv2d layer."""
    return layer(x)
