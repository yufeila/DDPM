"""
UNet model for diffusion models.

Migrated from: diffusion_tf/models/unet.py
Migration notes:
- TF uses NHWC, PyTorch uses NCHW
- tf.variable_scope -> nn.Module classes
- All convolutions operate on [B, C, H, W]
- group_norm: TF groups on last dim (channels), PyTorch same but channels first
- TF model() function -> UNet class with forward()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from .. import nn as diffusion_nn


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    """Swish activation function."""
    return F.silu(x)


def normalize(x: torch.Tensor, num_groups: int = 32) -> torch.Tensor:
    """Group normalization. Input: [B, C, H, W]."""
    return F.group_norm(x, num_groups=num_groups)


class GroupNorm32(nn.GroupNorm):
    """GroupNorm with 32 groups by default."""
    def __init__(self, num_channels: int):
        super().__init__(num_groups=32, num_channels=num_channels)


class Upsample(nn.Module):
    """
    Upsampling layer with optional convolution.
    Corresponds to TF's upsample().
    Input: [B, C, H, W] -> Output: [B, C, H*2, W*2]
    """
    def __init__(self, channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = diffusion_nn.Conv2d(channels, channels, filter_size=3, stride=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        assert x.shape == (B, C, H * 2, W * 2)
        return x


class Downsample(nn.Module):
    """
    Downsampling layer with optional convolution.
    Corresponds to TF's downsample().
    Input: [B, C, H, W] -> Output: [B, C, H//2, W//2]
    """
    def __init__(self, channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = diffusion_nn.Conv2d(channels, channels, filter_size=3, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.with_conv:
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, 2, 2)
        assert x.shape == (B, C, H // 2, W // 2)
        return x


class ResnetBlock(nn.Module):
    """
    Residual block with timestep embedding.
    Corresponds to TF's resnet_block().
    Input: [B, C, H, W], temb: [B, temb_dim]
    """
    def __init__(self, in_ch: int, out_ch: int = None, temb_dim: int = None,
                 conv_shortcut: bool = False, dropout: float = 0.0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch if out_ch is not None else in_ch
        
        self.norm1 = GroupNorm32(in_ch)
        self.conv1 = diffusion_nn.Conv2d(in_ch, self.out_ch, filter_size=3)
        
        if temb_dim is not None:
            self.temb_proj = diffusion_nn.Dense(temb_dim, self.out_ch)
        
        self.norm2 = GroupNorm32(self.out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = diffusion_nn.Conv2d(self.out_ch, self.out_ch, filter_size=3, init_scale=0.)
        
        if in_ch != self.out_ch:
            if conv_shortcut:
                self.shortcut = diffusion_nn.Conv2d(in_ch, self.out_ch, filter_size=3)
            else:
                self.shortcut = diffusion_nn.NIN(in_ch, self.out_ch)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        h = x
        h = nonlinearity(self.norm1(h))
        h = self.conv1(h)
        
        # Add timestep embedding
        if temb is not None and hasattr(self, 'temb_proj'):
            temb_out = self.temb_proj(nonlinearity(temb))
            # temb_out: [B, out_ch] -> [B, out_ch, 1, 1] for NCHW
            h = h + temb_out[:, :, None, None]
        
        h = nonlinearity(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        
        x = self.shortcut(x)
        
        assert x.shape == h.shape
        return x + h


class AttnBlock(nn.Module):
    """
    Self-attention block.
    Corresponds to TF's attn_block().
    Input: [B, C, H, W]
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = GroupNorm32(channels)
        self.q = diffusion_nn.NIN(channels, channels)
        self.k = diffusion_nn.NIN(channels, channels)
        self.v = diffusion_nn.NIN(channels, channels)
        self.proj_out = diffusion_nn.NIN(channels, channels, init_scale=0.)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        h = self.norm(x)
        # Reshape for NIN: [B, C, H, W] -> [B, H, W, C]
        h = h.permute(0, 2, 3, 1)
        
        q = self.q(h)  # [B, H, W, C]
        k = self.k(h)
        v = self.v(h)
        
        # Attention: q @ k^T / sqrt(C)
        # [B, H, W, C] @ [B, H, W, C]^T -> [B, H, W, H, W]
        w = torch.einsum('bhwc,bHWc->bhwHW', q, k) * (C ** (-0.5))
        w = w.reshape(B, H, W, H * W)
        w = F.softmax(w, dim=-1)
        w = w.reshape(B, H, W, H, W)
        
        # Apply attention to values
        h = torch.einsum('bhwHW,bHWc->bhwc', w, v)
        h = self.proj_out(h)
        
        # Reshape back: [B, H, W, C] -> [B, C, H, W]
        h = h.permute(0, 3, 1, 2)
        
        assert h.shape == x.shape
        return x + h


class UNet(nn.Module):
    """
    UNet model for diffusion.
    
    Corresponds to TF's model() function.
    
    Args:
        in_ch: input channels (default 3 for RGB)
        out_ch: output channels (default 3)
        ch: base channel count
        ch_mult: channel multipliers for each resolution level
        num_res_blocks: number of residual blocks per level
        attn_resolutions: resolutions at which to apply attention
        dropout: dropout rate
        resamp_with_conv: use convolution for up/downsampling
    
    Input: x [B, C, H, W], t [B]
    Output: [B, out_ch, H, W]
    """
    
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions: Tuple[int, ...] = (16,),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
    ):
        super().__init__()
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.num_resolutions = len(ch_mult)
        temb_dim = ch * 4
        
        # Timestep embedding
        self.temb_dense0 = diffusion_nn.Dense(ch, temb_dim)
        self.temb_dense1 = diffusion_nn.Dense(temb_dim, temb_dim)
        
        # Initial convolution
        self.conv_in = diffusion_nn.Conv2d(in_ch, ch, filter_size=3)
        
        # Downsampling
        self.down_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        curr_res = None  # Will be set based on input
        in_ch_block = ch
        self.down_block_chans = [ch]
        
        for i_level in range(self.num_resolutions):
            out_ch_block = ch * ch_mult[i_level]
            
            for i_block in range(num_res_blocks):
                self.down_blocks.append(
                    ResnetBlock(in_ch_block, out_ch_block, temb_dim=temb_dim, dropout=dropout)
                )
                in_ch_block = out_ch_block
                self.down_block_chans.append(in_ch_block)
            
            # Placeholder for attention (will check resolution at forward time)
            self.down_attn.append(AttnBlock(in_ch_block))
            
            if i_level != self.num_resolutions - 1:
                self.down_samples.append(Downsample(in_ch_block, with_conv=resamp_with_conv))
                self.down_block_chans.append(in_ch_block)
        
        # Middle
        self.mid_block1 = ResnetBlock(in_ch_block, in_ch_block, temb_dim=temb_dim, dropout=dropout)
        self.mid_attn = AttnBlock(in_ch_block)
        self.mid_block2 = ResnetBlock(in_ch_block, in_ch_block, temb_dim=temb_dim, dropout=dropout)
        
        # Upsampling
        self.up_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for i_level in reversed(range(self.num_resolutions)):
            out_ch_block = ch * ch_mult[i_level]
            
            for i_block in range(num_res_blocks + 1):
                skip_ch = self.down_block_chans.pop()
                self.up_blocks.append(
                    ResnetBlock(in_ch_block + skip_ch, out_ch_block, temb_dim=temb_dim, dropout=dropout)
                )
                in_ch_block = out_ch_block
            
            # Placeholder for attention
            self.up_attn.append(AttnBlock(in_ch_block))
            
            if i_level != 0:
                self.up_samples.append(Upsample(in_ch_block, with_conv=resamp_with_conv))
        
        # Output
        self.norm_out = GroupNorm32(in_ch_block)
        self.conv_out = diffusion_nn.Conv2d(in_ch_block, out_ch, filter_size=3, init_scale=0.)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, C, H, W] input images (in [-1, 1])
            t: [B] timestep indices
            y: [B] class labels (not used, for API compatibility)
        
        Returns:
            [B, out_ch, H, W] predicted noise
        """
        assert x.dtype == torch.float32
        B, C, H, W = x.shape
        
        # Timestep embedding
        temb = diffusion_nn.get_timestep_embedding(t, self.ch)
        temb = self.temb_dense0(temb)
        temb = nonlinearity(temb)
        temb = self.temb_dense1(temb)
        assert temb.shape == (B, self.ch * 4)
        
        # Initial conv
        h = self.conv_in(x)
        hs = [h]
        
        # Downsampling
        block_idx = 0
        attn_idx = 0
        sample_idx = 0
        
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down_blocks[block_idx](h, temb)
                block_idx += 1
                # Apply attention if resolution matches
                if h.shape[2] in self.attn_resolutions:
                    h = self.down_attn[attn_idx](h)
                hs.append(h)
            attn_idx += 1
            
            if i_level != self.num_resolutions - 1:
                h = self.down_samples[sample_idx](h)
                sample_idx += 1
                hs.append(h)
        
        # Middle
        h = self.mid_block1(h, temb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb)
        
        # Upsampling
        block_idx = 0
        attn_idx = 0
        sample_idx = 0
        
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = torch.cat([h, hs.pop()], dim=1)
                h = self.up_blocks[block_idx](h, temb)
                block_idx += 1
                # Apply attention if resolution matches
                if h.shape[2] in self.attn_resolutions:
                    h = self.up_attn[attn_idx](h)
            attn_idx += 1
            
            if i_level != 0:
                h = self.up_samples[sample_idx](h)
                sample_idx += 1
        
        assert len(hs) == 0
        
        # Output
        h = nonlinearity(self.norm_out(h))
        h = self.conv_out(h)
        assert h.shape == (B, self.conv_out.W.shape[0], H, W)
        return h
