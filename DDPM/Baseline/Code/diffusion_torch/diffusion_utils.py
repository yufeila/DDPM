"""
Gaussian diffusion utilities for DDPM.

Migrated from: diffusion_tf/diffusion_utils.py
Migration notes:
- TF uses NHWC, PyTorch uses NCHW
- tf.constant -> torch.tensor (registered as buffer)
- tf.while_loop -> Python for loop (eager execution)
- All shapes: [B, C, H, W] instead of [B, H, W, C]
- _extract broadcasts to [B, 1, 1, 1] for NCHW
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, Optional, Tuple, Union

from . import nn as diffusion_nn

# kl divergence between two normal distributions
def normal_kl(mean1: torch.Tensor, logvar1: torch.Tensor, 
              mean2: torch.Tensor, logvar2: torch.Tensor) -> torch.Tensor:
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    Corresponds to TF's normal_kl().
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                  + (mean1 - mean2) ** 2 * torch.exp(-logvar2))

# Linear warmup for beta schedule
def _warmup_beta(beta_start: float, beta_end: float, 
                 num_diffusion_timesteps: int, warmup_frac: float) -> np.ndarray:
    """Linear warmup for beta schedule."""
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(beta_schedule: str, *, beta_start: float, beta_end: float, 
                      num_diffusion_timesteps: int) -> np.ndarray:
    """
    Get beta schedule for diffusion process.
    
    Corresponds to TF's get_beta_schedule().
    
    Args:
        beta_schedule: type of schedule ('quad', 'linear', 'warmup10', 'warmup50', 'const', 'jsd')
        beta_start: starting beta value
        beta_end: ending beta value
        num_diffusion_timesteps: number of diffusion steps
    
    Returns:
        numpy array of beta values with shape [num_diffusion_timesteps]
    """
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, 
                           num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'warmup10':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == 'warmup50':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(num_diffusion_timesteps, 1, 
                                 num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

# Generate noise tensor
def noise_like(shape: Tuple[int, ...], noise_fn: Callable = torch.randn, 
               repeat: bool = False, device: torch.device = None) -> torch.Tensor:
    """
    Generate noise tensor.
    
    Corresponds to TF's noise_like().
    
    Args:
        shape: shape of noise tensor [B, C, H, W]
        noise_fn: function to generate noise (default: torch.randn)
        repeat: if True, use same noise for all batch elements
        device: torch device
    """
    if repeat:
        noise = noise_fn((1, *shape[1:]), device=device)
        return noise.repeat(shape[0], 1, 1, 1)
    else:
        return noise_fn(shape, device=device)


class GaussianDiffusion:
    """
    Contains utilities for the diffusion model.
    
    Corresponds to TF's GaussianDiffusion class.
    
    Note: All image tensors are in NCHW format [B, C, H, W].
    """
    
    def __init__(self, *, betas: np.ndarray, loss_type: str, device: torch.device = None):
        """
        Initialize diffusion utilities.
        
        Args:
            betas: numpy array of beta values
            loss_type: type of loss ('noisepred')
            device: torch device (default: cuda if available)
        """
        self.loss_type = loss_type
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        assert alphas_cumprod_prev.shape == (timesteps,)
        
        # Convert to torch tensors
        self.betas = torch.tensor(betas, dtype=torch.float32, device=self.device)
        self.alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32, device=self.device)
        self.alphas_cumprod_prev = torch.tensor(alphas_cumprod_prev, dtype=torch.float32, device=self.device)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.tensor(
            np.sqrt(alphas_cumprod), dtype=torch.float32, device=self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.tensor(
            np.sqrt(1. - alphas_cumprod), dtype=torch.float32, device=self.device)
        self.log_one_minus_alphas_cumprod = torch.tensor(
            np.log(1. - alphas_cumprod), dtype=torch.float32, device=self.device)
        self.sqrt_recip_alphas_cumprod = torch.tensor(
            np.sqrt(1. / alphas_cumprod), dtype=torch.float32, device=self.device)
        self.sqrt_recipm1_alphas_cumprod = torch.tensor(
            np.sqrt(1. / alphas_cumprod - 1), dtype=torch.float32, device=self.device)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_variance = torch.tensor(
            posterior_variance, dtype=torch.float32, device=self.device)
        # Log calculation clipped because posterior variance is 0 at beginning
        self.posterior_log_variance_clipped = torch.tensor(
            np.log(np.maximum(posterior_variance, 1e-20)), dtype=torch.float32, device=self.device)
        self.posterior_mean_coef1 = torch.tensor(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod), 
            dtype=torch.float32, device=self.device)
        self.posterior_mean_coef2 = torch.tensor(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod),
            dtype=torch.float32, device=self.device)
    
    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Extract coefficients at specified timesteps, then reshape for broadcasting.
        
        For NCHW format, reshapes to [batch_size, 1, 1, 1].
        
        Corresponds to TF's _extract().
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = a.gather(0, t)
        assert out.shape == (bs,)
        # Reshape to [B, 1, 1, 1] for NCHW broadcasting
        return out.view(bs, *([1] * (len(x_shape) - 1)))
    
    def q_mean_variance(self, x_start: torch.Tensor, t: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the distribution q(x_t | x_0).
        
        Corresponds to TF's q_mean_variance().
        """
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, 
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Diffuse the data (t == 0 means diffused for 1 step).
        
        Corresponds to TF's q_sample().
        
        Args:
            x_start: [B, C, H, W] original images
            t: [B] timestep indices
            noise: optional noise tensor
        
        Returns:
            x_t: [B, C, H, W] noised images
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, 
                                  noise: torch.Tensor) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise.
        
        Corresponds to TF's predict_start_from_noise().
        """
        assert x_t.shape == noise.shape
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0).
        
        Corresponds to TF's q_posterior().
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == 
                posterior_log_variance_clipped.shape[0] == x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_losses(self, denoise_fn: Callable, x_start: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Training loss calculation.
        
        Corresponds to TF's p_losses().
        
        Args:
            denoise_fn: denoising model function (x, t) -> predicted_noise
            x_start: [B, C, H, W] original images in [-1, 1]
            t: [B] timestep indices
            noise: optional noise tensor
        
        Returns:
            losses: [B] per-sample losses
        """
        B, C, H, W = x_start.shape
        assert t.shape == (B,)
        
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape and noise.dtype == x_start.dtype
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = denoise_fn(x_noisy, t)
        assert x_noisy.shape == x_start.shape
        assert x_recon.shape[:2] == (B, C) and len(x_recon.shape) == 4
        
        if self.loss_type == 'noisepred':
            # Predict the noise instead of x_start
            assert x_recon.shape == x_start.shape
            losses = diffusion_nn.meanflat((noise - x_recon) ** 2)
        else:
            raise NotImplementedError(self.loss_type)
        
        assert losses.shape == (B,)
        return losses
    
    def p_mean_variance(self, denoise_fn: Callable, *, x: torch.Tensor, t: torch.Tensor,
                        clip_denoised: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance for p(x_{t-1} | x_t).
        
        Corresponds to TF's p_mean_variance().
        """
        if self.loss_type == 'noisepred':
            x_recon = self.predict_start_from_noise(x, t=t, noise=denoise_fn(x, t))
        else:
            raise NotImplementedError(self.loss_type)
        
        if clip_denoised:
            x_recon = torch.clamp(x_recon, -1., 1.)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        assert model_mean.shape == x_recon.shape == x.shape
        assert posterior_variance.shape == posterior_log_variance.shape == (x.shape[0], 1, 1, 1)
        return model_mean, posterior_variance, posterior_log_variance
    
    def p_sample(self, denoise_fn: Callable, *, x: torch.Tensor, t: torch.Tensor,
                 clip_denoised: bool = True, repeat_noise: bool = False) -> torch.Tensor:
        """
        Sample from the model (one step of reverse diffusion).
        
        Corresponds to TF's p_sample().
        """
        model_mean, _, model_log_variance = self.p_mean_variance(
            denoise_fn, x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device=x.device, repeat=repeat_noise)
        assert noise.shape == x.shape
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, denoise_fn: Callable, *, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Generate samples by running the full reverse diffusion process.
        
        Corresponds to TF's p_sample_loop().
        
        Args:
            denoise_fn: denoising model function
            shape: shape of samples to generate [B, C, H, W]
        
        Returns:
            samples: [B, C, H, W] generated images
        """
        assert isinstance(shape, (tuple, list))
        img = torch.randn(shape, device=self.device)
        
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            img = self.p_sample(denoise_fn=denoise_fn, x=img, t=t)
        
        assert img.shape == shape
        return img
    
    @torch.no_grad()
    def p_sample_loop_trajectory(self, denoise_fn: Callable, *, shape: Tuple[int, ...],
                                  repeat_noise_steps: int = -1
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate samples, returning intermediate images.
        Useful for visualizing how denoised images evolve over time.
        
        Corresponds to TF's p_sample_loop_trajectory().
        
        Args:
            denoise_fn: denoising model function
            shape: shape of samples [B, C, H, W]
            repeat_noise_steps: number of steps to use repeated noise (-1 for none)
        
        Returns:
            times: [T+1] timestep values
            imgs: [T+1, B, C, H, W] images at each timestep
        """
        assert isinstance(shape, (tuple, list))
        
        # Initial noise
        use_repeat_noise = repeat_noise_steps >= 0
        img = noise_like(shape, device=self.device, repeat=use_repeat_noise)
        
        times = [self.num_timesteps - 1]
        imgs = [img]
        
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            # Use repeated noise for first few steps if specified
            repeat_noise = use_repeat_noise and (self.num_timesteps - i <= repeat_noise_steps)
            img = self.p_sample(denoise_fn=denoise_fn, x=img, t=t, repeat_noise=repeat_noise)
            times.append(i - 1)
            imgs.append(img)
        
        times = torch.tensor(times, device=self.device)
        imgs = torch.stack(imgs, dim=0)
        
        assert imgs[-1].shape == shape
        return times, imgs
