"""
PyTorch implementation of DDPM (Denoising Diffusion Probabilistic Models)

Migrated from: diffusion_tf/__init__.py
Migration notes:
- This is the PyTorch version of the diffusion package
- All tensors use NCHW format (unlike TF's NHWC)
- Data range: [0, 255] for dataset output, [-1, 1] for model input/output
"""

from . import nn
from . import utils
from . import diffusion_utils

__version__ = "1.0.0"
__all__ = ["nn", "utils", "diffusion_utils"]
