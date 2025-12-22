"""
Training and sampling scripts for diffusion models.

Migrated from: diffusion_tf/scripts/
Migration notes:
- TF training loops -> PyTorch training loops
- TPU training -> DDP multi-GPU training
- tf.train.Checkpoint -> torch.save/load
"""

from .run_cifar import main as run_cifar_main
from .run_cifar import train as train_cifar
from .run_cifar import sample as sample_cifar

__all__ = [
    "run_cifar_main",
    "train_cifar",
    "sample_cifar",
]
