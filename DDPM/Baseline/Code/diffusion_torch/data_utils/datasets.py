"""
Dataset loading utilities.

Migrated from: diffusion_tf/tpu_utils/datasets.py
Migration notes:
- tfds -> torchvision.datasets
- TF data pipeline -> torch.utils.data.DataLoader
- Output format: dict with 'image' and 'label' keys
- Images are in NCHW format [B, C, H, W], pixel values in [0, 255] as int32
- TF uses NHWC, PyTorch uses NCHW
"""

import functools
import os
from typing import Dict, Optional, Tuple, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms


def pack(image: torch.Tensor, label: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Pack image and label into a dictionary.
    Corresponds to TF's pack().
    """
    return {'image': image, 'label': label.int()}


class SimpleDataset(Dataset):
    """
    Simple dataset wrapper for CIFAR10 and CelebA-HQ.
    Corresponds to TF's SimpleDataset.
    
    Output images are in NCHW format with pixel values in [0, 255] as int32.
    """
    DATASET_NAMES = ('cifar10', 'celebahq256')
    
    def __init__(self, name: str, data_dir: str = None, train: bool = True):
        """
        Args:
            name: dataset name ('cifar10' or 'celebahq256')
            data_dir: directory for dataset storage
            train: if True, use training split
        """
        self._name = name
        self._data_dir = data_dir or './data'
        self._train = train
        
        self._img_size = {'cifar10': 32, 'celebahq256': 256}[name]
        self._img_shape = (3, self._img_size, self._img_size)  # NCHW
        
        self.num_train_examples, self.num_eval_examples = {
            'cifar10': (50000, 10000),
            'celebahq256': (30000, 0),
        }[name]
        self.num_classes = 1  # unconditional
        
        # Load dataset
        if name == 'cifar10':
            self._dataset = torchvision.datasets.CIFAR10(
                root=self._data_dir,
                train=train,
                download=True,
                transform=None  # We handle transforms manually
            )
        elif name == 'celebahq256':
            # CelebA-HQ needs to be downloaded separately
            # Using ImageFolder for custom dataset
            split = 'train' if train else 'val'
            celebahq_path = os.path.join(self._data_dir, 'celebahq256', split)
            if os.path.exists(celebahq_path):
                self._dataset = torchvision.datasets.ImageFolder(
                    root=celebahq_path,
                    transform=transforms.Resize((256, 256))
                )
            else:
                # Fallback: create dummy dataset for API compatibility
                print(f"Warning: CelebA-HQ not found at {celebahq_path}")
                self._dataset = None
        else:
            raise ValueError(f"Unknown dataset: {name}")
    
    @property
    def image_shape(self) -> Tuple[int, int, int]:
        """Returns image shape in NCHW format (C, H, W)."""
        return tuple(self._img_shape)
    
    def __len__(self) -> int:
        if self._dataset is None:
            return 0
        return len(self._dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            dict with 'image' [C, H, W] int32 in [0, 255] and 'label' int32
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded")
        
        img, label = self._dataset[idx]
        
        # Convert PIL Image to tensor
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(np.array(img))
        
        # Ensure NCHW format
        if img.ndim == 3 and img.shape[-1] == 3:
            # HWC -> CHW
            img = img.permute(2, 0, 1)
        
        # Convert to int32 with values in [0, 255]
        img = img.to(torch.int32)
        label = torch.tensor(0, dtype=torch.int32)  # unconditional, always 0
        
        return pack(img, label)


class LsunDataset(Dataset):
    """
    LSUN dataset.
    Corresponds to TF's LsunDataset.
    
    Note: Requires LSUN data to be prepared as image folder.
    Output images are in NCHW format with pixel values in [0, 255] as int32.
    """
    
    def __init__(
        self,
        data_dir: str,
        resolution: int = 256,
        max_images: Optional[int] = None,
    ):
        """
        Args:
            data_dir: path to LSUN image folder
            resolution: image resolution
            max_images: maximum number of images to use (None for all)
        """
        self.data_dir = data_dir
        self.resolution = resolution
        self.max_images = max_images
        self.num_classes = 1  # unconditional
        self.image_shape = (3, resolution, resolution)  # NCHW
        
        # Load using ImageFolder
        transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
        ])
        
        if os.path.exists(data_dir):
            self._dataset = torchvision.datasets.ImageFolder(
                root=data_dir,
                transform=transform
            )
            if max_images is not None:
                # Limit dataset size
                indices = list(range(min(max_images, len(self._dataset))))
                self._dataset = torch.utils.data.Subset(self._dataset, indices)
        else:
            print(f"Warning: LSUN data not found at {data_dir}")
            self._dataset = None
    
    def __len__(self) -> int:
        if self._dataset is None:
            return 0
        return len(self._dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            dict with 'image' [C, H, W] int32 in [0, 255] and 'label' int32
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded")
        
        img, _ = self._dataset[idx]
        
        # img is already tensor from ToTensor(), in [0, 1]
        # Convert to [0, 255] int32
        img = (img * 255).to(torch.int32)
        label = torch.tensor(0, dtype=torch.int32)  # unconditional
        
        return pack(img, label)


# Dataset registry
DATASETS = {
    "cifar10": functools.partial(SimpleDataset, name="cifar10"),
    "celebahq256": functools.partial(SimpleDataset, name="celebahq256"),
    "lsun": LsunDataset,
}


def get_dataset(
    name: str,
    *,
    data_dir: str = None,
    train: bool = True,
) -> Dataset:
    """
    Instantiate a dataset.
    
    Corresponds to TF's get_dataset().
    
    Args:
        name: dataset name ('cifar10', 'celebahq256', 'lsun')
        data_dir: directory for dataset storage
        train: if True, use training split
    
    Returns:
        Dataset instance
    """
    if name not in DATASETS:
        raise ValueError(f"Dataset {name} is not available. Choose from {list(DATASETS.keys())}")
    
    if name == 'lsun':
        if data_dir is None:
            raise ValueError("data_dir is required for LSUN dataset")
        return DATASETS[name](data_dir=data_dir)
    else:
        return DATASETS[name](data_dir=data_dir, train=train)


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    drop_last: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader from a dataset.
    
    Args:
        dataset: Dataset instance
        batch_size: batch size
        shuffle: whether to shuffle
        num_workers: number of data loading workers
        drop_last: drop last incomplete batch
        pin_memory: pin memory for faster GPU transfer
    
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )


# ===== Data normalization utilities =====

def normalize_data(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize image data from [0, 255] to [-1, 1].
    Corresponds to TF's normalize_data.
    
    Args:
        x: image tensor with values in [0, 255]
    
    Returns:
        normalized tensor with values in [-1, 1]
    """
    return x.float() / 127.5 - 1.0


def unnormalize_data(x: torch.Tensor) -> torch.Tensor:
    """
    Unnormalize image data from [-1, 1] to [0, 255].
    Corresponds to TF's unnormalize_data.
    
    Args:
        x: normalized tensor with values in [-1, 1]
    
    Returns:
        tensor with values in [0, 255]
    """
    return (x + 1.0) * 127.5
