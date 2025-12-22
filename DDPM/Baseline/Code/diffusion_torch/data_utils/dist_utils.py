"""
Distributed training utilities.

Migrated from: diffusion_tf/tpu_utils/tpu_utils.py
Migration notes:
- TPU-specific code replaced with PyTorch DDP (DistributedDataParallel)
- tf.distribute -> torch.distributed
- TPU mesh/topology -> NCCL/Gloo backend
- Cross-replica operations -> all_reduce/all_gather
"""

import os
import random
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


def setup_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
) -> bool:
    """
    Initialize distributed training.
    
    Corresponds to TF's TPU initialization.
    
    Args:
        backend: distributed backend ('nccl' for GPU, 'gloo' for CPU)
        init_method: initialization method
    
    Returns:
        True if distributed training is enabled, False otherwise
    """
    # Check if distributed environment is set up
    if "RANK" not in os.environ:
        print("Distributed environment not set up. Running in single-GPU mode.")
        return False
    
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if world_size == 1:
        return False
    
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
    
    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    return True


def cleanup_distributed():
    """
    Clean up distributed training resources.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """
    Check if current process is the main process (rank 0).
    
    Returns:
        True if main process or not in distributed mode
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """
    Get the rank of current process.
    
    Returns:
        Rank (0 if not in distributed mode)
    """
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """
    Get the total number of processes.
    
    Returns:
        World size (1 if not in distributed mode)
    """
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_local_rank() -> int:
    """
    Get the local rank of current process on this node.
    
    Returns:
        Local rank (0 if not in distributed mode)
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def seed_all(seed: int, deterministic: bool = False):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: random seed
        deterministic: if True, use deterministic algorithms (slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def wrap_model_ddp(
    model: torch.nn.Module,
    device_ids: Optional[list] = None,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    """
    Wrap model with DistributedDataParallel.
    
    Args:
        model: model to wrap
        device_ids: list of GPU device IDs (None for CPU)
        find_unused_parameters: whether to find unused parameters
    
    Returns:
        DDP-wrapped model (or original if not in distributed mode)
    """
    if not dist.is_initialized():
        return model
    
    if device_ids is None and torch.cuda.is_available():
        device_ids = [get_local_rank()]
    
    return DDP(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters,
    )


def get_distributed_dataloader(
    dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader with DistributedSampler for distributed training.
    
    Args:
        dataset: dataset to load
        batch_size: batch size per GPU
        num_workers: number of data loading workers
        shuffle: whether to shuffle (per epoch)
        drop_last: drop last incomplete batch
        pin_memory: pin memory for faster GPU transfer
    
    Returns:
        DataLoader with DistributedSampler if in distributed mode
    """
    if dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce a tensor and compute mean across all processes.
    
    Corresponds to TF's cross_replica_mean.
    
    Args:
        tensor: tensor to reduce
    
    Returns:
        Mean tensor across all processes
    """
    if not dist.is_initialized():
        return tensor
    
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce a tensor and compute sum across all processes.
    
    Args:
        tensor: tensor to reduce
    
    Returns:
        Sum tensor across all processes
    """
    if not dist.is_initialized():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def all_gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors from all processes.
    
    Args:
        tensor: tensor to gather [...]
    
    Returns:
        Gathered tensor [world_size, ...] or original if not distributed
    """
    if not dist.is_initialized():
        return tensor.unsqueeze(0)
    
    world_size = get_world_size()
    if world_size == 1:
        return tensor.unsqueeze(0)
    
    # Create output tensor list
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    
    return torch.stack(tensor_list, dim=0)


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast tensor from source process to all processes.
    
    Args:
        tensor: tensor to broadcast
        src: source rank
    
    Returns:
        Broadcasted tensor
    """
    if not dist.is_initialized():
        return tensor
    
    dist.broadcast(tensor, src=src)
    return tensor


def barrier():
    """
    Synchronize all processes.
    """
    if dist.is_initialized():
        dist.barrier()


def print_once(*args, **kwargs):
    """
    Print only on main process.
    """
    if is_main_process():
        print(*args, **kwargs)
