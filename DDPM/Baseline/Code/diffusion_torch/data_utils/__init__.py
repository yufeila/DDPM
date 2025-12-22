"""
Data utilities package for diffusion models.

Migrated from: diffusion_tf/tpu_utils/__init__.py
Migration notes:
- TPU-specific code replaced with PyTorch DDP equivalents
- This package provides dataset loading, metrics, and distributed training utilities
"""

from .datasets import (
    get_dataset,
    get_dataloader,
    SimpleDataset,
    LsunDataset,
    DATASETS,
    normalize_data,
    unnormalize_data,
)
from .metrics_numpy import (
    classifier_score_from_logits,
    frechet_classifier_distance_from_activations,
    compute_statistics_from_activations,
    frechet_distance_from_statistics,
)
from .dist_utils import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_world_size,
    get_rank,
    get_local_rank,
    seed_all,
    wrap_model_ddp,
    get_distributed_dataloader,
    all_reduce_mean,
    all_reduce_sum,
    all_gather_tensor,
    broadcast_tensor,
    barrier,
    print_once,
)
from .summaries import (
    DistributedSummaryWriter,
    ScalarTracker,
)
from .eval_worker import (
    CheckpointPoller,
    EvalWorker,
    extract_step_from_path,
    load_samples_from_npz,
    compute_fid_from_samples,
)

__all__ = [
    # Datasets
    "get_dataset",
    "get_dataloader",
    "SimpleDataset",
    "LsunDataset",
    "DATASETS",
    "normalize_data",
    "unnormalize_data",
    # Metrics
    "classifier_score_from_logits",
    "frechet_classifier_distance_from_activations",
    "compute_statistics_from_activations",
    "frechet_distance_from_statistics",
    # Distributed
    "setup_distributed",
    "cleanup_distributed",
    "is_main_process",
    "get_world_size",
    "get_rank",
    "get_local_rank",
    "seed_all",
    "wrap_model_ddp",
    "get_distributed_dataloader",
    "all_reduce_mean",
    "all_reduce_sum",
    "all_gather_tensor",
    "broadcast_tensor",
    "barrier",
    "print_once",
    # Summaries
    "DistributedSummaryWriter",
    "ScalarTracker",
    # Eval worker
    "CheckpointPoller",
    "EvalWorker",
    "extract_step_from_path",
    "load_samples_from_npz",
    "compute_fid_from_samples",
]
