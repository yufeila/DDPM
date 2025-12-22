"""
TensorBoard summary utilities for distributed training.

Migrated from: diffusion_tf/tpu_utils/tpu_summaries.py
Migration notes:
- TF's tf.summary -> torch.utils.tensorboard.SummaryWriter
- TPU-specific summary aggregation -> DDP-compatible summaries
- Only main process writes summaries
"""

import os
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .dist_utils import is_main_process, all_reduce_mean


class DistributedSummaryWriter:
    """
    TensorBoard SummaryWriter wrapper for distributed training.
    
    Corresponds to TF's TpuSummaries.
    
    Only writes summaries on the main process.
    Supports aggregating scalars across all processes before logging.
    """
    
    def __init__(
        self,
        log_dir: str,
        enabled: bool = True,
        flush_secs: int = 120,
    ):
        """
        Initialize the summary writer.
        
        Args:
            log_dir: directory for TensorBoard logs
            enabled: whether to enable writing
            flush_secs: how often to flush to disk
        """
        self.log_dir = log_dir
        self.enabled = enabled and is_main_process()
        
        if self.enabled:
            os.makedirs(log_dir, exist_ok=True)
            self._writer = SummaryWriter(
                log_dir=log_dir,
                flush_secs=flush_secs,
            )
        else:
            self._writer = None
        
        # Buffer for accumulating scalars
        self._scalar_buffer: Dict[str, list] = {}
    
    def add_scalar(
        self,
        tag: str,
        scalar_value: Union[float, torch.Tensor],
        global_step: int,
        reduce: bool = False,
    ):
        """
        Add a scalar to TensorBoard.
        
        Args:
            tag: data identifier
            scalar_value: scalar value to log
            global_step: global step value
            reduce: if True, reduce across all processes before logging
        """
        if not self.enabled:
            return
        
        if isinstance(scalar_value, torch.Tensor):
            if reduce:
                scalar_value = all_reduce_mean(scalar_value)
            scalar_value = scalar_value.item()
        
        self._writer.add_scalar(tag, scalar_value, global_step)
    
    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, Union[float, torch.Tensor]],
        global_step: int,
    ):
        """
        Add multiple scalars under a common main tag.
        
        Args:
            main_tag: parent tag
            tag_scalar_dict: dict of tag -> scalar value
            global_step: global step value
        """
        if not self.enabled:
            return
        
        # Convert tensors to floats
        converted = {}
        for tag, value in tag_scalar_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            converted[tag] = value
        
        self._writer.add_scalars(main_tag, converted, global_step)
    
    def add_image(
        self,
        tag: str,
        img_tensor: Union[np.ndarray, torch.Tensor],
        global_step: int,
        dataformats: str = "CHW",
    ):
        """
        Add an image to TensorBoard.
        
        Args:
            tag: data identifier
            img_tensor: image tensor (C, H, W) or (H, W, C)
            global_step: global step value
            dataformats: format of img_tensor ('CHW' or 'HWC')
        """
        if not self.enabled:
            return
        
        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.detach().cpu()
        
        self._writer.add_image(tag, img_tensor, global_step, dataformats=dataformats)
    
    def add_images(
        self,
        tag: str,
        img_tensor: Union[np.ndarray, torch.Tensor],
        global_step: int,
        dataformats: str = "NCHW",
    ):
        """
        Add multiple images to TensorBoard.
        
        Args:
            tag: data identifier
            img_tensor: image tensor (N, C, H, W) or (N, H, W, C)
            global_step: global step value
            dataformats: format of img_tensor ('NCHW' or 'NHWC')
        """
        if not self.enabled:
            return
        
        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.detach().cpu()
        
        self._writer.add_images(tag, img_tensor, global_step, dataformats=dataformats)
    
    def add_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, torch.Tensor],
        global_step: int,
        bins: str = "tensorflow",
    ):
        """
        Add a histogram to TensorBoard.
        
        Args:
            tag: data identifier
            values: values for histogram
            global_step: global step value
            bins: binning strategy
        """
        if not self.enabled:
            return
        
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu()
        
        self._writer.add_histogram(tag, values, global_step, bins=bins)
    
    def add_text(self, tag: str, text_string: str, global_step: int):
        """
        Add text to TensorBoard.
        
        Args:
            tag: data identifier
            text_string: text to log
            global_step: global step value
        """
        if not self.enabled:
            return
        
        self._writer.add_text(tag, text_string, global_step)
    
    def add_graph(self, model: torch.nn.Module, input_to_model: torch.Tensor):
        """
        Add model graph to TensorBoard.
        
        Args:
            model: PyTorch model
            input_to_model: example input
        """
        if not self.enabled:
            return
        
        self._writer.add_graph(model, input_to_model)
    
    def flush(self):
        """Flush pending events to disk."""
        if self._writer is not None:
            self._writer.flush()
    
    def close(self):
        """Close the summary writer."""
        if self._writer is not None:
            self._writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ScalarTracker:
    """
    Track and aggregate scalar values over time.
    
    Useful for computing epoch-level or evaluation metrics.
    """
    
    def __init__(self):
        self._values: Dict[str, list] = {}
    
    def add(self, tag: str, value: Union[float, torch.Tensor]):
        """
        Add a value to track.
        
        Args:
            tag: metric name
            value: scalar value
        """
        if isinstance(value, torch.Tensor):
            value = value.item()
        
        if tag not in self._values:
            self._values[tag] = []
        self._values[tag].append(value)
    
    def mean(self, tag: str) -> float:
        """
        Get mean of tracked values.
        
        Args:
            tag: metric name
        
        Returns:
            Mean value
        """
        if tag not in self._values or len(self._values[tag]) == 0:
            return 0.0
        return sum(self._values[tag]) / len(self._values[tag])
    
    def sum(self, tag: str) -> float:
        """
        Get sum of tracked values.
        
        Args:
            tag: metric name
        
        Returns:
            Sum value
        """
        if tag not in self._values:
            return 0.0
        return sum(self._values[tag])
    
    def count(self, tag: str) -> int:
        """
        Get count of tracked values.
        
        Args:
            tag: metric name
        
        Returns:
            Count
        """
        if tag not in self._values:
            return 0
        return len(self._values[tag])
    
    def get_all_means(self) -> Dict[str, float]:
        """
        Get means of all tracked metrics.
        
        Returns:
            Dict of tag -> mean value
        """
        return {tag: self.mean(tag) for tag in self._values}
    
    def reset(self, tag: Optional[str] = None):
        """
        Reset tracked values.
        
        Args:
            tag: metric name to reset (None to reset all)
        """
        if tag is None:
            self._values = {}
        elif tag in self._values:
            self._values[tag] = []
    
    def log_to_writer(
        self,
        writer: DistributedSummaryWriter,
        global_step: int,
        prefix: str = "",
        reset: bool = True,
    ):
        """
        Log all tracked means to a SummaryWriter.
        
        Args:
            writer: DistributedSummaryWriter instance
            global_step: global step value
            prefix: prefix for all tags
            reset: whether to reset after logging
        """
        for tag, values in self._values.items():
            if len(values) > 0:
                mean_value = sum(values) / len(values)
                full_tag = f"{prefix}/{tag}" if prefix else tag
                writer.add_scalar(full_tag, mean_value, global_step)
        
        if reset:
            self.reset()
