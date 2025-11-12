import logging
import os
import re
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Tuple, Union
from tqdm import tqdm
import numpy as np
import torch

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Subset

from packages.data_objects.dataset import RANDOM_SEED, TorchDataset

logging.basicConfig(level=logging.INFO)


def _get_set_sizes(sets_size : Dict[str, float], dataset, indices):
    assert all(k in ["train", "val", "test"] for k in sets_size.keys()), "sets_size keys must be at least 'train', 'val'"
    train_size = int(sets_size["train"] * len(dataset))
    val_size = int(sets_size["val"] * len(dataset))
    if "test" not in sets_size:
        test_size = len(dataset) - train_size - val_size
    else:
        test_size = int(sets_size["test"] * len(dataset))

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]

    if "test" not in sets_size:
        test_idx = indices[train_size + val_size :]
    else:
        test_idx = indices[train_size + val_size : train_size + val_size + test_size]

    return train_idx, val_idx, test_idx

def _calc_norm_params(
    train_loader: DataLoader, 
    axes: Tuple[int], 
    target_axes: Tuple[int] = None,
    norm_params: Tuple[float, float] | Tuple[torch.Tensor, torch.Tensor] | None = None,
    target_norm_params: Tuple[float, float] | Tuple[torch.Tensor, torch.Tensor] | None = None
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor] | None]:
    """
    Calculate global mean and std using batched Welford's algorithm.
    Faster and more memory efficient. Calculates both input and target in single pass.
    
    Args:
        train_loader: DataLoader for training data
        axes: Dimensions to calculate mean/std across for 'input' (note: tensor shape is [batch, ...])
        target_axes: Dimensions to calculate mean/std across for 'target'. If None, target not normalized
        norm_params: Pre-computed (mean, std) for input. If None, calculate from data
        target_norm_params: Pre-computed (mean, std) for target. If None, calculate from data
    
    Returns:
        ((input_mean, input_std), (target_mean, target_std) or None)
    """
    sample_batch = next(iter(train_loader))

    if not isinstance(sample_batch, dict):
        raise ValueError("Expected batch to be a dict with 'input' key.")
    
    assert 'input' in sample_batch, "Expected batch dict to contain 'input' key."
    
    sample_input = sample_batch['input']
    
    batch_size = sample_input.size(0)
    input_shape = sample_input.shape
    
    # Check if target normalization is needed
    has_target = 'target' in sample_batch and target_axes is not None
    if has_target:
        sample_target = sample_batch['target']
        target_shape = sample_target.shape
    
    # ========== CALCULATE NORMALIZATION PARAMETERS IN SINGLE LOOP ==========
    
    # Input Welford variables
    input_n = 0
    input_mean = None
    input_M2 = None
    
    # Target Welford variables
    target_n = 0
    target_mean = None
    target_M2 = None
    
    calculate_input = norm_params is None
    calculate_target = has_target and target_norm_params is None
    
    if calculate_input or calculate_target:
        desc = "Calculating normalization parameters"
        if calculate_input and calculate_target:
            desc += " (input & target)"
        elif calculate_input:
            desc += " (input only)"
        else:
            desc += " (target only)"
        
        for batch in tqdm(train_loader, desc=desc):
            # ===== Process Input =====
            if calculate_input:
                batch_input = batch['input']
                
                batch_mean = batch_input.mean(dim=axes)
                batch_var = batch_input.var(dim=axes, unbiased=False)
                
                if input_mean is None:
                    input_mean = batch_mean
                    input_M2 = batch_var * batch_size
                    input_n = batch_size
                else:
                    # Combine batch stats with running stats (Welford's algorithm)
                    delta = batch_mean - input_mean
                    new_n = input_n + batch_size
                    
                    input_mean = (input_n * input_mean + batch_size * batch_mean) / new_n
                    input_M2 = input_M2 + batch_var * batch_size + delta ** 2 * input_n * batch_size / new_n
                    input_n = new_n
            
            # ===== Process Target =====
            if calculate_target and 'target' in batch:
                batch_target = batch['target']
                
                batch_mean = batch_target.mean(dim=target_axes)
                batch_var = batch_target.var(dim=target_axes, unbiased=False)
                
                if target_mean is None:
                    target_mean = batch_mean
                    target_M2 = batch_var * batch_size
                    target_n = batch_size
                else:
                    # Combine batch stats with running stats
                    delta = batch_mean - target_mean
                    new_n = target_n + batch_size
                    
                    target_mean = (target_n * target_mean + batch_size * batch_mean) / new_n
                    target_M2 = target_M2 + batch_var * batch_size + delta ** 2 * target_n * batch_size / new_n
                    target_n = new_n
    
    # ========== FINALIZE INPUT NORMALIZATION ==========
    if calculate_input:
        if input_n < 2:
            input_std = torch.zeros_like(input_mean)
        else:
            variance = input_M2 / input_n
            input_std = torch.sqrt(variance)
        
        # Reshape for broadcasting
        item_ndim = len(input_shape)
        reshape_dims = []
        mean_idx = 0
        
        for dim in range(item_ndim):
            if dim != 0:  # Batch dim was averaged but norm happens without batch dim
                if dim in axes:
                    reshape_dims.append(1)  # This dimension was averaged
                else:
                    reshape_dims.append(input_mean.shape[mean_idx])  # Keep actual size
                    mean_idx += 1
        
        input_mean = input_mean.view(*reshape_dims)
        input_std = input_std.view(*reshape_dims)
        
        logging.info(f"Calculated input mean shape: {input_mean.shape}, std shape: {input_std.shape}")
    else:
        # Use provided normalization parameters
        if not isinstance(norm_params[0], torch.Tensor):
            try:
                input_mean = torch.tensor(norm_params[0], dtype=sample_input.dtype)
                input_std = torch.tensor(norm_params[1], dtype=sample_input.dtype)
            except Exception as e:
                raise ValueError(f"Error converting norm_params to tensors: {e}")
        else:
            input_mean = norm_params[0]
            input_std = norm_params[1]
        logging.info(f"Using provided input mean {input_mean.shape} and std {input_std.shape} for normalization.")
    
    # ========== FINALIZE TARGET NORMALIZATION ==========
    if calculate_target:
        if target_n < 2:
            target_std = torch.zeros_like(target_mean)
        else:
            variance = target_M2 / target_n
            target_std = torch.sqrt(variance)
        
        # Reshape for broadcasting
        item_ndim = len(target_shape)
        reshape_dims = []
        mean_idx = 0
        
        for dim in range(item_ndim):
            if dim != 0:  # Batch dim was averaged but norm happens without batch dim
                if dim in target_axes:
                    reshape_dims.append(1)  # This dimension was averaged
                else:
                    reshape_dims.append(target_mean.shape[mean_idx])  # Keep actual size
                    mean_idx += 1
        
        target_mean = target_mean.view(*reshape_dims)
        target_std = target_std.view(*reshape_dims)
        
        logging.info(f"Calculated target mean shape: {target_mean.shape}, std shape: {target_std.shape}")
    elif has_target and target_norm_params is not None:
        # Use provided normalization parameters
        if not isinstance(target_norm_params[0], torch.Tensor):
            try:
                target_mean = torch.tensor(target_norm_params[0], dtype=sample_target.dtype)
                target_std = torch.tensor(target_norm_params[1], dtype=sample_target.dtype)
            except Exception as e:
                raise ValueError(f"Error converting target_norm_params to tensors: {e}")
        else:
            target_mean = target_norm_params[0]
            target_std = target_norm_params[1]
        logging.info(f"Using provided target mean {target_mean.shape} and std {target_std.shape} for normalization.")
    
    # Return tuple of tuples
    input_norm = (input_mean, input_std)
    target_norm = (target_mean, target_std) if has_target else None
    
    return input_norm, target_norm



def get_data_loaders(
    dataset: TorchDataset,
    batch_size: int = 32,
    sets_size: dict = {"train": 0.6, "val": 0.2, "test": 0.2},
    num_workers: int = 4,
    norm_axes: Tuple[int] = None,
    target_norm_axes: Tuple[int] = None,
    norm_params: Tuple[float, float] = None,
    target_norm_params: Tuple[float, float] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    '''
    Create train/val/test data loaders with optional normalization.
    
    Args:
        dataset: TorchDataset instance
        batch_size: Batch size for dataloaders
        sets_size: Dict with 'train', 'val', 'test' split ratios
        num_workers: Number of workers for data loading
        norm_axes: Axes to normalize 'input' across, e.g. (0, 2, 3) for [batch, channels, height, width]
        target_norm_axes: Axes to normalize 'target' across. If None, target not normalized
        norm_params: Pre-computed (mean, std) for input. If None, calculate from training set
        target_norm_params: Pre-computed (mean, std) for target. If None, calculate from training set
    
    Returns:
        (train_loader, val_loader, test_loader)
    '''
    assert isinstance(sets_size, dict), "sets_size must be a dict with keys 'train', 'val', 'test'"
    indices = np.arange(len(dataset))

    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)

    train_idx, val_idx, test_idx = _get_set_sizes(sets_size, dataset, indices)

    if norm_axes is not None or target_norm_axes is not None:
        if norm_axes is not None:
            logging.info(f"Calculating input normalization parameters over axes {norm_axes}")
        if target_norm_axes is not None:
            logging.info(f"Calculating target normalization parameters over axes {target_norm_axes}")

        temp_train_dataset = Subset(dataset, train_idx)

        logging.info(f"Using {len(train_idx)} samples for normalization calculation")
        temp_train_loader = DataLoader(
            temp_train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Calculate normalization parameters for both input and target in single pass
        input_norm, target_norm = _calc_norm_params(
            temp_train_loader, 
            axes=norm_axes, 
            target_axes=target_norm_axes,
            norm_params=norm_params,
            target_norm_params=target_norm_params
        )

        # Store in dataset
        if norm_axes is not None:
            dataset._norm_params = input_norm
        
        if target_norm is not None:
            dataset._target_norm_params = target_norm
            logging.info("Target normalization parameters calculated and stored")
        
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader, test_loader