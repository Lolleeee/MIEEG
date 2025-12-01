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

from packages.data_objects.dataset import RANDOM_SEED, TorchDataset, TorchH5Dataset, AugmentedDataset

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
    target_norm_params: Tuple[float, float] | Tuple[torch.Tensor, torch.Tensor] | None = None,
    max_samples: int = None,
    max_batches: int = None,
    convergence_threshold: float = 1e-3,
    min_batches: int = 10
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor] | None]:
    """
    Calculate global mean and std using batched Welford's algorithm with early stopping.
    Faster and more memory efficient. Calculates both input and target in single pass.
    
    Args:
        train_loader: DataLoader for training data
        axes: Dimensions to calculate mean/std across for 'input' (note: tensor shape is [batch, ...])
        target_axes: Dimensions to calculate mean/std across for 'target'. If None, target not normalized
        norm_params: Pre-computed (mean, std) for input. If None, calculate from data
        target_norm_params: Pre-computed (mean, std) for target. If None, calculate from data
        max_samples: Maximum number of samples to use for calculation. If None, use all data
        max_batches: Maximum number of batches to process. If None, use all batches
        convergence_threshold: Stop if relative change in mean/std is below this threshold
        min_batches: Minimum number of batches to process before checking convergence
    
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
    
    # Calculate stopping conditions
    total_batches = len(train_loader)
    if max_samples is not None:
        max_batches_from_samples = (max_samples + batch_size - 1) // batch_size
        max_batches = min(max_batches_from_samples, max_batches) if max_batches else max_batches_from_samples
    
    if max_batches is not None:
        max_batches = min(max_batches, total_batches)
    else:
        max_batches = total_batches
    
    logging.info(f"Will process at most {max_batches}/{total_batches} batches for normalization")
    
    # ========== CALCULATE NORMALIZATION PARAMETERS IN SINGLE LOOP ==========
    
    # Input Welford variables
    input_n = 0
    input_mean = None
    input_M2 = None
    input_prev_mean = None
    input_prev_std = None
    
    # Target Welford variables
    target_n = 0
    target_mean = None
    target_M2 = None
    target_prev_mean = None
    target_prev_std = None
    
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
        
        batch_count = 0
        converged_input = False
        converged_target = False
        
        pbar = tqdm(train_loader, desc=desc, total=max_batches)
        
        for batch in pbar:
            batch_count += 1
            
            # ===== Process Input =====
            if calculate_input and not converged_input:
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
                
                # Check convergence for input
                if batch_count >= min_batches and batch_count % 5 == 0:  # Check every 5 batches
                    current_std = torch.sqrt(input_M2 / input_n)
                    
                    if input_prev_mean is not None and input_prev_std is not None:
                        mean_change = torch.abs(input_mean - input_prev_mean).max().item()
                        std_change = torch.abs(current_std - input_prev_std).max().item()
                        
                        mean_rel_change = mean_change / (torch.abs(input_mean).max().item() + 1e-8)
                        std_rel_change = std_change / (current_std.max().item() + 1e-8)
                        
                        if mean_rel_change < convergence_threshold and std_rel_change < convergence_threshold:
                            converged_input = True
                            logging.info(f"Input normalization converged at batch {batch_count}")
                    
                    input_prev_mean = input_mean.clone()
                    input_prev_std = current_std.clone()
            
            # ===== Process Target =====
            if calculate_target and not converged_target and 'target' in batch:
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
                
                # Check convergence for target
                if batch_count >= min_batches and batch_count % 5 == 0:
                    current_std = torch.sqrt(target_M2 / target_n)
                    
                    if target_prev_mean is not None and target_prev_std is not None:
                        mean_change = torch.abs(target_mean - target_prev_mean).max().item()
                        std_change = torch.abs(current_std - target_prev_std).max().item()
                        
                        mean_rel_change = mean_change / (torch.abs(target_mean).max().item() + 1e-8)
                        std_rel_change = std_change / (current_std.max().item() + 1e-8)
                        
                        if mean_rel_change < convergence_threshold and std_rel_change < convergence_threshold:
                            converged_target = True
                            logging.info(f"Target normalization converged at batch {batch_count}")
                    
                    target_prev_mean = target_mean.clone()
                    target_prev_std = current_std.clone()
            
            # Update progress bar
            pbar.set_postfix({
                'samples': input_n if calculate_input else target_n,
                'input_conv': converged_input if calculate_input else None,
                'target_conv': converged_target if calculate_target else None
            })
            
            # Early stopping conditions
            if batch_count >= max_batches:
                logging.info(f"Reached max_batches limit ({max_batches})")
                break
            
            if (not calculate_input or converged_input) and (not calculate_target or converged_target):
                logging.info(f"Both input and target converged at batch {batch_count}")
                break
        
        pbar.close()
    
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
        logging.info(f"Used {input_n} samples for input normalization")
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
        logging.info(f"Used {target_n} samples for target normalization")
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
    dataset: Union[TorchDataset, TorchH5Dataset],
    batch_size: int = 32,
    sets_size: dict = {"train": 0.6, "val": 0.2, "test": 0.2},
    num_workers: int = 4,
    norm_axes: Tuple[int] = None,
    target_norm_axes: Tuple[int] = None,
    norm_params: Tuple = None,
    target_norm_params: Tuple = None,
    augmentation_func: Callable = None,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    max_norm_samples: int = 5000, 
    min_norm_batches: int = 10,
    norm_convergence_threshold: float = 1e-3
    
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates optimized DataLoaders for Training, Validation, and Testing.
    Handles HDF5 lazy loading, normalization, and split argumentation safely.
    """
    
    # 1. SPLIT INDICES
    indices = np.arange(len(dataset))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
    
    train_idx, val_idx, test_idx = _get_set_sizes(sets_size, len(dataset), indices)
    
    # 2. CALCULATE NORMALIZATION (If needed)
    # We do this BEFORE creating subsets to inject params into the base dataset
    should_calc_input = (norm_axes is not None) and (norm_params is None)
    should_calc_target = (target_norm_axes is not None) and (target_norm_params is None)
    
    if should_calc_input or should_calc_target:
        logging.info("Calculating normalization parameters...")
        
        # Create a temporary subset for calculation (using only training data)
        norm_subset = Subset(dataset, train_idx[:max_norm_samples] if max_norm_samples else train_idx)
        
        # Use a transient DataLoader (kill workers immediately after use)
        norm_loader = DataLoader(
            norm_subset, 
            batch_size=batch_size * 2, 
            num_workers=min(num_workers, 2), # Don't need many workers for this
            persistent_workers=False
        )
        
        input_stats, target_stats = _calc_norm_params(
            norm_loader, 
            axes=norm_axes, 
            target_axes=target_norm_axes,
            norm_params=norm_params,
            target_norm_params=target_norm_params,
            min_batches=min_norm_batches,
            convergence_threshold=norm_convergence_threshold
        )
        
        # Inject into the underlying dataset
        # We handle nested wrappers (in case dataset is already wrapped)
        base_ds = dataset
        while hasattr(base_ds, 'dataset'):
            base_ds = base_ds.dataset
            
        if should_calc_input:
            base_ds._norm_params = input_stats
            logging.info(f"Input Normalization Set: {input_stats[0].shape}")
            
        if should_calc_target and target_stats:
            base_ds._target_norm_params = target_stats
            logging.info("Target Normalization Set")

    # 3. CREATE SUBSETS
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    # 4. APPLY AUGMENTATION (Train Only)
    # We wrap the training subset so augmentation doesn't leak to Val/Test
    if augmentation_func is not None:
        train_dataset = AugmentedDataset(train_dataset, augmentation_func)
        logging.info("Augmentation applied to Training set only.")

    # 5. CONFIGURE LOADERS
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        # Only use persistent workers if we have workers (avoid error)
        "persistent_workers": persistent_workers if num_workers > 0 else False,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    
    return train_loader, val_loader, test_loader

def _get_set_sizes(sets_size, total_len, indices):
    train_len = int(sets_size["train"] * total_len)
    val_len = int(sets_size["val"] * total_len)
    
    train_idx = indices[:train_len]
    val_idx = indices[train_len : train_len + val_len]
    
    if "test" in sets_size:
        test_len = int(sets_size["test"] * total_len)
        test_idx = indices[train_len + val_len : train_len + val_len + test_len]
    else:
        test_idx = indices[train_len + val_len:]
        
    return train_idx, val_idx, test_idx
