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

def _calc_norm_params(train_loader, axes, norm_params=None):
    """
    Calculate global mean and std using batched Welford's algorithm.
    Faster and more memory efficient.
    Axes specifies the the dimensions to calculate the mean and std across, note that tensor shape is [batch, ...]
    """
    sample_batch = next(iter(train_loader))
    if isinstance(sample_batch, (list, tuple)):
        sample_batch = sample_batch[0]
    if isinstance(sample_batch, np.ndarray):
        sample_batch = torch.from_numpy(sample_batch)

    batch_size = sample_batch.size(0)
    item_shape = sample_batch.shape[0:]

    n = 0
    mean = None
    M2 = None

    if norm_params is None:
        for batch in tqdm(train_loader, desc="Calculating global parameters"):

            batch_mean = batch.mean(dim=axes)
            batch_var = batch.var(dim=axes, unbiased=False)
            
            if mean is None:
                mean = batch_mean
                M2 = batch_var * batch_size
                n = batch_size
            else:
                # Combine batch stats with running stats
                delta = batch_mean - mean
                new_n = n + batch_size
                
                mean = (n * mean + batch_size * batch_mean) / new_n
                M2 = M2 + batch_var * batch_size + delta ** 2 * n * batch_size / new_n
                n = new_n
        if n < 2:
            return mean, torch.zeros_like(mean)
        
        variance = M2 / n
        std = torch.sqrt(variance)
        item_ndim = len(item_shape)
        reshape_dims = []

        mean_idx = 0
        for dim in range(item_ndim):
            if dim != 0: # Batch dim was averaged but the norm happens without batch dim
                if dim in axes:
                    # This dimension was averaged, use 1 for broadcasting
                    reshape_dims.append(1)
                else:
                    # This dimension was kept, use the actual size
                    reshape_dims.append(mean.shape[mean_idx])
                    mean_idx += 1

        mean = mean.view(*reshape_dims)
        std = std.view(*reshape_dims)
        print(f"Calculated mean shape: {mean.shape}, std shape: {std.shape}")
    else:
        mean, std = norm_params
        print(f"Using provided mean {mean} and std {std} for normalization.")

    return mean, std


def get_data_loaders(
    dataset: TorchDataset,
    batch_size: int = 32,
    sets_size: dict = {"train": 0.6, "val": 0.2, "test": 0.2},
    num_workers: int = 4,
    norm_axes: Tuple[int] = None,
    norm_params: Tuple[float, float] = None,
) -> DataLoader:
    '''
    norm_params : (mean, std) to use for normalization, if None, calculate from training set. they'll be casted to tensors
    and stored in dataset._norm_params based on norm_axes
    norm_axes : axes to calculate normalization across, e.g. (0, 2, 3) to normalize across batch, height and width for 4D tensors [batch, channels, height, width]
    '''
    assert isinstance(sets_size, dict), "sets_size must be a dict with keys 'train', 'val', 'test'"
    indices = np.arange(len(dataset))

    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)

    train_idx, val_idx, test_idx = _get_set_sizes(sets_size, dataset, indices)

    if norm_axes is not None:
        logging.info(f"Calculating normalization parameters over axes {norm_axes}")

        temp_train_dataset = Subset(dataset, train_idx)

        logging.info(f"Using {len(train_idx)} samples for normalization calculation")
        temp_train_loader = DataLoader(
            temp_train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        mean, std = _calc_norm_params(temp_train_loader, axes=norm_axes, norm_params=norm_params)

        dataset._norm_params = (mean, std)
        
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


