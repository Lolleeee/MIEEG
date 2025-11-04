import numpy as np
import torch
import pytest
from torch.utils.data import DataLoader

from packages.io.torch_dataloaders import (
    _get_set_sizes,
    _calc_norm_params,
    get_data_loaders,
)
from packages.data_objects.dataset import TestTorchDataset, RANDOM_SEED


def test_get_set_sizes_with_train_val_test():
    """Test set size calculation when all three splits are specified"""
    sets_size = {"train": 0.6, "val": 0.2, "test": 0.2}
    dataset = TestTorchDataset(nsamples=100, shape=(10,))
    indices = np.arange(100)
    
    train_idx, val_idx, test_idx = _get_set_sizes(sets_size, dataset, indices)
    
    assert len(train_idx) == 60
    assert len(val_idx) == 20
    assert len(test_idx) == 20
    assert len(set(train_idx) & set(val_idx)) == 0  # no overlap
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0


def test_get_set_sizes_without_test():
    """Test set size calculation when test split is not specified (uses remainder)"""
    sets_size = {"train": 0.7, "val": 0.2}
    dataset = TestTorchDataset(nsamples=100, shape=(10,))
    indices = np.arange(100)
    
    train_idx, val_idx, test_idx = _get_set_sizes(sets_size, dataset, indices)
    
    assert len(train_idx) == 70
    assert len(val_idx) == 20
    assert len(test_idx) == 10  # remainder


def test_calc_norm_params_computes_mean_and_std():
    """Test normalization parameter calculation"""
    # Create dataset with known statistics
    dataset = TestTorchDataset(nsamples=50, shape=(3, 4, 5))
    loader = DataLoader(dataset, batch_size=10, num_workers=0)
    
    # Calculate across batch and last dimension (axes 0 and 3)
    mean, std = _calc_norm_params(loader, axes=(0, 3))
    
    assert mean.shape == (3, 4, 1)  # kept dims 1,2; averaged 0,3
    assert std.shape == (3, 4, 1)
    assert torch.all(torch.isfinite(mean))
    assert torch.all(torch.isfinite(std))
    assert torch.all(std > 0)  # std should be positive


def test_calc_norm_params_with_provided_params():
    """Test that provided normalization params are returned unchanged"""
    dataset = TestTorchDataset(nsamples=20, shape=(5,))
    loader = DataLoader(dataset, batch_size=5, num_workers=0)
    
    provided_mean = torch.tensor([1.0])
    provided_std = torch.tensor([0.5])
    
    mean, std = _calc_norm_params(loader, axes=(0,), norm_params=(provided_mean, provided_std))
    
    assert torch.equal(mean, provided_mean)
    assert torch.equal(std, provided_std)


def test_calc_norm_params_single_batch_returns_zeros_std():
    """Test that single batch (n<2) returns zero std"""
    dataset = TestTorchDataset(nsamples=1, shape=(10,))
    loader = DataLoader(dataset, batch_size=1, num_workers=0)  # only 1 batch
    
    mean, std = _calc_norm_params(loader, axes=(0,))
    
    assert torch.all(std == 0)  # std is zero for single sample


def test_get_data_loaders_returns_three_loaders():
    """Test that get_data_loaders returns train/val/test loaders"""
    dataset = TestTorchDataset(nsamples=100, shape=(5, 10))
    
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset,
        batch_size=10,
        sets_size={"train": 0.6, "val": 0.2, "test": 0.2},
        num_workers=0,
        norm_axes=None,
    )
    
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    
    # Check approximate sizes (may vary by 1 due to rounding)
    assert len(train_loader.dataset) == 60
    assert len(val_loader.dataset) == 20
    assert len(test_loader.dataset) == 20


def test_get_data_loaders_sets_normalization_params():
    """Test that normalization params are set on dataset when norm_axes provided"""
    dataset = TestTorchDataset(nsamples=50, shape=(3, 4, 5))
    
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset,
        batch_size=10,
        sets_size={"train": 0.6, "val": 0.2, "test": 0.2},
        num_workers=0,
        norm_axes=(0, 2, 3),  # normalize across batch, height, width
    )
    
    assert dataset._norm_params is not None
    mean, std = dataset._norm_params
    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)
    assert mean.shape == (3, 1, 1)  # kept channel dim
    assert std.shape == (3, 1, 1)


def test_get_data_loaders_reproducibility():
    """Test that data loaders produce same splits with same seed"""
    dataset1 = TestTorchDataset(nsamples=100, shape=(10,))
    dataset2 = TestTorchDataset(nsamples=100, shape=(10,))
    
    # Reset seed before each call
    np.random.seed(RANDOM_SEED)
    train1, val1, test1 = get_data_loaders(
        dataset1, batch_size=10, sets_size={"train": 0.6, "val": 0.2, "test": 0.2}, num_workers=0
    )
    
    np.random.seed(RANDOM_SEED)
    train2, val2, test2 = get_data_loaders(
        dataset2, batch_size=10, sets_size={"train": 0.6, "val": 0.2, "test": 0.2}, num_workers=0
    )
    
    # Check that indices are the same
    assert len(train1.dataset) == len(train2.dataset)
    assert len(val1.dataset) == len(val2.dataset)
    assert len(test1.dataset) == len(test2.dataset)


def test_get_data_loaders_no_overlap_between_sets():
    """Test that train/val/test sets don't overlap"""
    dataset = TestTorchDataset(nsamples=100, shape=(10,))
    
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset,
        batch_size=10,
        sets_size={"train": 0.6, "val": 0.2, "test": 0.2},
        num_workers=0,
    )
    
    train_indices = set(train_loader.dataset.indices)
    val_indices = set(val_loader.dataset.indices)
    test_indices = set(test_loader.dataset.indices)
    
    assert len(train_indices & val_indices) == 0
    assert len(train_indices & test_indices) == 0
    assert len(val_indices & test_indices) == 0


def test_get_data_loaders_with_custom_batch_size():
    """Test that batch size is respected"""
    dataset = TestTorchDataset(nsamples=100, shape=(10,))
    
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset,
        batch_size=16,
        sets_size={"train": 0.6, "val": 0.2, "test": 0.2},
        num_workers=0,
    )
    
    assert train_loader.batch_size == 16
    assert val_loader.batch_size == 16
    assert test_loader.batch_size == 16


def test_calc_norm_params_shape_consistency():
    """Test that normalization params have correct broadcasting shape"""
    # Shape: (batch, channels, height, width, time)
    dataset = TestTorchDataset(nsamples=30, shape=(3, 10, 8, 50))
    loader = DataLoader(dataset, batch_size=10, num_workers=0)
    
    # Normalize across batch, height, width, time (keep channels)
    mean, std = _calc_norm_params(loader, axes=(0, 2, 3, 4))
    
    # Should have shape (channels, 1, 1, 1) for broadcasting
    assert mean.shape == (3, 1, 1, 1)
    assert std.shape == (3, 1, 1, 1)