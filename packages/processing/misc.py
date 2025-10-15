from typing import Dict, Iterator, List

import numpy as np
import torch

from packages.data_objects.signal import SignalObject


def absolute_values(Signal: SignalObject) -> np.ndarray:
    Signal.signal = np.abs(Signal.signal)
    return Signal


def normalize_values(
    Signal: SignalObject, across_dims: List[str] = None, method: str = "zscore", global_norm: Dict = None
) -> SignalObject:
    """
    Normalize signal values across specified dimensions.

    Args:
        Signal: SignalObject to normalize
        across_dims: List of dimension names to calculate mean/std across.
                    Example: ['channels', 'freqs'] means normalize across channels and frequencies,
                    keeping separate stats for each time step.
                    If None, normalizes across all dimensions.
        method: Normalization method ('zscore', 'minmax', etc.)

    Returns:
        SignalObject with normalized signal
    """
    if global_norm is not None:
        mean = global_norm.get("mean")
        std = global_norm.get("std")
        if mean is None or std is None:
            raise ValueError("Global normalization requires 'mean' and 'std' in global_norm dictionary.")
        Signal.signal = (Signal.signal - mean) / (std + 1e-10)
        return Signal
    
    if across_dims is None:
        # Normalize across all dimensions
        axis = None
    else:
        # Convert dimension names to axis indices
        axis = []
        for dim_name in across_dims:
            if dim_name in Signal.dim_dict:
                axis.append(Signal.dim_dict[dim_name])
            else:
                raise ValueError(
                    f"Dimension '{dim_name}' not found in signal dimensions: {list(Signal.dim_dict.keys())}"
                )

        axis = tuple(axis) if len(axis) > 1 else axis[0]

    if method == "zscore":
        mean = np.mean(Signal.signal, axis=axis, keepdims=True)
        std = np.std(Signal.signal, axis=axis, keepdims=True)
        Signal.signal = (Signal.signal - mean) / (std + 1e-10)
    elif method == "minmax":
        min_val = np.min(Signal.signal, axis=axis, keepdims=True)
        max_val = np.max(Signal.signal, axis=axis, keepdims=True)
        Signal.signal = (Signal.signal - min_val) / (max_val - min_val + 1e-10)

    return Signal

def calculate_global_normalization_params(data_loader: Iterator, ) -> Dict[str, np.ndarray]:
    """
    Calculate global mean and std across a dataset.

    Args:
        data_loader: Iterator yielding data
    Returns:
        Dictionary with global 'mean' and 'std' arrays
    """

    means = []
    stds = []
    for data in data_loader:
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        means.append(np.mean(data))
        stds.append(np.std(data))
    return {"mean": np.mean(means), "std": np.mean(stds)}