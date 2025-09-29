import numpy as np
from packages.data_objects.signal import SignalObject
from typing import List


def window_delta_displacement(sensor_data: np.ndarray, window: int, offset: int) -> np.ndarray:
    
    if sensor_data.ndim not in [2, 3]:
        raise ValueError("sensor_data must be 2D or 3D")

    if sensor_data.ndim == 2:
        if sensor_data.shape[0] > sensor_data.shape[1]:
            sensor_data = sensor_data.T  # transpose if samples axis is not last

    n_samples = sensor_data.shape[-1]

    if offset >= 0:
        start = offset
    else:
        start = n_samples + offset - window + 1
    end = start + window

    if start < 0 or end > n_samples:
        raise ValueError("Window with given offset is out of bounds.")

    windowed = sensor_data[..., start:end]

    first = windowed[..., 0]
    last = windowed[..., -1]
    displacement = last - first

    return displacement

def absolute_values(Signal: SignalObject) -> np.ndarray:
     Signal.signal = np.abs(Signal.signal)
     return Signal

def normalize_values(Signal: SignalObject, across_dims: List[str] = None, method: str = 'zscore') -> SignalObject:
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
                raise ValueError(f"Dimension '{dim_name}' not found in signal dimensions: {list(Signal.dim_dict.keys())}")
        
        axis = tuple(axis) if len(axis) > 1 else axis[0]
    
    if method == 'zscore':
        mean = np.mean(Signal.signal, axis=axis, keepdims=True)
        std = np.std(Signal.signal, axis=axis, keepdims=True)
        Signal.signal = (Signal.signal - mean) / (std + 1e-10)
    elif method == 'minmax':
        min_val = np.min(Signal.signal, axis=axis, keepdims=True)
        max_val = np.max(Signal.signal, axis=axis, keepdims=True)
        Signal.signal = (Signal.signal - min_val) / (max_val - min_val + 1e-10)
    
    return Signal
