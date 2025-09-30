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

