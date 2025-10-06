from typing import List

import numpy as np

from packages.data_objects.signal import GLOBAL_DIM_KEYS, SignalObject


def window_delta_value(
    sensor_data: SignalObject,
    window: int,
    offset: int,
    dim: str = GLOBAL_DIM_KEYS.TIME.value,
) -> SignalObject:
    n_samples = getattr(sensor_data, dim)
    dim_axis = sensor_data.dim_dict[dim]
    if offset >= 0:
        start = offset
    else:
        start = n_samples + offset - window + 1
    end = start + window

    if start < 0 or end > n_samples:
        raise ValueError("Window with given offset is out of bounds.")

    slices = [slice(None)] * sensor_data.signal.ndim
    slices[dim_axis] = slice(start, end)
    windowed = sensor_data.signal[tuple(slices)]

    first = windowed.take(indices=0, axis=dim_axis)
    last = windowed.take(indices=-1, axis=dim_axis)
    displacement = last - first
    sensor_data.signal = displacement
    sensor_data._delete_from_dim_dict([dim], pipe=None)
    return sensor_data
