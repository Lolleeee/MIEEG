import numpy as np
from typing import Tuple, List, Union

SPATIAL_DOMAIN_MATRIX = np.array([
            [None, 'Fp1', None, 'Fp2', None],
            ['F7', 'F3', 'Fz', 'F4', 'F8'],
            ['FC5', 'FC1', 'Cz', 'FC2', 'FC6'],
            ['T7', 'C3', 'CP1', 'C4', 'T8'],
            ['TP9', 'CP5', 'CP2', 'CP6', 'TP10'],
            ['P7', 'P3', 'Pz', 'P4', 'P8'],
            ['PO9', 'O1', 'Oz', 'O2', 'PO10']
        ])

CHANNELS = np.array([
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
    'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
    'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10'
])


def reshape_to_spatial(eeg_data: np.ndarray, nan: str = "0") -> np.ndarray:
    """
    Reshape EEG data from (channels, frequencies, samples) to (rows, cols, frequencies, samples) based on the spatial layout of electrodes.
    or from (channels, samples) to (rows, cols, samples) if input is 2D.
    Parameters:
    - eeg_data: 2D or 3D numpy array of shape (channels, samples) or (channels, frequencies, samples)
    
    Returns:
    - reshaped_data: 3D numpy array of shape (rows, cols, frequencies, samples)
    """
    if eeg_data.ndim not in [2, 3] or eeg_data.shape[0] != len(CHANNELS):
        raise ValueError(f"Input data must be a 2D or 3D array with {len(CHANNELS)} channels.")
    
    rows, cols = SPATIAL_DOMAIN_MATRIX.shape
    
    if eeg_data.ndim == 2:
        samples = eeg_data.shape[1]
        reshaped_data = np.zeros((rows, cols, samples))
        for i in range(rows):
            for j in range(cols):
                channel = SPATIAL_DOMAIN_MATRIX[i, j]
                if channel is not None:
                    channel_index = np.where(CHANNELS == channel)[0][0]
                    reshaped_data[i, j, :] = eeg_data[channel_index, :]
                else:
                    reshaped_data[i, j, :] = 0 if nan == "0" else np.nan
    else:  # eeg_data.ndim == 3
        frequencies = eeg_data.shape[1]
        samples = eeg_data.shape[2]
        reshaped_data = np.zeros((rows, cols, frequencies, samples))
        for i in range(rows):
            for j in range(cols):
                channel = SPATIAL_DOMAIN_MATRIX[i, j]
                if channel is not None:
                    channel_index = np.where(CHANNELS == channel)[0][0]
                    reshaped_data[i, j, :, :] = eeg_data[channel_index, :, :]
                else:
                    reshaped_data[i, j, :, :] = 0 if nan == "0" else np.nan
    
    return reshaped_data


def segment_data(eeg_data: np.ndarray, sensor_data: np.ndarray = None, window: int = 0, overlap: float = 0, axis: int = -1, segment_sensor_signal: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Batch EEG data into smaller segments for processing.
    Allows for overlapping and windowed using input parameters.
    Parameters:
    - eeg_data: numpy array
    - sensor_data: numpy array
    - window: Integer specifying the window size for each segment
    - overlap: Integer specifying the overlap between segments
    - axis: Axis along which to batch the data
    - segment_sensor_signal: True if there's an additional sensor signal which has to be in synch with the eeg data
    Returns:
    - batched_data: array of segmented EEG data with shape (num_segments, ...)
    - tuple if sensor data is segmented 
    """

    step = window - overlap
    if step <= 0:
        raise ValueError("Detected negative step value, ensure that window > overlap.")
    
    total_length = eeg_data.shape[axis]
    eeg_segments = []
    sensor_segments = []
    for start in range(0, total_length - window + 1, step):
        end = start + window
        slices = [slice(None)] * eeg_data.ndim
        slices[axis] = slice(start, end)
        eeg_segments.append(eeg_data[tuple(slices)])

        if segment_sensor_signal and sensor_data.size > 0:
            slices = [slice(None)] * sensor_data.ndim
            slices[axis] = slice(start, end)
            sensor_segments.append(sensor_data[tuple(slices)])
    if segment_sensor_signal and sensor_data.size > 0:    
        return np.stack(eeg_segments, axis=0), np.stack(sensor_segments, axis=0)
    else:
        return np.stack(eeg_segments, axis=0)
    

def dim_labeler(signal_tensor: np.ndarray):
    pass