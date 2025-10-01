import numpy as np
from typing import Tuple, List, Union

from packages.data_objects.signal import GLOBAL_DIM_KEYS, EegSignal, RandomSignal, NULL_VALUES, SignalObject, MultimodalTimeSignal

import logging
logger = logging.getLogger('ReshapeLogger')
logger.setLevel(logging.DEBUG)


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

def _find_electrode_idx(current_schema, electrode):
    """Find the index/indices of an electrode in the schema."""
    idx = np.where(current_schema == electrode)
    idx = list(zip(*idx))
    if not idx:
        raise ValueError(f"Electrode {electrode} not found in current schema.")
    return idx[0] if len(idx) == 1 else idx[0]

def _assign_to_reshaped(reshaped_signal, Signal, idx_out, idx_in, is_spatial):
    """Assign the correct slice from Signal.signal to reshaped_signal."""
    if is_spatial:
        reshaped_signal[idx_out] = Signal.signal[idx_in]
    else:
        reshaped_signal[idx_out] = Signal.signal[idx_in[0]]

def _ensure_dim_order(Signal, order):
    if not Signal._check_dim_order(order):
        Signal._reorder_signal_dimensions(order)

def reshape_to_spatial(Signal: EegSignal, spatial_domain_matrix: np.ndarray) -> np.ndarray:
    """
    Reshape EEG signal to spatial configuration based on electrode layout.
    """
    current_schema = Signal.electrode_schema

    # Check if all electrodes are present
    if not all(elem in spatial_domain_matrix for elem in current_schema if elem is not None):
        raise ValueError("Target Spatial domain matrix does not contain all electrodes from the current schema.")

    is_spatial = Signal.is_spatial_signal

    if spatial_domain_matrix.ndim == 2:
        order = ['rows', 'cols'] if is_spatial else ['channels']
        _ensure_dim_order(Signal, order)

        num_rows, num_cols = spatial_domain_matrix.shape
        original_shape = Signal.signal.shape
        new_shape = (num_rows, num_cols) + original_shape[2:] if is_spatial else (num_rows, num_cols) + original_shape[1:]
        reshaped_signal = np.zeros(new_shape)

        for r in range(num_rows):
            for c in range(num_cols):
                electrode = spatial_domain_matrix[r, c]
                idx_out = (r, c) + (slice(None),) * (reshaped_signal.ndim - 2)
                if electrode in NULL_VALUES:
                    reshaped_signal[idx_out] = 0
                else:
                    idx_in = _find_electrode_idx(current_schema, electrode)
                    _assign_to_reshaped(reshaped_signal, Signal, idx_out, idx_in, is_spatial)

    elif spatial_domain_matrix.ndim == 1:
        order = ['rows', 'cols'] if is_spatial else ['channels']
        _ensure_dim_order(Signal, order)

        num_electrodes = len(spatial_domain_matrix)
        original_shape = Signal.signal.shape
        new_shape = (num_electrodes,) + original_shape[2:] if is_spatial else (num_electrodes,) + original_shape[1:]
        reshaped_signal = np.zeros(new_shape)

        for c in range(num_electrodes):
            electrode = spatial_domain_matrix[c]
            idx_out = (c,) + (slice(None),) * (reshaped_signal.ndim - 1)
            if electrode in NULL_VALUES:
                reshaped_signal[idx_out] = 0
            else:
                idx_in = _find_electrode_idx(current_schema, electrode)
                _assign_to_reshaped(reshaped_signal, Signal, idx_out, idx_in, is_spatial)

    else:
        raise ValueError("1D or 2D spatial domain matrix currently supported.")

    Signal.signal = reshaped_signal
    return Signal
    
def segment_signal(Signal: Union[SignalObject, MultimodalTimeSignal], window: int = 0, overlap: float = 0) -> SignalObject:
    """
    Batch EEG data into smaller segments for processing.
    Allows for overlapping and windowed using input parameters.
    Requires SignalObject with TIME dimension.
    Parameters:
    - Signal: SignalObject to segment
    - window: Integer specifying the window size for each segment
    - overlap: Integer specifying the overlap between segments
    Returns:
    - SignalObject with signal segmented along specified dimension
    """
    step = window - overlap
    total_length = Signal.time
    _validate_segmentation_params(window, overlap, step)
    _validate_time_series(Signal)
    
    
    if isinstance(Signal, MultimodalTimeSignal):
        
        num_signals = Signal.num_signals
        for idx, signal in enumerate(Signal.signals):
            signal = _segment_time_series(signal, window, step, total_length)
            Signal.signals[idx] = signal
        return Signal
    elif isinstance(Signal, SignalObject):
        Signal = _segment_time_series(Signal, window, step, total_length)
        return Signal
    else:
        raise ValueError("Input must be a SignalObject or MultimodalTimeSignal instance.")
    

def _segment_time_series(Signal: SignalObject, window: int, step: int, total_length: int) -> SignalObject:
    axis = Signal.dim_dict.get(GLOBAL_DIM_KEYS.TIME.value, 0)
    segments = []
    for start in range(0, total_length - window + 1, step):
        end = start + window
        slices = [slice(None)] * Signal.signal.ndim
        slices[axis] = slice(start, end)
        segments.append(Signal.signal[tuple(slices)])

    Signal.signal = np.stack(segments, axis=0)

    return Signal

def _validate_segmentation_params(window: int, overlap: float, step: int, total_length: int):
    if window <= 0 or window > total_length:
        raise ValueError("Window size must be positive and smaller than signal length.")
    if overlap < 0:
        raise ValueError("Overlap must be non-negative.")
    if step <= 0:
        raise ValueError("Detected negative step value, ensure that window > overlap.")
    
def _validate_time_series(self):
    # Check that all signals have TIME dimension
    time_attr = GLOBAL_DIM_KEYS.TIME.value
    for i, signal in enumerate(self.signals):
        if time_attr not in signal.dim_dict:
            raise ValueError(f"Signal at index {i} does not have a TIME dimension.")
        

# MARK Deprecated Functions

def raw_segment_data(eeg_data: np.ndarray, sensor_data: np.ndarray = None, window: int = 0, overlap: float = 0, axis: int = -1, segment_sensor_signal: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    WARNING: This function is deprecated and may be removed in future versions.
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
    logger.warning("The function 'raw_segment_data' is deprecated and may be removed in future versions. Consider using 'segment_data' instead.")
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


def raw_reshape_to_spatial(Signal: EegSignal, spatial_domain_matrix: np.ndarray) -> np.ndarray:
    """
    WARNING: This function is deprecated and may be removed in future versions.
    Reshape EEG signal to spatial configuration based on electrode layout.
    Parameters:
    - eeg_data: numpy array of shape (channels, samples) or (channels, frequencies, samples)
    - spatial_domain_matrix: 2D numpy array defining the spatial layout of electrodes

    Returns:
    - SignalObject with signal = numpy array reshaped to spatial configuration
    """
    logger.warning("The function 'raw_reshape_to_spatial' is deprecated and may be removed in future versions. Consider using 'reshape_to_spatial' instead.")
    current_schema = Signal.electrode_schema
    
    # Check if spatial_domain_matrix has all the electrodes of the current schema
    if not all(elem in spatial_domain_matrix for elem in current_schema if elem is not None):
        raise ValueError("Target Spatial domain matrix does not contain all electrodes from the current schema.")
    # Rows and Cols case
    if spatial_domain_matrix.ndim == 2:
        if Signal.is_spatial_signal:

            if not Signal._check_dim_order(['rows', 'cols']):
                Signal._reorder_signal_dimensions(['rows', 'cols'])

            num_rows, num_cols = spatial_domain_matrix.shape
            original_shape = Signal.signal.shape
            new_shape = (num_rows, num_cols) + original_shape[2:]
            reshaped_signal = np.zeros(new_shape)
            
            for r in range(num_rows):
                for c in range(num_cols):
                    electrode = spatial_domain_matrix[r, c]
                    if electrode in NULL_VALUES:
                        reshaped_signal[r, c, ...] = 0  # or np.nan
                    else:
                        electrode_idx = np.where(current_schema == electrode)
                        electrode_idx = list(zip(*electrode_idx))[0]
                        row = electrode_idx[0]
                        col = electrode_idx[1]
                        reshaped_signal[r, c, ...] = Signal.signal[row, col, ...]

            Signal.signal = reshaped_signal

        else:
            if not Signal._check_dim_order(['channels']):
                Signal._reorder_signal_dimensions(['channels'])

            num_rows, num_cols = spatial_domain_matrix.shape
            original_shape = Signal.signal.shape
            new_shape = (num_rows, num_cols) + original_shape[1:]
            reshaped_signal = np.zeros(new_shape)
            
            for r in range(num_rows):
                for c in range(num_cols):
                    electrode = spatial_domain_matrix[r, c]
                    if electrode in NULL_VALUES:
                        reshaped_signal[r, c, ...] = 0  # or np.nan
                    else:
                        electrode_idx = np.where(current_schema == electrode)
                        electrode_idx = list(zip(*electrode_idx))[0]
                        channel = electrode_idx[0]
                        reshaped_signal[r, c, ...] = Signal.signal[channel, ...]
                        
            # Update SignalObject attributes
            Signal.signal = reshaped_signal

    if spatial_domain_matrix.ndim == 1:
        if Signal.is_spatial_signal:
            if not Signal._check_dim_order(['rows', 'cols']):
                Signal._reorder_signal_dimensions(['rows', 'cols'])

            num_electrodes = len(spatial_domain_matrix)
            original_shape = Signal.signal.shape
            new_shape = (num_electrodes,) + original_shape[2:]
            reshaped_signal = np.zeros(new_shape)

            for c in range(num_electrodes):
                electrode = spatial_domain_matrix[c]
                if electrode in NULL_VALUES:
                    reshaped_signal[c, ...] = 0  # or np.nan
                else:
                    electrode_idx = np.where(current_schema == electrode)
                    electrode_idx = list(zip(*electrode_idx))[0]
                    row = electrode_idx[0]
                    col = electrode_idx[1]
                    reshaped_signal[c, ...] = Signal.signal[row, col, ...]
            
            # Update SignalObject attributes
            Signal.signal = reshaped_signal
        else:
            if not Signal._check_dim_order(['channels']):
                Signal._reorder_signal_dimensions(['channels'])

            num_electrodes = len(spatial_domain_matrix)
            original_shape = Signal.signal.shape
            new_shape = (num_electrodes,) + original_shape[1:]
            reshaped_signal = np.zeros(new_shape)

            for c in range(num_electrodes):
                electrode = spatial_domain_matrix[c]
                if electrode in NULL_VALUES:
                    reshaped_signal[c, ...] = 0  # or np.nan
                else:
                    electrode_idx = np.where(current_schema == electrode)
                    electrode_idx = list(zip(*electrode_idx))[0]
                    channel = electrode_idx[0]
                    reshaped_signal[c, ...] = Signal.signal[channel, ...]
            
            # Update SignalObject attributes
            Signal.signal = reshaped_signal
    return Signal
