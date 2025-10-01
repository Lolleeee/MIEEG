import logging
import numpy as np
import os
import re
from typing import Any, Callable, Dict, List, Tuple, Union
from enum import Enum

logging.basicConfig(level=logging.WARNING)
file_handler = logging.StreamHandler()
file_handler.setLevel(logging.WARNING)
logger = logging.getLogger('signal_object')
logger.setLevel(logging.WARNING)
logger.addHandler(file_handler)
logger.propagate = False


class GLOBAL_DIM_KEYS(Enum):
    TIME = 'time'
    EPOCHS = 'epochs'

NULL_VALUES = [None, 'none', 'nan', 0, '0', '']
"""
Class used to create a object of a signal in which data and metadata are stored.
Info like patient number and trial number are extracted from the file name if not provided.
To guide the data into processing functions the meaning of each dimension can be provided via a dictionary.
Parameters:
- unpacked_data: numpy array containing data
- fs: Sampling frequency of the signal
- dim_dict: Dictionary mapping dimension indices to their meanings (e.g., {'rows': 0, 'cols': 1, 'frequencies': 2, 'time': 3, 'epochs': 4})
- patient_number: Optional patient identifier
- trial_number: Optional trial identifier
- file_name: Optional file name to extract patient and trial identifiers if not provided 
"""
class SignalObject:
    def __init__(self, unpacked_data: np.ndarray, fs: int, dim_dict: Dict[str, int], patient: Union[int, str], trial: Union[int, str]):
        self.signal = unpacked_data
        self.patient = patient
        self.trial = trial
        self._validate_fs(fs)

        self._validate_dim_dict_dimensions(dim_dict)

    def _validate_fs(self, fs):
        if not isinstance(fs, int) or fs <= 0:
            raise ValueError("Sampling frequency (fs) must be a positive integer.")
        self.fs = fs
    
    def _validate_dim_dict_dimensions(self, dim_dict: Dict):
        """
        Validate the dimension dictionary to ensure it contains valid keys and indices.
        Used the DIM_DICT_KEYS enum of the current child class to check for valid keys.
        """
        if dim_dict is None:
            raise ValueError("Dimension dictionary must be provided.")

        if len(dim_dict.keys()) != self.signal.ndim:
            raise ValueError(f"Dimension dictionary length ({len(dim_dict.keys())}) does not match signal dimensions ({self.signal.ndim}).")
            
        if len(set(dim_dict.values())) != len(dim_dict.values()):
            raise ValueError("Dimension indices must be unique.")

        if any(idx < 0 or idx >= self.signal.ndim for idx in dim_dict.values()):
            raise ValueError(f"Dimension indices must be within the range of signal dimensions [0-{self.signal.ndim - 1}].")

        if not isinstance(any(dim_dict.values()), int):
            raise ValueError("All dimension indices must be integers.")
        
        self.dim_dict = dim_dict
    

    def _apply_dim_dict(self):
        self._validate_dim_dict()
        for dim_key, dim_idx in self.dim_dict.items():
            dim_size = self.signal.shape[dim_idx]
            setattr(self, dim_key, dim_size)

    def _edit_dim_dict(self, new_dim_dict: Dict[str, int]):
        self._validate_dim_dict_dimensions(new_dim_dict)
        self._apply_dim_dict()

    def _switch_dim_dict_keys(self, key_map: Dict[str, str]):
        """
        Switch keys in the dimension dictionary according to key_map.
        WARNING: This does not change the order of dimensions in the signal array.
        """
        new_dim_dict = {}
        for old_key, new_key in key_map.items():
            if old_key in self.dim_dict:
                new_dim_dict[new_key] = self.dim_dict[old_key]
            else:
                raise ValueError(f"Old key '{old_key}' not found in current dimension dictionary.")
        self._edit_dim_dict(new_dim_dict)

    def _reorder_signal_dimensions(self, new_order: List[str]):
        """
        Reorder signal dimensions to match the specified dimension names order.
        Only the specified dimensions are reordered, others remain in their current positions.
        
        Parameters:
        - new_order: List of dimension names in desired order (e.g., ['rows', 'cols'])
        """
        # Validate that all specified dimensions exist in dim_dict
        for dim_name in new_order:
            if dim_name not in self.dim_dict:
                raise ValueError(f"Dimension '{dim_name}' not found in current dimension dictionary.")
        
        # Get current dimension indices for the specified order
        current_indices = [self.dim_dict[dim_name] for dim_name in new_order]
        
        # Create the axes permutation
        axes_order = list(range(self.signal.ndim))
        
        # Replace the first len(new_order) positions with the specified dimensions
        for i, current_idx in enumerate(current_indices):
            axes_order[i] = current_idx
        
        # Fill remaining positions with unspecified dimensions
        specified_indices = set(current_indices)
        remaining_indices = [i for i in range(self.signal.ndim) if i not in specified_indices]
        
        for i, remaining_idx in enumerate(remaining_indices):
            axes_order[len(new_order) + i] = remaining_idx
        
        # Apply the transpose
        self.signal = np.transpose(self.signal, axes=axes_order)
        
        # Update dim_dict to reflect new order
        new_dim_dict = {}
        for i, dim_name in enumerate(new_order):
            new_dim_dict[dim_name] = i
        
        # Add remaining dimensions
        remaining_dims = [dim for dim in self.dim_dict.keys() if dim not in new_order]
        for i, dim_name in enumerate(remaining_dims):
            new_dim_dict[dim_name] = len(new_order) + i
        
        self._edit_dim_dict(new_dim_dict)


    def _check_dim_order(self, required_order: List[str]) -> bool:
        """
        Check if the dimensions of the SignalObject are in the required order.
        Parameters:
        - Signal: EegSignal object
        - required_order: List of dimension names in the required order
        Returns:
        - True if dimensions are in the required order, False otherwise
        """
        current_order = [key for key, _ in sorted(self.dim_dict.items(), key=lambda item: item[1])]
        return current_order == required_order

    def _validate_dim_dict(self):
        pass

    class DIM_DICT_KEYS(Enum):
        pass

        


class EegSignal(SignalObject):
    def __init__(self, electrode_schema: np.ndarray = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.electrode_schema = electrode_schema
        self.is_spatial_signal: bool = None
        self.seconds_length: float = None

        self._apply_dim_dict()

        if isinstance(electrode_schema, np.ndarray) and electrode_schema is not None:
            self._validate_electrode_schema()
            self._infer_spatial_info()
        
        self._infer_temporal_info()

    def _validate_dim_dict(self):
        rows_attr = self.DIM_DICT_KEYS.ROWS.value
        cols_attr = self.DIM_DICT_KEYS.COLS.value
        chan_attr = self.DIM_DICT_KEYS.CHANNELS.value
        # dim dict should contain either (rows and cols) or channels

        possible_keys = {key.value for key in self.DIM_DICT_KEYS.__members__.values()}
        for key in self.dim_dict.keys():
            if key not in possible_keys:
                raise ValueError(f"Invalid dimension key: {key}. Must be one of {possible_keys}.")
            
        if (rows_attr in self.dim_dict and cols_attr in self.dim_dict) and not chan_attr in self.dim_dict:
            self.is_spatial_signal = True
        elif chan_attr in self.dim_dict and not (rows_attr in self.dim_dict or cols_attr in self.dim_dict):
            self.is_spatial_signal = False
        else:
            raise ValueError("Invalid dimension dictionary configuration. Dict should contain either (rows and cols) or channels")


    def _validate_electrode_schema(self):
        rows_attr = self.DIM_DICT_KEYS.ROWS.value
        cols_attr = self.DIM_DICT_KEYS.COLS.value
        
        if self.is_spatial_signal:
            if self.electrode_schema.ndim != 2:
                raise ValueError("Electrode schema of a spatial signal must be a 2D array.")

            # Check if shape of schema and shape of rows by cols in dim_dict match
            if getattr(self, rows_attr) != self.electrode_schema.shape[0] or getattr(self, cols_attr) != self.electrode_schema.shape[1]:
                raise ValueError("Electrode schema rows and cols don't match signal shape")
        else:
            if len(self.electrode_schema) != self.channels:
                raise ValueError("Electrode schema and channel dimension size don't match")

    def _infer_spatial_info(self):
        channels_attr = self.DIM_DICT_KEYS.CHANNELS.value
        rows_attr = self.DIM_DICT_KEYS.ROWS.value
        cols_attr = self.DIM_DICT_KEYS.COLS.value
        if self.is_spatial_signal:
            # Removing placeholder channels from eeg_channels
            self.placeholder_channels = []
            # For spatial signals, electrode_schema is a 2D np.ndarray
            for row_idx in range(self.electrode_schema.shape[0]):
                for col_idx in range(self.electrode_schema.shape[1]):
                    channel = self.electrode_schema[row_idx, col_idx]
                    if str(channel).lower() in NULL_VALUES:
                        self.placeholder_channels.append((row_idx, col_idx))
            self.eeg_channels = getattr(self, rows_attr) * getattr(self, cols_attr) - len(self.placeholder_channels)
        else:
            self.eeg_channels = getattr(self, channels_attr)
            self.placeholder_channels = []
    
    def _infer_temporal_info(self):
        time_attr = self.DIM_DICT_KEYS.TIME.value
        epochs_attr = self.DIM_DICT_KEYS.EPOCHS.value
        
        if hasattr(self, time_attr):
            self.seconds_length = getattr(self, time_attr) / self.fs

            if hasattr(self, epochs_attr):
                self.seconds_total_length = getattr(self, epochs_attr) * getattr(self, time_attr) / self.fs
        else:
            logger.warning("No time dimension found in dim_dict")


    class DIM_DICT_KEYS(Enum):
        ROWS = 'rows'
        COLS = 'cols'
        CHANNELS = 'channels'
        FREQUENCIES = 'frequencies'
        TIME = GLOBAL_DIM_KEYS.TIME.value
        EPOCHS = GLOBAL_DIM_KEYS.EPOCHS.value
        

        
class KinematicSignal(SignalObject):
    def __init__(self, fs: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._apply_dim_dict()

    class DIM_DICT_KEYS(Enum):
        X = 'x'
        Y = 'y'
        Z = 'z'
        ALPHA = 'alpha'
        BETA = 'beta'
        GAMMA = 'gamma'
        ROLL = 'roll'
        PITCH = 'pitch'
        YAW = 'yaw'
        TIME = GLOBAL_DIM_KEYS.TIME.value
        EPOCHS = GLOBAL_DIM_KEYS.EPOCHS.value



"""
This class combines multiple eterogenic time series of instances of SignalObject.
Handles the synchronization of all the time series when preprocessed.
"""
class MultimodalTimeSignal():
    def __init__(self, signals: List[SignalObject]):

        self._check_multiple_signals(signals)

        self.signals = signals
        self.num_signals = len(signals)

        self._validate_time_series()
        self._check_sampling()

    def _check_multiple_signals(self, signals: List[SignalObject]):
        if not isinstance(signals, list) or len(signals) < 2:
            raise ValueError("signals must be a list with at least two SignalObject instances.")
        
        for i, signal in enumerate(signals):
            if not isinstance(signal, SignalObject):
                raise ValueError(f"Element at index {i} is not an instance of SignalObject.")

    def _validate_time_series(self):
        # Check that all signals have TIME dimension
        time_attr = GLOBAL_DIM_KEYS.TIME.value
        for i, signal in enumerate(self.signals):
            if  not hasattr(signal, time_attr):
                raise ValueError(f"Signal at index {i} missing required TIME dimension")

        # Get time values from all signals
        time_values = [getattr(signal, time_attr) for signal in self.signals]

        # Check that all time values are equal
        if not all(time_val == time_values[0] for time_val in time_values):
            raise ValueError("All signals must have the same TIME dimension size")

        # Check epochs consistency
        signals_with_epochs = [signal for signal in self.signals if time_attr in signal.dim_dict]

        if signals_with_epochs:
            # If any signal has epochs, all must have epochs
            if len(signals_with_epochs) != len(self.signals):
                raise ValueError("If any signal has EPOCHS dimension, all signals must have it")
            
            # Check that all epoch values are equal
            epoch_values = [getattr(signal, time_attr) for signal in signals_with_epochs]
            if not all(epoch_val == epoch_values[0] for epoch_val in epoch_values):
                raise ValueError("All signals must have the same EPOCHS dimension size")

    def _check_sampling(self):
        # Check that all signals have the same sampling frequency
        fs_values = [signal.fs for signal in self.signals]
        if not all(fs_val == fs_values[0] for fs_val in fs_values):
            logger.warning("Found signals with different sampling frequency")



class RandomSignal(SignalObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    class DIM_DICT_KEYS(Enum):
        CHANNELS = 'channels'
        DIM1 = 'dim1'
        DIM2 = 'dim2'
        FREQUENCIES = 'frequencies'
        TIME = GLOBAL_DIM_KEYS.TIME.value
        EPOCHS = GLOBAL_DIM_KEYS.EPOCHS.value

    @classmethod
    def spatial(cls, signal_shape: Tuple[int] = (2, 2, 2), fs: int = 250):
        unpacked_data = np.random.rand(*signal_shape)
        dim_dict = {"rows": 0, "cols": 1, "time": 2, "frequencies": 3}
        
        instance = cls(unpacked_data, fs, dim_dict, "random", "random")
        instance.electrode_schema = [[f'C{x}{y}' for x in range(signal_shape[dim_dict["rows"]])] for y in range(signal_shape[dim_dict["cols"]])]
        return instance

    @classmethod
    def non_spatial(cls, signal_shape: Tuple[int] = (2, 2, 2), fs: int = 250):
        unpacked_data = np.random.rand(*signal_shape)
        dim_dict = {"channels": 0, "time": 1, "frequencies": 2}
        
        instance = cls(unpacked_data, fs, dim_dict, "random", "random")
        instance.electrode_schema = [f'C{i}' for i in range(signal_shape[dim_dict["channels"]])]
        return instance