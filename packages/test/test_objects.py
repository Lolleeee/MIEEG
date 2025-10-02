from packages.data_objects.signal import SignalObject, EegSignal, GLOBAL_DIM_KEYS
from typing import Dict, List, Tuple
import numpy as np
from enum import Enum

class RandomSignal(SignalObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    class DIM_DICT_KEYS(Enum):
        CHANNELS = 'channels'
        DIM1 = 'dim1'
        DIM2 = 'dim2'
        FREQUENCIES = 'frequencies'
        ROWS = 'rows'
        COLS = 'cols'
        TIME = GLOBAL_DIM_KEYS.TIME.value
        EPOCHS = GLOBAL_DIM_KEYS.EPOCHS.value

    @classmethod
    def spatial(cls, signal_shape: Tuple[int] = (2, 2, 2), fs: int = 250, electrode_schema: List[str] = None, dim_dict: Dict = None):
        unpacked_data = np.random.rand(*signal_shape)
        
        if dim_dict is None:
            dim_dict = {"rows": 0, "cols": 1, "time": 2, "frequencies": 3}

        instance = cls(unpacked_data, fs, dim_dict, "random", "random")
        instance.is_spatial_signal = True
        instance._apply_dim_dict(dim_dict)
        if electrode_schema is not None:
            instance.electrode_schema = electrode_schema
        else:
            instance.electrode_schema = [[f'C{x}{y}' for x in range(signal_shape[dim_dict["rows"]])] for y in range(signal_shape[dim_dict["cols"]])]
        return instance

    @classmethod
    def non_spatial(cls, signal_shape: Tuple[int] = (2, 2, 2), fs: int = 250, electrode_schema: List[str] = None, dim_dict: Dict = None):
        unpacked_data = np.random.rand(*signal_shape)

        if dim_dict is None:
            dim_dict = {"channels": 0, "time": 1, "frequencies": 2}
        instance._apply_dim_dict(dim_dict)
        instance = cls(unpacked_data, fs, dim_dict, "random", "random")
        instance.is_spatial_signal = False
        if electrode_schema is not None:
            instance.electrode_schema = electrode_schema
        else:
            instance.electrode_schema = [f'C{i}' for i in range(signal_shape[dim_dict["channels"]])]
        return instance