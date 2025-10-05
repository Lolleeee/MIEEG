from packages.data_objects.signal import SignalObject, EegSignal, GLOBAL_DIM_KEYS
from typing import Dict, List, Tuple
import numpy as np
from enum import Enum

from packages.data_objects.signal import SignalObject, GLOBAL_DIM_KEYS
from typing import Dict, List, Tuple, Optional
import numpy as np
from enum import Enum

class RandomSignal(SignalObject):
    class DIM_DICT_KEYS(Enum):
        CHANNELS = 'channels'
        ROWS = 'rows'
        COLS = 'cols'
        TIME = GLOBAL_DIM_KEYS.TIME.value
        EPOCHS = GLOBAL_DIM_KEYS.EPOCHS.value
        FREQUENCIES = 'frequencies'

    def __init__(
        self,
        unpacked_data: np.ndarray,
        fs: int,
        dim_dict: Dict[str, int],
        patient: str = "random",
        trial: str = "random",
        is_spatial_signal: bool = False,
        electrode_schema: Optional[np.ndarray] = None
    ):
        super().__init__(unpacked_data, fs, dim_dict, patient, trial)
        self.is_spatial_signal = is_spatial_signal
        
        # Generate electrode schema if not provided
        if electrode_schema is not None:
            self.electrode_schema = electrode_schema
        else:
            self._generate_electrode_schema(dim_dict, unpacked_data.shape)

    def _generate_electrode_schema(self, dim_dict: Dict[str, int], shape: Tuple[int]):
        """Generate default electrode schema based on signal type."""
        if self.is_spatial_signal and 'rows' in dim_dict and 'cols' in dim_dict:
            n_rows = shape[dim_dict['rows']]
            n_cols = shape[dim_dict['cols']]
            self.electrode_schema = np.array(
                [[f"E{r}_{c}" for c in range(n_cols)] for r in range(n_rows)],
                dtype=object
            )
        elif 'channels' in dim_dict:
            n_channels = shape[dim_dict['channels']]
            self.electrode_schema = np.array([f"Ch{i}" for i in range(n_channels)])
        else:
            self.electrode_schema = np.array([])

    @classmethod
    def spatial(
        cls,
        signal_shape: Tuple[int, ...] = (5, 5, 1000),
        fs: int = 250,
        electrode_schema: Optional[np.ndarray] = None,
        dim_dict: Optional[Dict[str, int]] = None
    ):
        """
        Create a random spatial signal (rows × cols × time).
        
        Args:
            signal_shape: Shape of the signal (rows, cols, time, ...)
            fs: Sampling frequency
            electrode_schema: Custom electrode layout (2D array)
            dim_dict: Custom dimension dictionary
        
        Returns:
            RandomSignal instance with spatial configuration
        """
        if dim_dict is None:
            dim_dict = {"rows": 0, "cols": 1, "time": 2}
        
        unpacked_data = np.random.randn(*signal_shape)
        
        return cls(
            unpacked_data=unpacked_data,
            fs=fs,
            dim_dict=dim_dict,
            is_spatial_signal=True,
            electrode_schema=electrode_schema
        )

    @classmethod
    def non_spatial(
        cls,
        signal_shape: Tuple[int, ...] = (32, 1000),
        fs: int = 250,
        electrode_schema: Optional[np.ndarray] = None,
        dim_dict: Optional[Dict[str, int]] = None
    ):
        """
        Create a random non-spatial signal (channels × time).
        
        Args:
            signal_shape: Shape of the signal (channels, time, ...)
            fs: Sampling frequency
            electrode_schema: Custom electrode names (1D array)
            dim_dict: Custom dimension dictionary
        
        Returns:
            RandomSignal instance with channel configuration
        """
        if dim_dict is None:
            dim_dict = {"channels": 0, "time": 1}
        
        unpacked_data = np.random.randn(*signal_shape)
        
        return cls(
            unpacked_data=unpacked_data,
            fs=fs,
            dim_dict=dim_dict,
            is_spatial_signal=False,
            electrode_schema=electrode_schema
        )

    @classmethod
    def with_frequencies(
        cls,
        signal_shape: Tuple[int, ...] = (32, 50, 1000),
        fs: int = 250,
        is_spatial: bool = False,
        dim_dict: Optional[Dict[str, int]] = None
    ):
        """
        Create a random signal with frequency dimension (e.g., after wavelet transform).
        
        Args:
            signal_shape: Shape (channels, frequencies, time) or (rows, cols, frequencies, time)
            fs: Sampling frequency
            is_spatial: Whether signal has spatial layout
            dim_dict: Custom dimension dictionary
        
        Returns:
            RandomSignal instance with frequency dimension
        """
        if dim_dict is None:
            if is_spatial:
                dim_dict = {"rows": 0, "cols": 1, "frequencies": 2, "time": 3}
            else:
                dim_dict = {"channels": 0, "frequencies": 1, "time": 2}
        
        unpacked_data = np.random.randn(*signal_shape)
        
        return cls(
            unpacked_data=unpacked_data,
            fs=fs,
            dim_dict=dim_dict,
            is_spatial_signal=is_spatial
        )