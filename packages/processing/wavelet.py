import numpy as np
import pywt
from typing import List, Tuple, Union
from functools import lru_cache

from packages.data_objects.signal import GLOBAL_DIM_KEYS, EegSignal

def raw_wavelet_transform(signal: np.ndarray, wavelet: str = 'cmor1.5-1.0', bandwidth: Tuple[float] = None, fs: float = None, num_samples: int = None, abs_out: bool = True, norm_out: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Continuous Wavelet Transform (CWT) on the input signal using the specified wavelet.
    Parameters:
    - signal: 1D or 2D numpy array (channels x samples) representing the input signal.
    - wavelet: String specifying the wavelet to use (default is 'cmor1.5-1.0').
    - bandwidth: List or tuple with two elements [min_freq, max_freq] specifying the frequency range for the transform.
    - fs: Sampling frequency of the input signal.
    Returns:
    - coefficients: 2D or 3D numpy array of wavelet coefficients (channels x frequencies x samples).
    - frequencies: 1D numpy array of frequencies corresponding to the wavelet coefficients.
    Notes:
    - If the input signal is 2D, the function assumes the first dimension represents channels
        and the second dimension represents samples. If the first dimension has more samples
        than the second, the signal is transposed.
    """


    if fs is None:
        raise ValueError("Sampling frequency 'fs' must be provided.")
    if bandwidth is None:
        raise ValueError("Frequency bandwidth 'bandwidth' must be provided as a list or tuple of two elements [min_freq, max_freq].")
    if not isinstance(bandwidth, (list, tuple)) or len(bandwidth) != 2:
        raise ValueError("Bandwidth must be a list or tuple with two elements: [min_freq, max_freq].")
    if bandwidth[0] <= 0 or bandwidth[1] <= 0 or bandwidth[0] >= bandwidth[1]:
        raise ValueError("Bandwidth values must be positive and min_freq must be less than max_freq.")
    
    # Using 1 Hz resolution by default
    if num_samples is None:
        num_samples = bandwidth[1] - bandwidth[0] + 1

    scales = _compute_scales(wavelet, fs, bandwidth, num_samples)


    # Multichannels case
    if signal.ndim == 2:
        if signal.shape[0] > signal.shape[1]:
            # Setting as default channel the dimension with the least samples
            signal = signal.T


        all_coeffs = np.zeros((signal.shape[0], num_samples, max(signal.shape)), dtype=complex)
        for i in range(signal.shape[0]):
            coeffs, freqs = pywt.cwt(signal[i, :], scales, wavelet)
            all_coeffs[i, :, :] = coeffs

        if abs_out:
            all_coeffs = abs(all_coeffs)

        if norm_out:
            # Normalize coefficients for each frequency using z normalization
            all_coeffs = (all_coeffs - np.mean(all_coeffs, axis=1, keepdims=True)) / (np.std(all_coeffs, axis=1, keepdims=True) + 1e-10)

        return all_coeffs, freqs
    
    # Single channel case
    elif signal.ndim == 1:     
        coefficients, frequencies = pywt.cwt(signal, scales, wavelet)

        if abs_out:
            coefficients = abs(coefficients)

        if norm_out:
            # Normalize coefficients for each frequency using z normalization
            coefficients = (coefficients - np.mean(coefficients, axis=1, keepdims=True)) / (np.std(coefficients, axis=1, keepdims=True) + 1e-10)

        return coefficients, frequencies
    
    else:
        raise ValueError(f"Input signal must be 1D or 2D array, got {signal.ndim}D array.")
    


def eeg_wavelet_transform(EegSignal: EegSignal, bandwidth: Tuple[float, float], wavelet: str = 'cmor1.5-1.0', freq_samples: int = None, abs_out: bool = True, norm_out: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Continuous Wavelet Transform (CWT) on the input EegSignal using the specified wavelet.
    Parameters:
    - signal: EegSignal object representing the input EEG signal.
    - wavelet: String specifying the wavelet to use (default is 'cmor1.5-1.0').
    - bandwidth: List or tuple with two elements [min_freq, max_freq] specifying the frequency range for the transform.
    Returns:
    - coefficients: 2D or 3D numpy array of wavelet coefficients (channels x frequencies x samples).
    - frequencies: 1D numpy array of frequencies corresponding to the wavelet coefficients.
    Notes:
    - If the input signal is 2D, the function assumes the first dimension represents channels
        and the second dimension represents samples. If the first dimension has more samples
        than the second, the signal is transposed.
    """

    _validate_eeg_signal(EegSignal)

    fs = EegSignal.fs
    signal = EegSignal.signal

    _validate_bandwidth(bandwidth)
    
    # Using 1 Hz resolution by default
    if freq_samples is None:
        freq_samples = bandwidth[1] - bandwidth[0] + 1

    scales = _compute_scales(wavelet, fs, bandwidth, freq_samples)



    # Spatial Multichannel case
    if EegSignal.is_spatial_signal:
        
        rows_key = EegSignal.DIM_DICT_KEYS.ROWS.value
        cols_key = EegSignal.DIM_DICT_KEYS.COLS.value
        time_key = EegSignal.DIM_DICT_KEYS.TIME.value
        freq_key = EegSignal.DIM_DICT_KEYS.FREQUENCIES.value

        time_dim = EegSignal.dim_dict.get(time_key, None)

        rows_dim, cols_dim = EegSignal.dim_dict.get(rows_key, None), EegSignal.dim_dict.get(cols_key, None)

        num_rows, num_cols, num_times = EegSignal.rows, EegSignal.cols, EegSignal.time

        all_coeffs, freqs = _spatial_wavelet(signal, scales, wavelet, num_rows, num_cols, num_times, time_dim, rows_dim, cols_dim)

        # Update EegSignal attributes
        EegSignal._edit_dim_dict({rows_key: 0, cols_key: 1, freq_key: 2, time_key: 3})
        EegSignal.signal = all_coeffs
        EegSignal.wavelet_frequencies = freqs

    # Non spatial case
    elif not EegSignal.is_spatial_signal:
        chan_key = EegSignal.DIM_DICT_KEYS.CHANNELS.value
        time_key = EegSignal.DIM_DICT_KEYS.TIME.value
        freq_key = EegSignal.DIM_DICT_KEYS.FREQUENCIES.value

        time_dim = EegSignal.dim_dict.get(time_key, None)
        chan_dim = EegSignal.dim_dict.get(chan_key, None)

        num_chans, num_times = EegSignal.channels, EegSignal.time
        
        all_coeffs, freqs = _channelwise_wavelet(signal, scales, wavelet)

        # Update EegSignal attributes
        EegSignal._edit_dim_dict({chan_key: 0, freq_key: 1, time_key: 2})
        EegSignal.signal = all_coeffs
        EegSignal.wavelet_frequencies = freqs
    
    return EegSignal

# MARK: Validation
def _validate_bandwidth(bandwidth: Union[List[float], Tuple[float, float]]) -> None:
    if not isinstance(bandwidth, (list, tuple)) or len(bandwidth) != 2:
        raise ValueError("Bandwidth must be a list or tuple with two elements: [min_freq, max_freq].")
    if bandwidth[0] <= 0 or bandwidth[1] <= 0 or bandwidth[0] >= bandwidth[1]:
        raise ValueError("Bandwidth values must be positive and min_freq must be less than max_freq.")

def _validate_eeg_signal(EegSignal: EegSignal) -> None:
    """
    Validate that the input is an instance of EegSignal.
    Dim_dict must contain TIME key but not FREQUENCY key.

    Waveletable signals pass this validation.
    """
    if not isinstance(EegSignal, EegSignal):
        raise ValueError("Input must be an instance of EegSignal.")
    if EegSignal.DIM_DICT_KEYS.TIME.value not in EegSignal.dim_dict:
        raise ValueError("EegSignal.dim_dict must contain TIME key.")
    if EegSignal.DIM_DICT_KEYS.FREQUENCIES.value in EegSignal.dim_dict:
        raise ValueError("EegSignal.dim_dict must not contain FREQUENCY key.")


# MARK: Helper Functions
@lru_cache(maxsize=4)
def _compute_scales(wavelet, fs, bandwidth_tuple, num_samples):
    frequencies = np.linspace(bandwidth_tuple[0], bandwidth_tuple[1], num=num_samples)
    return pywt.frequency2scale(wavelet, frequencies/fs)

def _spatial_wavelet(all_coeffs, signal, scales, wavelet, num_rows, num_cols, num_times, time_dim, rows_dim, cols_dim):
    """
    Apply wavelet transform to each spatial location in the signal.
    Parameters:
    - signal: 3D numpy array (rows x cols x time) representing the spatial signal.
    - scales: 1D numpy array of scales for the wavelet transform.
    - wavelet: String specifying the wavelet to use.
    - num_rows: Number of rows in the spatial grid.
    - num_cols: Number of columns in the spatial grid.
    - num_times: Number of time points in the signal.
    - time_dim: Dimension index for time in the signal array.
    - rows_dim: Dimension index for rows in the signal array.
    - cols_dim: Dimension index for columns in the signal array.
    Returns:
    - all_coeffs: 4D numpy array (rows x cols x frequencies x times) of wavelet coefficients.
    """
    all_coeffs = np.zeros((num_rows, num_cols, scales, num_times), dtype=complex)

    for i in range(num_rows):
        for j in range(num_cols):
            # Create slice indices for the signal based on dimension positions
            signal_slice = [slice(None)] * signal.ndim
            signal_slice[rows_dim] = i
            signal_slice[cols_dim] = j
            signal_slice[time_dim] = slice(None)
            
            coeffs, freqs = pywt.cwt(signal[tuple(signal_slice)], scales, wavelet)
            all_coeffs[i, j, :, :] = coeffs
    return all_coeffs, freqs

def _channelwise_wavelet(signal, scales, wavelet):
    """
    Apply wavelet transform to each channel in the signal.
    Parameters:
    - signal: 2D numpy array representing the multichannel signal
    - scales: 1D numpy array of scales for the wavelet transform.
    - wavelet: String specifying the wavelet to use.
    Returns:
    - all_coeffs: 3D numpy array (channels x frequencies x times) of wavelet coefficients.
    """
    all_coeffs = np.zeros((num_chans, scales, num_times), dtype=complex)
    for i in range(num_chans):
        signal_slice = [slice(None)] * signal.ndim
        signal_slice[chan_dim] = i
        signal_slice[time_dim] = slice(None)

        coeffs, freqs = pywt.cwt(signal[tuple(signal_slice)], scales, wavelet)
        all_coeffs[i, :, :] = coeffs
    return all_coeffs, freqs