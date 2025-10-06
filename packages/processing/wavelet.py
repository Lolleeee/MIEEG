from functools import lru_cache
from typing import List, Tuple, Union

import numpy as np
import pywt

from packages.data_objects.signal import GLOBAL_DIM_KEYS, EegSignal, SignalObject


def eeg_wavelet_transform(
    Signal: SignalObject,
    bandwidth: Tuple[float, float],
    wavelet: str = "cmor1.5-1.0",
    freq_samples: int = None,
) -> tuple[np.ndarray, np.ndarray]:
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

    _validate_signal(Signal)

    fs = Signal.fs
    signal = Signal.signal

    bandwidth_min, bandwidth_max = _validate_bandwidth(bandwidth)

    # Using 1 Hz resolution by default
    if freq_samples is None:
        freq_samples = bandwidth_max - bandwidth_min + 1

    scales = _compute_scales(wavelet, fs, bandwidth_min, bandwidth_max, freq_samples)

    # Spatial Multichannel case
    if Signal.is_spatial_signal:
        rows_key = Signal.DIM_DICT_KEYS.ROWS.value
        cols_key = Signal.DIM_DICT_KEYS.COLS.value
        time_key = Signal.DIM_DICT_KEYS.TIME.value
        freq_key = Signal.DIM_DICT_KEYS.FREQUENCIES.value

        time_dim = Signal.dim_dict.get(time_key, None)

        rows_dim, cols_dim = (
            Signal.dim_dict.get(rows_key, None),
            Signal.dim_dict.get(cols_key, None),
        )

        num_rows, num_cols, num_times = Signal.rows, Signal.cols, Signal.time

        all_coeffs, freqs = _spatial_wavelet(
            signal,
            scales,
            wavelet,
            num_rows,
            num_cols,
            num_times,
            time_dim,
            rows_dim,
            cols_dim,
        )

        # Update EegSignal attributes
        Signal.signal = all_coeffs

        Signal._edit_dim_dict({rows_key: 0, cols_key: 1, freq_key: 2, time_key: 3})

        Signal.wavelet_frequencies = freqs

    # Non spatial case
    elif not Signal.is_spatial_signal:
        chan_key = Signal.DIM_DICT_KEYS.CHANNELS.value
        time_key = Signal.DIM_DICT_KEYS.TIME.value
        freq_key = Signal.DIM_DICT_KEYS.FREQUENCIES.value

        time_dim = Signal.dim_dict.get(time_key, None)
        chan_dim = Signal.dim_dict.get(chan_key, None)

        num_chans, num_times = Signal.channels, Signal.time

        all_coeffs, freqs = _channelwise_wavelet(
            signal, scales, wavelet, chan_dim, time_dim, num_chans, num_times
        )

        # Update EegSignal attributes
        Signal.signal = all_coeffs

        Signal._edit_dim_dict({chan_key: 0, freq_key: 1, time_key: 2})

        Signal.wavelet_frequencies = freqs

    return Signal


# MARK: Validation
def _validate_bandwidth(
    bandwidth: Union[List[float], Tuple[float, float]],
) -> Tuple[float, float]:
    if not isinstance(bandwidth, (list, tuple)) or len(bandwidth) != 2:
        raise ValueError(
            "Bandwidth must be a list or tuple with two elements: [min_freq, max_freq]."
        )
    if bandwidth[0] <= 0 or bandwidth[1] <= 0 or bandwidth[0] >= bandwidth[1]:
        raise ValueError(
            "Bandwidth values must be positive and min_freq must be less than max_freq."
        )

    bandwidth_min, bandwidth_max = bandwidth[0], bandwidth[1]
    return bandwidth_min, bandwidth_max


def _validate_signal(Signal: SignalObject) -> None:
    """
    Validate that the input is an instance of EegSignal.
    Dim_dict must contain TIME key but not FREQUENCY key.

    Waveletable signals pass this validation.
    """
    if not isinstance(Signal, SignalObject):
        raise ValueError("Input must be an instance of SignalObject.")
    if Signal.DIM_DICT_KEYS.TIME.value not in Signal.dim_dict:
        raise ValueError("SignalObject.dim_dict must contain TIME key.")
    if Signal.DIM_DICT_KEYS.FREQUENCIES.value in Signal.dim_dict:
        raise ValueError("SignalObject.dim_dict must not contain FREQUENCY key.")


# MARK: Helper Functions
@lru_cache(maxsize=4)
def _compute_scales(wavelet, fs, bandwidth_min, bandwidth_max, num_samples):
    frequencies = np.linspace(bandwidth_min, bandwidth_max, num=num_samples)
    return pywt.frequency2scale(wavelet, frequencies / fs)


def _spatial_wavelet(
    all_coeffs,
    signal,
    scales,
    wavelet,
    num_rows,
    num_cols,
    num_times,
    time_dim,
    rows_dim,
    cols_dim,
):
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
    all_coeffs = np.zeros((num_rows, num_cols, len(scales), num_times), dtype=complex)

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


def _channelwise_wavelet(
    signal, scales, wavelet, chan_dim, time_dim, num_chans, num_times
):
    """
    Apply wavelet transform to each channel in the signal.
    Parameters:
    - signal: 2D numpy array representing the multichannel signal
    - scales: 1D numpy array of scales for the wavelet transform.
    - wavelet: String specifying the wavelet to use.
    - chan_dim: Dimension index for channels in the signal array.
    - time_dim: Dimension index for time in the signal array.
    - num_chans: Number of channels in the signal.
    - num_times: Number of time points in the signal.
    Returns:
    - all_coeffs: 3D numpy array (channels x frequencies x times) of wavelet coefficients.
    """
    all_coeffs = np.zeros((num_chans, len(scales), num_times), dtype=complex)
    for i in range(num_chans):
        signal_slice = [slice(None)] * signal.ndim
        signal_slice[chan_dim] = i
        signal_slice[time_dim] = slice(None)

        coeffs, freqs = pywt.cwt(signal[tuple(signal_slice)], scales, wavelet)
        all_coeffs[i, :, :] = coeffs
    return all_coeffs, freqs
