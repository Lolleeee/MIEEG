import numpy as np
import pywt
from typing import List


# This function transforms the input signal using wavelet transform
def wavelet_transform(signal: np.ndarray, wavelet: str = 'cmor1.5-1.0', bandwidth: List[float] = None, fs: float = None, num_samples: int = None, abs_out: bool = True, norm_out: bool = True) -> tuple[np.ndarray, np.ndarray]:
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


    frequencies = np.linspace(bandwidth[0], bandwidth[1], num=num_samples)
    scales = pywt.frequency2scale(wavelet, frequencies/fs)


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


