from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_reconstruction_distribution(
    original: Union[np.ndarray, torch.Tensor],
    reconstructed: Union[np.ndarray, torch.Tensor],
) -> None:
    """
    Plot the distribution of original and reconstructed signals using histograms,
    and plots the distributions of the differences.
    Parameters:
    - original: Original signal as a numpy array or torch tensor.
    - reconstructed: Reconstructed signal as a numpy array or torch tensor.
    """

    if isinstance(original, torch.Tensor):
        original = original.cpu().detach().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.cpu().detach().numpy()
    difference = original - reconstructed

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.hist(
        original.flatten(),
        bins=100,
        alpha=0.5,
        label="Original",
        color="blue",
        density=True,
    )
    plt.hist(
        reconstructed.flatten(),
        bins=100,
        alpha=0.5,
        label="Reconstructed",
        color="orange",
        density=True,
    )
    plt.title("Original vs Reconstructed Signal Distribution")
    plt.xlabel("Signal Value")
    plt.ylabel("Density")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.hist(difference.flatten(), bins=100, alpha=0.7, color="red", density=True)
    plt.title("Difference Distribution (Original - Reconstructed)")
    plt.xlabel("Difference Value")
    plt.ylabel("Density")

    plt.subplot(3, 1, 3)
    plt.hist(
        np.abs(difference).flatten(), bins=100, alpha=0.7, color="green", density=True
    )
    plt.title("Absolute Difference Distribution |Original - Reconstructed|")
    plt.xlabel("Absolute Difference Value")
    plt.ylabel("Density")

    plt.tight_layout()
    plt.show()

def plot_reconstruction_scatter(
    original: Union[np.ndarray, torch.Tensor],
    reconstructed: Union[np.ndarray, torch.Tensor],
) -> None:
    """
    Plot a scatter plot comparing original and reconstructed signals.
    Parameters:
    - original: Original signal as a numpy array or torch tensor.
    - reconstructed: Reconstructed signal as a numpy array or torch tensor.
    """

    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.scatter(
        original.flatten(),
        reconstructed.flatten(),
        alpha=0.5,
        color="purple",
        s=1,
    )
    plt.plot(
        [original.min(), original.max()],
        [original.min(), original.max()],
        color="red",
        linestyle="--",
        label="y=x",
    )
    plt.title("Scatter Plot of Original vs Reconstructed Signals")
    plt.xlabel("Original Signal")
    plt.ylabel("Reconstructed Signal")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def plot_reconstruction_slices(
    original: Union[np.ndarray, torch.Tensor],
    reconstructed: Union[np.ndarray, torch.Tensor], 
    freqs: List[int] = None,
    n_channels: int = 4
) -> None:
    """
    Plots slices of the original and reconstructed 4D signals for visual comparison.
    Concats the second and third dimensions which correspond to EEG channels.
    Selects 3 indexes in the middle of the frequency (first) dimension like [11, 12, 13].
    Then for n_channels random channels the reconstruction vs the original is plotted.
    
    Args:
        original: 4D array (freq, channels, channels, time)
        reconstructed: 4D array (freq, channels, channels, time)
        freqs: List of 3 frequency indices to plot. If None, uses middle frequencies.
        n_channels: Number of random channels to plot (default: 4)
    """
    import matplotlib.pyplot as plt
    
    if isinstance(original, torch.Tensor):
        original = original.cpu().detach().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.cpu().detach().numpy()
    
    assert original.shape == reconstructed.shape, "Original and reconstructed shapes must match."
    
    # Handle batch dimension or extra dimensions
    if len(original.shape) > 4:
        original = original.squeeze()
        reconstructed = reconstructed.squeeze()
        
    if len(original.shape) > 4 or len(original.shape) < 4:
        original = original[0, ...] if len(original.shape) > 4 else original
        reconstructed = reconstructed[0, ...] if len(reconstructed.shape) > 4 else reconstructed
    
    assert len(original.shape) == 4, "Input data must be 4D (freq, channels, channels, time)."
    
    freq_dim, ch1_dim, ch2_dim, time_dim = original.shape
    
    # Determine frequency indices
    if freqs is not None:
        assert len(freqs) == 3, "Must provide exactly 3 frequency indices"
        assert all(0 <= f < freq_dim for f in freqs), "Frequency indices out of bounds."
        freq_indices = freqs
    else:
        mid_freq = freq_dim // 2
        freq_indices = [mid_freq - 1, mid_freq, mid_freq + 1]
    
    # Reshape to combine channel dimensions
    combined_original = original.reshape(freq_dim, ch1_dim * ch2_dim, time_dim)
    combined_reconstructed = reconstructed.reshape(freq_dim, ch1_dim * ch2_dim, time_dim)
    
    # Select random channels
    total_channels = ch1_dim * ch2_dim
    n_channels = min(n_channels, total_channels)  # Don't exceed available channels
    random_channels = np.random.choice(total_channels, size=n_channels, replace=False)
    
    # Calculate grid dimensions
    # Each channel gets 3 plots (one per frequency), arranged horizontally
    n_cols = 3
    n_rows = n_channels
    
    # Dynamic figure height based on number of channels
    fig_height = max(3 * n_rows, 10)
    fig_width = 15
    
    plt.figure(figsize=(fig_width, fig_height))
    
    for i, ch in enumerate(random_channels):
        # Plot frequency 1
        plt.subplot(n_rows, n_cols, i * n_cols + 1)
        plt.plot(combined_original[freq_indices[0], ch, :], label="Original", color="blue")
        plt.plot(combined_reconstructed[freq_indices[0], ch, :], label="Reconstructed", color="orange", alpha=0.7)
        plt.title(f"Channel {ch} - Freq {freq_indices[0]}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        if i == 0:
            plt.legend()
        plt.grid(True)
        
        # Plot frequency 2
        plt.subplot(n_rows, n_cols, i * n_cols + 2)
        plt.plot(combined_original[freq_indices[1], ch, :], label="Original", color="blue")
        plt.plot(combined_reconstructed[freq_indices[1], ch, :], label="Reconstructed", color="orange", alpha=0.7)
        plt.title(f"Channel {ch} - Freq {freq_indices[1]}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        if i == 0:
            plt.legend()
        plt.grid(True)
        
        # Plot frequency 3
        plt.subplot(n_rows, n_cols, i * n_cols + 3)
        plt.plot(combined_original[freq_indices[2], ch, :], label="Original", color="blue")
        plt.plot(combined_reconstructed[freq_indices[2], ch, :], label="Reconstructed", color="orange", alpha=0.7)
        plt.title(f"Channel {ch} - Freq {freq_indices[2]}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        if i == 0:
            plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
