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


import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from typing import List, Union

def plot_reconstruction_slices(
    original: Union[np.ndarray, torch.Tensor],
    reconstructed: Union[np.ndarray, torch.Tensor], 
    freqs: List[int] = None,
    n_channels: int = 4
) -> None:
    """
    Plots slices of the original and reconstructed 4D signals for visual comparison.
    Concats the second and third dimensions (EEG channels).
    Selects 3 middle frequencies, unless provided.
    For n_channels random channels, plots original vs reconstructed signals.
    Displays RMSE, R², and Pearson correlation for each subplot.
    
    Args:
        original: 4D array (freq, channels, channels, time)
        reconstructed: 4D array (freq, channels, channels, time)
        freqs: List of 3 frequency indices to plot. If None, uses middle frequencies.
        n_channels: Number of random channels to plot (default: 4)
    """
    
    if isinstance(original, torch.Tensor):
        original = original.cpu().detach().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.cpu().detach().numpy()
    
    assert original.shape == reconstructed.shape, "Original and reconstructed shapes must match."
    
    # Handle batch or extra dimensions
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
    
    # Flatten channel grid
    combined_original = original.reshape(freq_dim, ch1_dim * ch2_dim, time_dim)
    combined_reconstructed = reconstructed.reshape(freq_dim, ch1_dim * ch2_dim, time_dim)
    
    # Randomly choose channels
    total_channels = ch1_dim * ch2_dim
    n_channels = min(n_channels, total_channels)
    random_channels = np.random.choice(total_channels, size=n_channels, replace=False)
    
    # Layout
    n_cols = 3
    n_rows = n_channels
    fig_height = max(3 * n_rows, 10)
    fig_width = 15
    plt.figure(figsize=(fig_width, fig_height))
    
    # Iterate channels × freqs
    for i, ch in enumerate(random_channels):
        for j, freq in enumerate(freq_indices):
            idx = i * n_cols + j + 1
            
            y_true = combined_original[freq, ch, :]
            y_pred = combined_reconstructed[freq, ch, :]
            
            # Compute metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            corr, _ = pearsonr(y_true, y_pred)
            
            plt.subplot(n_rows, n_cols, idx)
            plt.plot(y_true, label="Original", color="blue")
            plt.plot(y_pred, label="Reconstructed", color="orange", alpha=0.7)
            plt.title(
                f"Ch {ch} - Freq {freq}\n"
                f"RMSE={rmse:.3f} | R²={r2:.3f} | r={corr:.3f}",
                fontsize=10
            )
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            if i == 0 and j == 0:
                plt.legend()
            plt.grid(True)
    
    plt.tight_layout()
    plt.show()


import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

def plot_reconstruction_performance(
    original: np.ndarray | torch.Tensor,
    reconstructed: np.ndarray | torch.Tensor,
    metric: str = "rmse"
):
    """
    Plots how reconstruction performance changes across frequencies and channels.
    Computes RMSE, R², and Pearson correlation for every freq × channel pair.

    Args:
        original: 4D array (freq, chan_row, chan_col, time)
        reconstructed: same shape as original
        metric: which metric to display ('rmse', 'r2', 'corr')
    """
    
    # --- Handle tensors ---
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()

    assert original.shape == reconstructed.shape, "Original and reconstructed must match in shape."
    
    # Squeeze batch dimension if present
    if len(original.shape) > 4:
        original = original.squeeze()
        reconstructed = reconstructed.squeeze()
        
    if len(original.shape) > 4 or len(original.shape) < 4:
        original = original[0, ...] if len(original.shape) > 4 else original
        reconstructed = reconstructed[0, ...] if len(reconstructed.shape) > 4 else reconstructed
    
    assert original.ndim == 4, "Expected shape (freq, chan_row, chan_col, time)"
    
    freq_dim, ch1_dim, ch2_dim, _ = original.shape
    n_channels = ch1_dim * ch2_dim
    
    # Flatten channels for iteration
    orig_flat = original.reshape(freq_dim, n_channels, -1)
    recon_flat = reconstructed.reshape(freq_dim, n_channels, -1)
    
    # Initialize metric matrices
    rmse_map = np.zeros((freq_dim, n_channels))
    r2_map = np.zeros((freq_dim, n_channels))
    corr_map = np.zeros((freq_dim, n_channels))
    
    # Compute metrics
    for f in range(freq_dim):
        for c in range(n_channels):
            y_true = orig_flat[f, c]
            y_pred = recon_flat[f, c]
            
            rmse_map[f, c] = np.sqrt(mean_squared_error(y_true, y_pred))
            r2_map[f, c] = r2_score(y_true, y_pred)
            corr_map[f, c], _ = pearsonr(y_true, y_pred)
    
    # Choose which metric to plot
    metric = metric.lower()
    if metric == "rmse":
        data = rmse_map
        title = "RMSE (lower is better)"
    elif metric == "r2":
        data = r2_map
        title = "R² (higher is better)"
    elif metric == "corr":
        data = corr_map
        title = "Pearson r (higher is better)"
    else:
        raise ValueError("metric must be one of: 'rmse', 'r2', 'corr'")
    
    # Plot
    plt.figure(figsize=(12, 6))
    im = plt.imshow(data, aspect="auto", cmap="viridis", origin="lower")
    plt.colorbar(im, label=title.split(" ")[0])
    plt.title(title)
    plt.xlabel("Channel index (flattened)")
    plt.ylabel("Frequency index")
    plt.tight_layout()
    plt.show()
    
    return {"rmse": rmse_map, "r2": r2_map, "corr": corr_map}
