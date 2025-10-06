from typing import Union

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
