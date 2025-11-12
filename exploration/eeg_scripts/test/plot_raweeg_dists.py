import numpy as np

import matplotlib.pyplot as plt

def plot_random_channel_distributions(eeg_data, n_channels=5, bins=50):
    """
    Plot distributions of values from N random EEG channels.
    
    Parameters:
    -----------
    eeg_data : numpy.ndarray
        EEG data with shape (channels, samples)
    n_channels : int
        Number of random channels to plot
    bins : int
        Number of bins for histogram
    """
    n_total_channels = eeg_data.shape[0]
    n_channels = min(n_channels, n_total_channels)
    
    # Select random channels
    random_channels = np.random.choice(n_total_channels, n_channels, replace=False)
    
    # Create subplots
    fig, axes = plt.subplots(n_channels, 1, figsize=(10, 3 * n_channels))
    if n_channels == 1:
        axes = [axes]
    
    for idx, ch in enumerate(random_channels):
        channel_data = eeg_data[ch, :]
        axes[idx].hist(channel_data, bins=bins, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'Channel {ch} Distribution')
        axes[idx].set_xlabel('Amplitude')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data = np.load("scripts/test_output/EEG+Wavelet/patient1_trial1_seg0.npz")
    eeg = data['eeg']
    plot_random_channel_distributions(eeg, n_channels=6, bins=100)  