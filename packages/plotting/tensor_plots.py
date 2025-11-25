import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
import warnings


def plot_frequency_temporal_correlation(
    tensor: np.ndarray,
    channels: list = None,
    freq_labels: list = None,
    figsize: tuple = (15, 10),
    cmap: str = 'RdBu_r',
    method: str = 'pearson',
    skip_constant: bool = True,
    constant_threshold: float = 1e-10
) -> None:
    """
    Plot how much each frequency band is correlated in time for each channel.
    
    For a tensor of shape (freq, row, col, time), this creates a correlation
    matrix showing how frequency bands co-vary over time for each spatial channel.
    
    Args:
        tensor: Input tensor of shape (freq, row, col, time)
        channels: List of (row, col) tuples to plot. If None, plots all channels
        freq_labels: Labels for frequency bands. If None, uses indices
        figsize: Figure size
        cmap: Colormap for correlation matrix
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        skip_constant: If True, skip channels where all frequencies are constant
        constant_threshold: Threshold to consider a channel constant (std < threshold)
    
    Returns:
        None (displays plot)
    """
    if not isinstance(tensor, np.ndarray):
        try:
            tensor = tensor.cpu().numpy() if hasattr(tensor, 'cpu') else tensor.numpy()
        except (AttributeError, TypeError):
            tensor = np.array(tensor)
    
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor (freq, row, col, time), got shape {tensor.shape}")
    
    n_freq, n_row, n_col, n_time = tensor.shape
    
    # Set frequency labels
    if freq_labels is None:
        freq_labels = [f"Freq {i}" for i in range(n_freq)]
    if len(freq_labels) != n_freq:
        raise ValueError(f"freq_labels length {len(freq_labels)} doesn't match n_freq {n_freq}")
    
    # Helper function to check if a channel is constant
    def is_constant_channel(row, col):
        """Check if all frequencies at this channel are constant"""
        channel_data = tensor[:, row, col, :]
        # Check if all frequencies have near-zero variance
        stds = np.std(channel_data, axis=1)
        return np.all(stds < constant_threshold)
    
    # Determine which channels to plot
    if channels is None:
        # Get all non-constant channels
        valid_channels = [(r, c) for r in range(n_row) for c in range(n_col)
                         if not (skip_constant and is_constant_channel(r, c))]
        
        if len(valid_channels) == 0:
            print("Warning: All channels are constant!")
            return
        
        # Sample some channels if there are too many
        max_channels = 9  # 3x3 grid
        if len(valid_channels) > max_channels:
            # Sample uniformly
            indices = np.linspace(0, len(valid_channels) - 1, max_channels, dtype=int)
            channels = [valid_channels[i] for i in indices]
        else:
            channels = valid_channels
    else:
        # Filter out constant channels from provided list
        if skip_constant:
            original_count = len(channels)
            channels = [(r, c) for r, c in channels if not is_constant_channel(r, c)]
            if len(channels) < original_count:
                print(f"Filtered out {original_count - len(channels)} constant channels")
    
    if len(channels) == 0:
        print("Warning: All specified channels are constant!")
        return
    
    n_channels = len(channels)
    print(f"Plotting {n_channels} non-constant channels")
    
    # Calculate grid size
    nrows = int(np.ceil(np.sqrt(n_channels)))
    ncols = int(np.ceil(n_channels / nrows))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1) if n_channels > 1 else [axes]
    
    for idx, (row, col) in enumerate(channels):
        if idx >= len(axes):
            break
            
        # Extract time series for all frequencies at this channel
        channel_data = tensor[:, row, col, :]  # (n_freq, n_time)
        
        # Compute correlation matrix between frequency bands
        corr_matrix = np.zeros((n_freq, n_freq))
        
        for i in range(n_freq):
            for j in range(n_freq):
                # Check if either frequency is constant
                if np.std(channel_data[i]) < constant_threshold or np.std(channel_data[j]) < constant_threshold:
                    corr_matrix[i, j] = np.nan
                    continue
                
                try:
                    if method == 'pearson':
                        corr, _ = pearsonr(channel_data[i], channel_data[j])
                    elif method == 'spearman':
                        from scipy.stats import spearmanr
                        corr, _ = spearmanr(channel_data[i], channel_data[j])
                    elif method == 'kendall':
                        from scipy.stats import kendalltau
                        corr, _ = kendalltau(channel_data[i], channel_data[j])
                    else:
                        raise ValueError(f"Unknown correlation method: {method}")
                    
                    corr_matrix[i, j] = corr
                except Exception as e:
                    # Handle any other correlation computation errors
                    warnings.warn(f"Could not compute correlation for freq {i} and {j}: {e}")
                    corr_matrix[i, j] = np.nan
        
        # Plot correlation matrix (NaN values will appear as white/masked)
        im = axes[idx].imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        axes[idx].set_xticks(range(n_freq))
        axes[idx].set_yticks(range(n_freq))
        axes[idx].set_xticklabels(freq_labels, rotation=45, ha='right')
        axes[idx].set_yticklabels(freq_labels)
        axes[idx].set_title(f"Channel ({row}, {col})")
        
        # Add colorbar
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        # Add correlation values as text
        for i in range(n_freq):
            for j in range(n_freq):
                if np.isnan(corr_matrix[i, j]):
                    axes[idx].text(j, i, 'N/A',
                                 ha='center', va='center', color='gray', fontsize=8)
                else:
                    text_color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
                    axes[idx].text(j, i, f'{corr_matrix[i, j]:.2f}',
                                 ha='center', va='center', color=text_color, fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_channels, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Frequency Band Temporal Correlations per Channel ({method.capitalize()})',
                 fontsize=16, y=1.00)
    plt.tight_layout()
    plt.show()


def plot_frequency_temporal_correlation_summary(
    tensor: np.ndarray,
    freq_labels: list = None,
    figsize: tuple = (12, 8),
    cmap: str = 'viridis',
    skip_constant: bool = True,
    constant_threshold: float = 1e-10
) -> None:
    """
    Plot average frequency correlation across all channels with spatial heatmap.
    
    Creates two plots:
    1. Average correlation matrix across all spatial channels
    2. Spatial heatmap showing average correlation strength per channel
    
    Args:
        tensor: Input tensor of shape (freq, row, col, time)
        freq_labels: Labels for frequency bands
        figsize: Figure size
        cmap: Colormap
        skip_constant: If True, skip constant channels in averaging
        constant_threshold: Threshold to consider a channel constant
    """
    if not isinstance(tensor, np.ndarray):
        try:
            tensor = tensor.cpu().numpy() if hasattr(tensor, 'cpu') else tensor.numpy()
        except (AttributeError, TypeError):
            tensor = np.array(tensor)
    
    n_freq, n_row, n_col, n_time = tensor.shape
    
    if freq_labels is None:
        freq_labels = [f"Freq {i}" for i in range(n_freq)]
    
    # Compute correlation for each channel
    all_corr_matrices = np.full((n_row, n_col, n_freq, n_freq), np.nan)
    constant_mask = np.zeros((n_row, n_col), dtype=bool)
    
    for row in range(n_row):
        for col in range(n_col):
            channel_data = tensor[:, row, col, :]
            
            # Check if channel is constant
            if skip_constant and np.all(np.std(channel_data, axis=1) < constant_threshold):
                constant_mask[row, col] = True
                continue
            
            for i in range(n_freq):
                for j in range(n_freq):
                    # Check if either frequency is constant
                    if np.std(channel_data[i]) < constant_threshold or np.std(channel_data[j]) < constant_threshold:
                        continue
                    
                    try:
                        corr, _ = pearsonr(channel_data[i], channel_data[j])
                        all_corr_matrices[row, col, i, j] = corr
                    except Exception:
                        continue
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Average correlation matrix (ignoring NaN values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg_corr = np.nanmean(all_corr_matrices, axis=(0, 1))
    
    im1 = ax1.imshow(avg_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_xticks(range(n_freq))
    ax1.set_yticks(range(n_freq))
    ax1.set_xticklabels(freq_labels, rotation=45, ha='right')
    ax1.set_yticklabels(freq_labels)
    
    n_valid = np.sum(~constant_mask)
    ax1.set_title(f'Average Frequency Correlation\n(across {n_valid}/{n_row*n_col} non-constant channels)')
    plt.colorbar(im1, ax=ax1)
    
    # Add text annotations
    # for i in range(n_freq):
    #     for j in range(n_freq):
    #         if np.isnan(avg_corr[i, j]):
    #             ax1.text(j, i, 'N/A', ha='center', va='center', color='gray', fontsize=10)
    #         else:
    #             text_color = 'white' if abs(avg_corr[i, j]) > 0.5 else 'black'
    #             ax1.text(j, i, f'{avg_corr[i, j]:.2f}',
    #                     ha='center', va='center', color=text_color, fontsize=10)
    
    # Plot 2: Spatial map of average correlation strength
    channel_corr_strength = np.full((n_row, n_col), np.nan)
    for row in range(n_row):
        for col in range(n_col):
            if constant_mask[row, col]:
                continue
            
            # Get off-diagonal correlations
            corr_mat = all_corr_matrices[row, col]
            mask = ~np.eye(n_freq, dtype=bool)
            valid_corrs = corr_mat[mask]
            valid_corrs = valid_corrs[~np.isnan(valid_corrs)]
            
            if len(valid_corrs) > 0:
                channel_corr_strength[row, col] = np.abs(valid_corrs).mean()
    
    im2 = ax2.imshow(channel_corr_strength, cmap=cmap, aspect='auto')
    ax2.set_title('Average Correlation Strength per Channel\n(mean absolute off-diagonal correlation)')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    plt.colorbar(im2, ax=ax2, label='Avg |correlation|')
    
    # Add text annotations
    for row in range(n_row):
        for col in range(n_col):
            if constant_mask[row, col]:
                ax2.text(col, row, 'X', ha='center', va='center', 
                        color='red', fontsize=12, fontweight='bold')
            elif not np.isnan(channel_corr_strength[row, col]):
                ax2.text(col, row, f'{channel_corr_strength[row, col]:.2f}',
                        ha='center', va='center', color='white', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Constant channels: {np.sum(constant_mask)} / {n_row * n_col}")

def analyze_frequency_selection_information_loss(
    tensor: np.ndarray,
    selected_freqs: list,
    freq_labels: list = None,
    method: str = 'variance',
    figsize: tuple = (15, 10)
) -> dict:
    """
    Analyze how much information is lost by selecting a subset of frequencies.
    
    Args:
        tensor: Full tensor of shape (freq, row, col, time)
        selected_freqs: List of frequency indices to keep
        freq_labels: Labels for all frequency bands
        method: Information metric - 'variance', 'energy', 'entropy', or 'pca'
        figsize: Figure size for plots
    
    Returns:
        dict with metrics and analysis results
    """
    if not isinstance(tensor, np.ndarray):
        try:
            tensor = tensor.cpu().numpy() if hasattr(tensor, 'cpu') else tensor.numpy()
        except (AttributeError, TypeError):
            tensor = np.array(tensor)
    
    n_freq, n_row, n_col, n_time = tensor.shape
    
    if freq_labels is None:
        freq_labels = [f"Freq {i}" for i in range(n_freq)]
    
    # Get selected tensor
    selected_tensor = tensor[selected_freqs, :, :, :]
    
    # Initialize results
    results = {
        'method': method,
        'n_total_freqs': n_freq,
        'n_selected_freqs': len(selected_freqs),
        'selection_ratio': len(selected_freqs) / n_freq,
        'selected_indices': selected_freqs,
        'selected_labels': [freq_labels[i] for i in selected_freqs]
    }
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # === 1. Variance-based analysis ===
    if method in ['variance', 'all']:
        # Calculate variance per frequency across all spatial locations and time
        freq_variance = np.var(tensor.reshape(n_freq, -1), axis=1)
        selected_variance = freq_variance[selected_freqs]
        
        total_variance = np.sum(freq_variance)
        selected_variance_sum = np.sum(selected_variance)
        variance_retained = (selected_variance_sum / total_variance) * 100
        
        results['variance_retained_pct'] = variance_retained
        results['variance_lost_pct'] = 100 - variance_retained
        results['freq_variance'] = freq_variance
        
        # Plot variance per frequency
        ax1 = fig.add_subplot(gs[0, :])
        colors = ['red' if i in selected_freqs else 'blue' for i in range(n_freq)]
        bars = ax1.bar(range(n_freq), freq_variance, color=colors, alpha=0.7)
        ax1.set_xlabel('Frequency Band')
        ax1.set_ylabel('Variance')
        ax1.set_title(f'Variance per Frequency Band (Retained: {variance_retained:.2f}%)')
        ax1.set_xticks(range(n_freq))
        ax1.set_xticklabels(freq_labels, rotation=45, ha='right')
        ax1.legend([plt.Rectangle((0,0),1,1, color='blue', alpha=0.7),
                   plt.Rectangle((0,0),1,1, color='red', alpha=0.7)],
                  ['Not Selected', 'Selected'])
        ax1.grid(axis='y', alpha=0.3)
    
    # === 2. Energy-based analysis ===
    if method in ['energy', 'all']:
        # Calculate energy (sum of squared values)
        freq_energy = np.sum(tensor.reshape(n_freq, -1) ** 2, axis=1)
        selected_energy = freq_energy[selected_freqs]
        
        total_energy = np.sum(freq_energy)
        selected_energy_sum = np.sum(selected_energy)
        energy_retained = (selected_energy_sum / total_energy) * 100
        
        results['energy_retained_pct'] = energy_retained
        results['energy_lost_pct'] = 100 - energy_retained
        results['freq_energy'] = freq_energy
        
        # Plot energy per frequency
        ax2 = fig.add_subplot(gs[1, 0])
        colors = ['red' if i in selected_freqs else 'blue' for i in range(n_freq)]
        ax2.bar(range(n_freq), freq_energy, color=colors, alpha=0.7)
        ax2.set_xlabel('Frequency Band')
        ax2.set_ylabel('Energy')
        ax2.set_title(f'Energy per Frequency\n(Retained: {energy_retained:.2f}%)')
        ax2.set_xticks(range(n_freq))
        ax2.set_xticklabels(freq_labels, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
    
    # === 3. Spatial information distribution ===
    # Calculate how much information each spatial location has in selected vs all freqs
    spatial_variance_all = np.var(tensor, axis=(0, 3))  # (row, col)
    spatial_variance_selected = np.var(selected_tensor, axis=(0, 3))
    spatial_retention = (spatial_variance_selected / (spatial_variance_all + 1e-10)) * 100
    
    results['spatial_variance_retention'] = spatial_retention
    results['mean_spatial_retention_pct'] = np.mean(spatial_retention[~np.isnan(spatial_retention)])
    
    # Plot spatial retention heatmap
    ax3 = fig.add_subplot(gs[1, 1])
    im = ax3.imshow(spatial_retention, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    ax3.set_title('Spatial Information Retention (%)')
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')
    plt.colorbar(im, ax=ax3, label='Retention %')
    
    # Add text annotations
    for row in range(n_row):
        for col in range(n_col):
            if not np.isnan(spatial_retention[row, col]):
                text_color = 'black' if spatial_retention[row, col] > 50 else 'white'
                ax3.text(col, row, f'{spatial_retention[row, col]:.0f}',
                        ha='center', va='center', color=text_color, fontsize=8)
    
    # === 4. Correlation between selected and non-selected frequencies ===
    non_selected_freqs = [i for i in range(n_freq) if i not in selected_freqs]
    
    if len(non_selected_freqs) > 0:
        # Flatten spatial and temporal dimensions
        selected_flat = selected_tensor.reshape(len(selected_freqs), -1)
        non_selected_flat = tensor[non_selected_freqs, :, :, :].reshape(len(non_selected_freqs), -1)
        
        # Compute cross-correlation matrix
        cross_corr = np.zeros((len(selected_freqs), len(non_selected_freqs)))
        for i, sel_idx in enumerate(selected_freqs):
            for j, non_sel_idx in enumerate(non_selected_freqs):
                if np.std(selected_flat[i]) > 1e-10 and np.std(non_selected_flat[j]) > 1e-10:
                    cross_corr[i, j], _ = pearsonr(selected_flat[i], non_selected_flat[j])
                else:
                    cross_corr[i, j] = np.nan
        
        results['cross_correlation'] = cross_corr
        results['mean_cross_correlation'] = np.nanmean(np.abs(cross_corr))
        
        # Plot cross-correlation heatmap
        ax4 = fig.add_subplot(gs[1, 2])
        im = ax4.imshow(cross_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax4.set_title('Cross-Correlation:\nSelected vs Non-Selected Freqs')
        ax4.set_xlabel('Non-Selected Frequencies')
        ax4.set_ylabel('Selected Frequencies')
        ax4.set_yticks(range(len(selected_freqs)))
        ax4.set_yticklabels([freq_labels[i] for i in selected_freqs])
        ax4.set_xticks(range(len(non_selected_freqs)))
        ax4.set_xticklabels([freq_labels[i] for i in non_selected_freqs], rotation=45, ha='right')
        plt.colorbar(im, ax=ax4, label='Correlation')
    
    # === 5. Cumulative information retention ===
    if method in ['variance', 'all']:
        # Sort frequencies by variance
        sorted_indices = np.argsort(freq_variance)[::-1]
        cumulative_variance = np.cumsum(freq_variance[sorted_indices])
        cumulative_pct = (cumulative_variance / total_variance) * 100
        
        # Find how many top frequencies are needed for 90%, 95%, 99% retention
        n_for_90 = np.argmax(cumulative_pct >= 90) + 1
        n_for_95 = np.argmax(cumulative_pct >= 95) + 1
        n_for_99 = np.argmax(cumulative_pct >= 99) + 1
        
        results['n_freqs_for_90pct'] = n_for_90
        results['n_freqs_for_95pct'] = n_for_95
        results['n_freqs_for_99pct'] = n_for_99
        
        ax5 = fig.add_subplot(gs[2, :2])
        ax5.plot(range(1, n_freq + 1), cumulative_pct, 'b-', linewidth=2, label='All Frequencies')
        ax5.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90%')
        ax5.axhline(y=95, color='orange', linestyle='--', alpha=0.5, label='95%')
        ax5.axhline(y=99, color='purple', linestyle='--', alpha=0.5, label='99%')
        ax5.axvline(x=len(selected_freqs), color='green', linestyle='-', linewidth=2, 
                   label=f'Your Selection (n={len(selected_freqs)})')
        ax5.scatter([len(selected_freqs)], [variance_retained], color='green', s=100, zorder=5)
        ax5.set_xlabel('Number of Frequencies (sorted by variance)')
        ax5.set_ylabel('Cumulative Variance Retained (%)')
        ax5.set_title('Cumulative Information Retention')
        ax5.grid(alpha=0.3)
        ax5.legend()
        ax5.set_xlim(0, n_freq + 1)
        ax5.set_ylim(0, 105)
    
    # === 6. Summary statistics ===
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    summary_text = f"""
    INFORMATION LOSS SUMMARY
    {'=' * 30}
    
    Frequencies: {len(selected_freqs)}/{n_freq} ({results['selection_ratio']*100:.1f}%)
    Selected: {', '.join(results['selected_labels'])}
    
    """
    
    if 'variance_retained_pct' in results:
        summary_text += f"""
    Variance Retained: {results['variance_retained_pct']:.2f}%
    Variance Lost: {results['variance_lost_pct']:.2f}%
    """
    
    if 'energy_retained_pct' in results:
        summary_text += f"""
    Energy Retained: {results['energy_retained_pct']:.2f}%
    Energy Lost: {results['energy_lost_pct']:.2f}%
    """
    
    if 'mean_spatial_retention_pct' in results:
        summary_text += f"""
    Avg Spatial Retention: {results['mean_spatial_retention_pct']:.2f}%
    """
    
    if 'mean_cross_correlation' in results:
        summary_text += f"""
    Avg Cross-Correlation: {results['mean_cross_correlation']:.3f}
    """
    
    if 'n_freqs_for_90pct' in results:
        summary_text += f"""
    Freqs needed for:
      90% info: {results['n_freqs_for_90pct']}
      95% info: {results['n_freqs_for_95pct']}
      99% info: {results['n_freqs_for_99pct']}
    """
    
    ax6.text(0.1, 0.9, summary_text, fontsize=10, family='monospace',
            verticalalignment='top', transform=ax6.transAxes)
    
    plt.suptitle('Frequency Selection Information Loss Analysis', fontsize=16, y=0.98)
    plt.show()
    
    return results


def compare_frequency_selections(
    tensor: np.ndarray,
    selection_configs: dict,
    freq_labels: list = None,
    figsize: tuple = (12, 6)
) -> None:
    """
    Compare multiple frequency selection strategies.
    
    Args:
        tensor: Full tensor of shape (freq, row, col, time)
        selection_configs: Dict of {name: [freq_indices]} for different selections
        freq_labels: Labels for frequency bands
        figsize: Figure size
    """
    if not isinstance(tensor, np.ndarray):
        try:
            tensor = tensor.cpu().numpy() if hasattr(tensor, 'cpu') else tensor.numpy()
        except (AttributeError, TypeError):
            tensor = np.array(tensor)
    
    n_freq = tensor.shape[0]
    
    if freq_labels is None:
        freq_labels = [f"Freq {i}" for i in range(n_freq)]
    
    # Calculate variance per frequency
    freq_variance = np.var(tensor.reshape(n_freq, -1), axis=1)
    total_variance = np.sum(freq_variance)
    
    # Calculate energy per frequency
    freq_energy = np.sum(tensor.reshape(n_freq, -1) ** 2, axis=1)
    total_energy = np.sum(freq_energy)
    
    # Collect results
    comparison_data = []
    
    for name, selected_freqs in selection_configs.items():
        variance_retained = (np.sum(freq_variance[selected_freqs]) / total_variance) * 100
        energy_retained = (np.sum(freq_energy[selected_freqs]) / total_energy) * 100
        
        comparison_data.append({
            'name': name,
            'n_freqs': len(selected_freqs),
            'variance_retained': variance_retained,
            'energy_retained': energy_retained,
            'selected_freqs': selected_freqs
        })
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    names = [d['name'] for d in comparison_data]
    variance_vals = [d['variance_retained'] for d in comparison_data]
    energy_vals = [d['energy_retained'] for d in comparison_data]
    n_freqs_vals = [d['n_freqs'] for d in comparison_data]
    
    # Plot 1: Information retention comparison
    x = np.arange(len(names))
    width = 0.35
    
    ax1.bar(x - width/2, variance_vals, width, label='Variance Retained', alpha=0.8)
    ax1.bar(x + width/2, energy_vals, width, label='Energy Retained', alpha=0.8)
    ax1.set_ylabel('Percentage Retained (%)')
    ax1.set_title('Information Retention Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # Add value labels
    for i, (v, e) in enumerate(zip(variance_vals, energy_vals)):
        ax1.text(i - width/2, v + 2, f'{v:.1f}%', ha='center', fontsize=8)
        ax1.text(i + width/2, e + 2, f'{e:.1f}%', ha='center', fontsize=8)
    
    # Plot 2: Frequency selection visualization
    for i, data in enumerate(comparison_data):
        selection_vector = np.zeros(n_freq)
        selection_vector[data['selected_freqs']] = 1
        ax2.plot(range(n_freq), selection_vector + i * 1.2, label=data['name'], linewidth=2)
        ax2.fill_between(range(n_freq), i * 1.2, selection_vector + i * 1.2, alpha=0.3)
    
    ax2.set_xlabel('Frequency Band Index')
    ax2.set_ylabel('Selection Strategy')
    ax2.set_title('Frequency Selection Patterns')
    ax2.set_xticks(range(n_freq))
    ax2.set_xticklabels(freq_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary table
    print("\n" + "="*70)
    print(f"{'Strategy':<20} {'N Freqs':<10} {'Variance %':<15} {'Energy %':<15}")
    print("="*70)
    for data in comparison_data:
        print(f"{data['name']:<20} {data['n_freqs']:<10} {data['variance_retained']:>12.2f}%  {data['energy_retained']:>12.2f}%")
    print("="*70 + "\n")

def plot_temporal_instantaneous_correlation(
    tensor: np.ndarray,
    freq_samples: list = None,
    n_samples: int = 5,
    time_labels: list = None,
    figsize: tuple = (15, 10),
    cmap: str = 'RdBu_r',
    method: str = 'pearson',
    skip_constant: bool = True,
    constant_threshold: float = 1e-10
    ) -> None:
    """
    Plot how much each time instant is correlated across frequencies (averaged across channels).
    
    This is the time equivalent of plot_frequency_temporal_correlation_summary.
    For a tensor of shape (freq, row, col, time), this creates a correlation
    matrix showing how time instants co-vary across frequency bands, averaged
    across all spatial channels.
    
    Args:
        tensor: Input tensor of shape (freq, row, col, time)
        freq_samples: Specific frequency indices to sample. If None, samples uniformly
        n_samples: Number of frequency samples if freq_samples is None
        time_labels: Labels for time points. If None, uses indices
        figsize: Figure size
        cmap: Colormap for correlation matrix
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        skip_constant: If True, skip constant channels in averaging
        constant_threshold: Threshold to consider a channel constant
    
    Returns:
        None (displays plot)
    """
    if not isinstance(tensor, np.ndarray):
        try:
            tensor = tensor.cpu().numpy() if hasattr(tensor, 'cpu') else tensor.numpy()
        except (AttributeError, TypeError):
            tensor = np.array(tensor)
    
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor (freq, row, col, time), got shape {tensor.shape}")
    
    n_freq, n_row, n_col, n_time = tensor.shape
    
    # Set time labels
    if time_labels is None:
        time_labels = [f"T{i}" for i in range(n_time)]
    if len(time_labels) != n_time:
        raise ValueError(f"time_labels length {len(time_labels)} doesn't match n_time {n_time}")
    
    # Determine frequency samples to analyze
    if freq_samples is None:
        freq_samples = np.linspace(0, n_freq - 1, n_samples, dtype=int)
    
    n_samples_actual = len(freq_samples)
    
    print(f"Analyzing {n_samples_actual} frequency bands across {n_time} time instants")
    
    # For each frequency, compute temporal correlation across all channels
    all_corr_matrices = []
    freq_labels_selected = []
    constant_mask = np.zeros((n_row, n_col), dtype=bool)
    
    for freq_idx in freq_samples:
        # Get data for this frequency: (n_row, n_col, n_time)
        freq_data = tensor[freq_idx, :, :, :]
        
        # Identify constant channels
        if skip_constant:
            for row in range(n_row):
                for col in range(n_col):
                    if np.std(freq_data[row, col, :]) < constant_threshold:
                        constant_mask[row, col] = True
        
        # Collect valid channels
        valid_channel_data = []
        for row in range(n_row):
            for col in range(n_col):
                if not constant_mask[row, col]:
                    valid_channel_data.append(freq_data[row, col, :])  # (n_time,)
        
        if len(valid_channel_data) == 0:
            print(f"Warning: All channels constant at frequency {freq_idx}, skipping")
            continue
        
        # Stack into (n_valid_channels, n_time)
        valid_channel_data = np.array(valid_channel_data)
        
        # Compute correlation matrix for time instants across channels
        # For each pair of time instants, correlate their values across channels
        corr_matrix = np.zeros((n_time, n_time))
        
        for i in range(n_time):
            for j in range(n_time):
                # Get values at time i and j across all valid channels
                time_i_values = valid_channel_data[:, i]  # (n_valid_channels,)
                time_j_values = valid_channel_data[:, j]  # (n_valid_channels,)
                
                # Check if either time instant has constant values across channels
                if np.std(time_i_values) < constant_threshold or np.std(time_j_values) < constant_threshold:
                    corr_matrix[i, j] = np.nan
                    continue
                
                try:
                    if method == 'pearson':
                        corr, _ = pearsonr(time_i_values, time_j_values)
                    elif method == 'spearman':
                        from scipy.stats import spearmanr
                        corr, _ = spearmanr(time_i_values, time_j_values)
                    elif method == 'kendall':
                        from scipy.stats import kendalltau
                        corr, _ = kendalltau(time_i_values, time_j_values)
                    else:
                        raise ValueError(f"Unknown correlation method: {method}")
                    
                    corr_matrix[i, j] = corr
                except Exception as e:
                    warnings.warn(f"Could not compute correlation for time {i} and {j}: {e}")
                    corr_matrix[i, j] = np.nan
        
        all_corr_matrices.append(corr_matrix)
        freq_labels_selected.append(f"Freq {freq_idx}")
    
    if len(all_corr_matrices) == 0:
        print("Warning: No valid frequency bands to plot!")
        return
    
    # Calculate grid size
    nrows = int(np.ceil(np.sqrt(n_samples_actual)))
    ncols = int(np.ceil(n_samples_actual / nrows))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1) if n_samples_actual > 1 else [axes]
    
    # Plot correlation matrix for each frequency
    for idx, (corr_matrix, freq_label) in enumerate(zip(all_corr_matrices, freq_labels_selected)):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # Plot correlation matrix
        im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        
        # Set ticks - show subset to avoid overcrowding
        tick_interval = max(1, n_time // 10)
        tick_positions = range(0, n_time, tick_interval)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels([time_labels[i] for i in tick_positions], rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels([time_labels[i] for i in tick_positions], fontsize=8)
        ax.set_title(freq_label)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(len(all_corr_matrices), len(axes)):
        axes[idx].axis('off')
    
    n_valid_channels = np.sum(~constant_mask)
    plt.suptitle(f'Time Instant Correlations per Frequency Band ({method.capitalize()})\n' + 
                 f'Averaged across {n_valid_channels}/{n_row*n_col} non-constant channels',
                 fontsize=16, y=0.995)
    plt.tight_layout()
    plt.show()


def plot_temporal_instantaneous_correlation_summary(
    tensor: np.ndarray,
    time_labels: list = None,
    figsize: tuple = (12, 8),
    cmap: str = 'viridis',
    method: str = 'pearson',
    skip_constant: bool = True,
    constant_threshold: float = 1e-10
) -> None:
    """
    Plot average temporal correlation across all frequencies and channels with summary heatmap.
    
    This is the time equivalent of plot_frequency_temporal_correlation_summary.
    
    Creates two plots:
    1. Average correlation matrix of time instants across all frequencies and channels
    2. Per-frequency heatmap showing average temporal correlation strength
    
    Args:
        tensor: Input tensor of shape (freq, row, col, time)
        time_labels: Labels for time points
        figsize: Figure size
        cmap: Colormap
        method: Correlation method
        skip_constant: If True, skip constant channels
        constant_threshold: Threshold to consider constant
    """
    if not isinstance(tensor, np.ndarray):
        try:
            tensor = tensor.cpu().numpy() if hasattr(tensor, 'cpu') else tensor.numpy()
        except (AttributeError, TypeError):
            tensor = np.array(tensor)
    
    n_freq, n_row, n_col, n_time = tensor.shape
    
    if time_labels is None:
        time_labels = [f"T{i}" for i in range(n_time)]
    
    # Identify constant channels
    constant_mask = np.zeros((n_row, n_col), dtype=bool)
    if skip_constant:
        for row in range(n_row):
            for col in range(n_col):
                channel_data = tensor[:, row, col, :]
                if np.all(np.std(channel_data, axis=1) < constant_threshold):
                    constant_mask[row, col] = True
    
    # Compute correlation for each frequency
    all_corr_matrices = np.full((n_freq, n_time, n_time), np.nan)
    
    for freq_idx in range(n_freq):
        freq_data = tensor[freq_idx, :, :, :]  # (n_row, n_col, n_time)
        
        # Collect valid channels
        valid_channel_data = []
        for row in range(n_row):
            for col in range(n_col):
                if not constant_mask[row, col]:
                    if np.std(freq_data[row, col, :]) > constant_threshold:
                        valid_channel_data.append(freq_data[row, col, :])
        
        if len(valid_channel_data) == 0:
            continue
        
        valid_channel_data = np.array(valid_channel_data)  # (n_valid_channels, n_time)
        
        # Compute temporal correlation
        for i in range(n_time):
            for j in range(n_time):
                time_i_values = valid_channel_data[:, i]
                time_j_values = valid_channel_data[:, j]
                
                if np.std(time_i_values) < constant_threshold or np.std(time_j_values) < constant_threshold:
                    continue
                
                try:
                    if method == 'pearson':
                        corr, _ = pearsonr(time_i_values, time_j_values)
                    elif method == 'spearman':
                        from scipy.stats import spearmanr
                        corr, _ = spearmanr(time_i_values, time_j_values)
                    elif method == 'kendall':
                        from scipy.stats import kendalltau
                        corr, _ = kendalltau(time_i_values, time_j_values)
                    else:
                        raise ValueError(f"Unknown correlation method: {method}")
                    
                    all_corr_matrices[freq_idx, i, j] = corr
                except Exception:
                    continue
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Average temporal correlation matrix (across all frequencies)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg_corr = np.nanmean(all_corr_matrices, axis=0)
    
    im1 = ax1.imshow(avg_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Set ticks - show subset
    tick_interval = max(1, n_time // 10)
    tick_positions = range(0, n_time, tick_interval)
    ax1.set_xticks(tick_positions)
    ax1.set_yticks(tick_positions)
    ax1.set_xticklabels([time_labels[i] for i in tick_positions], rotation=45, ha='right')
    ax1.set_yticklabels([time_labels[i] for i in tick_positions])
    
    n_valid = np.sum(~constant_mask)
    n_valid_freqs = np.sum(~np.all(np.isnan(all_corr_matrices), axis=(1, 2)))
    ax1.set_title(f'Average Time Instant Correlation\n(across {n_valid_freqs}/{n_freq} frequencies, ' + 
                  f'{n_valid}/{n_row*n_col} channels)')
    plt.colorbar(im1, ax=ax1)
    
    # Plot 2: Per-frequency correlation strength
    freq_corr_strength = np.full(n_freq, np.nan)
    for freq_idx in range(n_freq):
        corr_mat = all_corr_matrices[freq_idx]
        # Get off-diagonal correlations
        mask = ~np.eye(n_time, dtype=bool)
        valid_corrs = corr_mat[mask]
        valid_corrs = valid_corrs[~np.isnan(valid_corrs)]
        
        if len(valid_corrs) > 0:
            freq_corr_strength[freq_idx] = np.abs(valid_corrs).mean()
    
    ax2.bar(range(n_freq), freq_corr_strength, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Frequency Band')
    ax2.set_ylabel('Avg Temporal Correlation Strength')
    ax2.set_title('Average Temporal Correlation per Frequency\n(mean absolute off-diagonal correlation)')
    ax2.set_xticks(range(n_freq))
    ax2.set_xticklabels([f"F{i}" for i in range(n_freq)], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, val in enumerate(freq_corr_strength):
        if not np.isnan(val):
            ax2.text(i, val + 0.02, f'{val:.2f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Constant channels: {np.sum(constant_mask)} / {n_row * n_col}")
    print(f"Valid frequencies: {n_valid_freqs} / {n_freq}")

if __name__ == "__main__":
    import torch
    import sys
    # Load your data
    tensor = np.load("scripts/test_output/super_freq_res/patient1_trial289.npz")['tensor'] # shape (freq, row, col, time)
    # Print tensor stats
    
    print(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}, min: {tensor.min()}, max: {tensor.max()}, mean: {tensor.mean():.4f}, std: {tensor.std():.4f}")
    sys.exit(0)
    # Define your frequency selection
    tensor = torch.tensor(tensor).permute(2, 0, 1, 3)  # to (freq, row, col, time)
    selected_freqs = list(range(1, 15))

    # Plot summary of frequency correlations across all channels
    plot_frequency_temporal_correlation_summary(
        tensor,
        skip_constant=True
    )
    # Analyze information loss
    results = analyze_frequency_selection_information_loss(
        tensor,
        selected_freqs,
        method='all'  # Can be 'variance', 'energy', or 'all'
    )
    
    sys.exit(0)
    # Compare different selection strategies
    selection_configs = {
        '1': range(1, 25),
        '2': range(2, 25),
        '3': range(1, 20),
        '4': range(1, 15),
        '5': range(2, 15),
        '6': range(3, 15),
    }

    compare_frequency_selections(tensor, selection_configs)

