from typing import List, Optional, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import torch

if TYPE_CHECKING:
    from packages.train.training import Trainer

def plot_raweeg_reconstruction(
    trainer: "Trainer",
    n_channels: int = 4,
) -> None:
    """
    Function to plot raweeg reconstruction comparison using train and val data side by side.
    
    Processes both samples in a single forward pass for efficiency.
    
    Args:
        trainer: Trainer object with model, train_loader, val_loader, and device
        n_channels: number of random channels to plot
    """

    # Get a sample from train loader
    train_batch = next(iter(trainer.train_loader))
    train_original = train_batch['target'][0].cpu().numpy()  # (channels, time)
    train_input = train_batch['input'][0].to(trainer.device)  # (channels, time) or appropriate shape
    
    # Get a sample from val loader
    val_batch = next(iter(trainer.val_loader))
    val_original = val_batch['target'][0].cpu().numpy()  # (channels, time)
    val_input = val_batch['input'][0].to(trainer.device)
    
    # Stack into a single batch of 2 samples
    combined_input = torch.stack([train_input, val_input], dim=0)  # (2, channels, time)
    
    # Single forward pass for both samples
    model = trainer.model
    model.eval()
    with torch.no_grad():
        combined_output = model(combined_input)
    
    # Extract reconstructions
    if isinstance(combined_output, dict):
        reconstructed = combined_output['reconstruction'].cpu().numpy()
    elif isinstance(combined_output, tuple):
        reconstructed = combined_output[0].cpu().numpy()
    else:
        reconstructed = combined_output.cpu().numpy()
    
    train_reconstructed = reconstructed[0]  # First sample
    val_reconstructed = reconstructed[1]    # Second sample
    
    # Verify shapes
    assert train_original.shape == train_reconstructed.shape, \
        f"Train shape mismatch: {train_original.shape} vs {train_reconstructed.shape}"
    assert val_original.shape == val_reconstructed.shape, \
        f"Val shape mismatch: {val_original.shape} vs {val_reconstructed.shape}"
    
    # Get channel dimension
    total_channels = train_original.shape[0]
    n_channels = min(n_channels, total_channels)
    random_channels = np.random.choice(total_channels, size=n_channels, replace=False)
    
    # Create subplot grid: n_channels rows x 2 columns (train | val)
    fig, axes = plt.subplots(n_channels, 2, figsize=(15, 3 * n_channels))
    
    # Handle single channel case
    if n_channels == 1:
        axes = axes.reshape(1, -1)
    
    for i, ch in enumerate(random_channels):
        # === TRAIN COLUMN ===
        train_true = train_original[ch, :]
        train_pred = train_reconstructed[ch, :]
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(train_true, train_pred))
        train_r2 = r2_score(train_true, train_pred)
        train_mse = mean_squared_error(train_true, train_pred)
        try:
            train_corr, _ = pearsonr(train_true, train_pred)
        except:
            train_corr = np.nan
        
        axes[i, 0].plot(train_true, label='Original', alpha=0.8, linewidth=1.5, color='blue')
        axes[i, 0].plot(train_pred, label='Reconstructed', alpha=0.7, linewidth=1.5, color='orange')
        axes[i, 0].set_title(
            f'Train - Channel {ch}\n'
            f'RMSE={train_rmse:.3f} | R²={train_r2:.3f} | r={train_corr:.3f} | MSE={train_mse:.3f}',
            fontsize=10
        )
        axes[i, 0].set_xlabel('Time')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].legend(loc='upper right', fontsize=8)
        axes[i, 0].grid(True, alpha=0.3)
        
        # === VAL COLUMN ===
        val_true = val_original[ch, :]
        val_pred = val_reconstructed[ch, :]
        
        # Calculate metrics
        val_rmse = np.sqrt(mean_squared_error(val_true, val_pred))
        val_r2 = r2_score(val_true, val_pred)
        val_mse = mean_squared_error(val_true, val_pred)
        try:
            val_corr, _ = pearsonr(val_true, val_pred)
        except:
            val_corr = np.nan
        
        axes[i, 1].plot(val_true, label='Original', alpha=0.8, linewidth=1.5, color='blue')
        axes[i, 1].plot(val_pred, label='Reconstructed', alpha=0.7, linewidth=1.5, color='orange')
        axes[i, 1].set_title(
            f'Val - Channel {ch}\n'
            f'RMSE={val_rmse:.3f} | R²={val_r2:.3f} | r={val_corr:.3f} | MSE={val_mse:.3f}',
            fontsize=10
        )
        axes[i, 1].set_xlabel('Time')
        axes[i, 1].set_ylabel('Amplitude')
        axes[i, 1].legend(loc='upper right', fontsize=8)
        axes[i, 1].grid(True, alpha=0.3)
    
    # Overall title
    plt.suptitle('Raw EEG Reconstruction: Train vs Validation', fontsize=14, y=1.0)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("RECONSTRUCTION QUALITY SUMMARY")
    print("="*60)
    
    # Calculate overall metrics
    train_overall_rmse = np.sqrt(mean_squared_error(train_original.flatten(), train_reconstructed.flatten()))
    train_overall_r2 = r2_score(train_original.flatten(), train_reconstructed.flatten())
    train_overall_mse = mean_squared_error(train_original.flatten(), train_reconstructed.flatten())

    val_overall_rmse = np.sqrt(mean_squared_error(val_original.flatten(), val_reconstructed.flatten()))
    val_overall_r2 = r2_score(val_original.flatten(), val_reconstructed.flatten())
    val_overall_mse = mean_squared_error(val_original.flatten(), val_reconstructed.flatten())
    print(f"Train Set:")
    print(f"  Overall RMSE: {train_overall_rmse:.4f}")
    print(f"  Overall R²:   {train_overall_r2:.4f}")
    print(f"  Overall MSE:   {train_overall_mse:.4f}")
    print(f"\nValidation Set:")
    print(f"  Overall RMSE: {val_overall_rmse:.4f}")
    print(f"  Overall R²:   {val_overall_r2:.4f}")
    print(f"  Overall MSE:   {val_overall_mse:.4f}")
    print("="*60 + "\n")

def plot_raweeg_fft_reconstruction(
    trainer: "Trainer",
    n_channels: int = 4,
) -> None:
    """
    Function to plot FFT of raweeg reconstruction comparison using train and val data side by side.
    
    Processes both samples in a single forward pass for efficiency.
    
    Args:
        trainer: Trainer object with model, train_loader, val_loader, and device
        n_channels: number of random channels to plot
    """

    # Get a sample from train loader
    train_batch = next(iter(trainer.train_loader))
    train_original = train_batch['target'][0].cpu().numpy()  # (channels, time)
    train_input = train_batch['input'][0].to(trainer.device)
    
    # Get a sample from val loader
    val_batch = next(iter(trainer.val_loader))
    val_original = val_batch['target'][0].cpu().numpy()  # (channels, time)
    val_input = val_batch['input'][0].to(trainer.device)
    
    # Stack into a single batch of 2 samples
    combined_input = torch.stack([train_input, val_input], dim=0)
    
    # Single forward pass for both samples
    model = trainer.model
    model.eval()
    with torch.no_grad():
        combined_output = model(combined_input)
    
    # Extract reconstructions
    if isinstance(combined_output, dict):
        reconstructed = combined_output['reconstruction'].cpu().numpy()
    elif isinstance(combined_output, tuple):
        reconstructed = combined_output[0].cpu().numpy()
    else:
        reconstructed = combined_output.cpu().numpy()
    
    train_reconstructed = reconstructed[0]
    val_reconstructed = reconstructed[1]
    
    # Verify shapes
    assert train_original.shape == train_reconstructed.shape, \
        f"Train shape mismatch: {train_original.shape} vs {train_reconstructed.shape}"
    assert val_original.shape == val_reconstructed.shape, \
        f"Val shape mismatch: {val_original.shape} vs {val_reconstructed.shape}"
    
    # Get channel dimension
    total_channels = train_original.shape[0]
    n_channels = min(n_channels, total_channels)
    random_channels = np.random.choice(total_channels, size=n_channels, replace=False)
    
    # Create subplot grid: n_channels rows x 2 columns (train | val)
    fig, axes = plt.subplots(n_channels, 2, figsize=(15, 3 * n_channels))
    
    # Handle single channel case
    if n_channels == 1:
        axes = axes.reshape(1, -1)
    
    for i, ch in enumerate(random_channels):
        # === TRAIN COLUMN ===
        train_true = train_original[ch, :]
        train_pred = train_reconstructed[ch, :]
        
        # Compute FFT
        train_fft_true = np.abs(np.fft.rfft(train_true))
        train_fft_pred = np.abs(np.fft.rfft(train_pred))
        freqs = np.fft.rfftfreq(len(train_true))
        
        # Calculate metrics on FFT
        train_rmse = np.sqrt(mean_squared_error(train_fft_true, train_fft_pred))
        train_r2 = r2_score(train_fft_true, train_fft_pred)
        try:
            train_corr, _ = pearsonr(train_fft_true, train_fft_pred)
        except:
            train_corr = np.nan
        
        axes[i, 0].plot(freqs, train_fft_true, label='Original', alpha=0.8, linewidth=1.5, color='blue')
        axes[i, 0].plot(freqs, train_fft_pred, label='Reconstructed', alpha=0.7, linewidth=1.5, color='orange')
        axes[i, 0].set_title(
            f'Train - Channel {ch} FFT\n'
            f'RMSE={train_rmse:.3f} | R²={train_r2:.3f} | r={train_corr:.3f}',
            fontsize=10
        )
        axes[i, 0].set_xlabel('Frequency (normalized)')
        axes[i, 0].set_ylabel('Magnitude')
        axes[i, 0].legend(loc='upper right', fontsize=8)
        axes[i, 0].grid(True, alpha=0.3)
        
        # === VAL COLUMN ===
        val_true = val_original[ch, :]
        val_pred = val_reconstructed[ch, :]
        
        # Compute FFT
        val_fft_true = np.abs(np.fft.rfft(val_true))
        val_fft_pred = np.abs(np.fft.rfft(val_pred))
        freqs = np.fft.rfftfreq(len(val_true))
        
        # Calculate metrics on FFT
        val_rmse = np.sqrt(mean_squared_error(val_fft_true, val_fft_pred))
        val_r2 = r2_score(val_fft_true, val_fft_pred)
        try:
            val_corr, _ = pearsonr(val_fft_true, val_fft_pred)
        except:
            val_corr = np.nan
        
        axes[i, 1].plot(freqs, val_fft_true, label='Original', alpha=0.8, linewidth=1.5, color='blue')
        axes[i, 1].plot(freqs, val_fft_pred, label='Reconstructed', alpha=0.7, linewidth=1.5, color='orange')
        axes[i, 1].set_title(
            f'Val - Channel {ch} FFT\n'
            f'RMSE={val_rmse:.3f} | R²={val_r2:.3f} | r={val_corr:.3f}',
            fontsize=10
        )
        axes[i, 1].set_xlabel('Frequency (normalized)')
        axes[i, 1].set_ylabel('Magnitude')
        axes[i, 1].legend(loc='upper right', fontsize=8)
        axes[i, 1].grid(True, alpha=0.3)
    
    # Overall title
    plt.suptitle('FFT Reconstruction Quality: Train vs Validation', fontsize=14, y=1.0)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("FFT RECONSTRUCTION QUALITY SUMMARY")
    print("="*60)
    
    # Calculate overall FFT metrics
    train_fft_original = np.abs(np.fft.rfft(train_original, axis=1))
    train_fft_reconstructed = np.abs(np.fft.rfft(train_reconstructed, axis=1))
    val_fft_original = np.abs(np.fft.rfft(val_original, axis=1))
    val_fft_reconstructed = np.abs(np.fft.rfft(val_reconstructed, axis=1))
    
    train_overall_rmse = np.sqrt(mean_squared_error(train_fft_original.flatten(), train_fft_reconstructed.flatten()))
    train_overall_r2 = r2_score(train_fft_original.flatten(), train_fft_reconstructed.flatten())
    
    val_overall_rmse = np.sqrt(mean_squared_error(val_fft_original.flatten(), val_fft_reconstructed.flatten()))
    val_overall_r2 = r2_score(val_fft_original.flatten(), val_fft_reconstructed.flatten())
    
    print(f"Train Set (FFT):")
    print(f"  Overall RMSE: {train_overall_rmse:.4f}")
    print(f"  Overall R²:   {train_overall_r2:.4f}")
    print(f"\nValidation Set (FFT):")
    print(f"  Overall RMSE: {val_overall_rmse:.4f}")
    print(f"  Overall R²:   {val_overall_r2:.4f}")
    print("="*60 + "\n")

def plot_reconstruction_scatter_analysis(
    trainer: "Trainer",
    max_samples_per_plot: int = 10000  # Subsample for faster plotting
) -> None:
    """
    Comprehensive reconstruction analysis for VQAE model.
    Analyzes errors across magnitude/phase, frequency bands, and spatial locations.
    
    Args:
        trainer: Trainer object with model, train_loader, val_loader, and device
        max_samples_per_plot: Maximum number of points to plot (for speed)
    """
    
    # Get samples from both loaders
    train_batch = next(iter(trainer.train_loader))
    val_batch = next(iter(trainer.val_loader))
    
    train_input = train_batch['input'][0].unsqueeze(0).to(trainer.device)  # (1, 32, 640)
    val_input = val_batch['input'][0].unsqueeze(0).to(trainer.device)
    
    # Forward pass
    model = trainer.model
    model.eval()
    with torch.no_grad():
        train_output = model(train_input)
        val_output = model(val_input)
    
    # Extract reconstructions and targets - KEEP AS TENSORS FIRST
    train_recon_tensor = train_output['reconstruction']  # (B, 1750, 640)
    train_target_tensor = train_output['target']         # (B, 1750, 640)
    val_recon_tensor = val_output['reconstruction']
    val_target_tensor = val_output['target']
    
    print(f"Train reconstruction shape (tensor): {train_recon_tensor.shape}")
    print(f"Train target shape (tensor): {train_target_tensor.shape}")
    
    # Reshape BEFORE converting to numpy to preserve tensor operations
    B_train = train_recon_tensor.shape[0]
    B_val = val_recon_tensor.shape[0]
    
    # Reshape: (B, 1750, 640) -> (B, 2, 25, 7, 5, 640)
    # Order: channels (mag/phase), frequency, rows, cols, time
    train_recon_struct_tensor = train_recon_tensor.view(B_train, 2, 25, 7, 5, 640)
    train_target_struct_tensor = train_target_tensor.view(B_train, 2, 25, 7, 5, 640)
    val_recon_struct_tensor = val_recon_tensor.view(B_val, 2, 25, 7, 5, 640)
    val_target_struct_tensor = val_target_tensor.view(B_val, 2, 25, 7, 5, 640)
    
    # Now convert to numpy and extract first sample
    train_recon_struct = train_recon_struct_tensor[0].cpu().numpy()  # (2, 25, 7, 5, 640)
    train_target_struct = train_target_struct_tensor[0].cpu().numpy()
    val_recon_struct = val_recon_struct_tensor[0].cpu().numpy()
    val_target_struct = val_target_struct_tensor[0].cpu().numpy()
    
    print(f"\\nReshaped structure: {train_recon_struct.shape}")
    print(f"  Dimension 0 (mag/phase): {train_recon_struct.shape[0]}")
    print(f"  Dimension 1 (frequency): {train_recon_struct.shape[1]}")
    print(f"  Dimension 2 (rows):      {train_recon_struct.shape[2]}")
    print(f"  Dimension 3 (cols):      {train_recon_struct.shape[3]}")
    print(f"  Dimension 4 (time):      {train_recon_struct.shape[4]}")
    
    # Verify separation
    print(f"\\nMagnitude range: [{train_recon_struct[0].min():.4f}, {train_recon_struct[0].max():.4f}]")
    print(f"Phase range:     [{train_recon_struct[1].min():.4f}, {train_recon_struct[1].max():.4f}]")
    
    # ========================================================================
    # 1. SCATTER PLOTS: SEPARATE FOR MAGNITUDE AND PHASE
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Extract magnitude and phase separately
    # Index 0 = Magnitude, Index 1 = Phase
    train_mag_target = train_target_struct[0].flatten()  # Magnitude
    train_mag_recon = train_recon_struct[0].flatten()
    train_phase_target = train_target_struct[1].flatten()  # Phase
    train_phase_recon = train_recon_struct[1].flatten()
    
    val_mag_target = val_target_struct[0].flatten()
    val_mag_recon = val_recon_struct[0].flatten()
    val_phase_target = val_target_struct[1].flatten()
    val_phase_recon = val_recon_struct[1].flatten()
    
    print(f"\\nMagnitude train samples: {len(train_mag_target)}")
    print(f"Phase train samples:     {len(train_phase_target)}")
    
    # Subsample for plotting speed
    def subsample(target, recon, max_samples):
        if len(target) > max_samples:
            indices = np.random.choice(len(target), max_samples, replace=False)
            return target[indices], recon[indices]
        return target, recon
    
    train_mag_target_sub, train_mag_recon_sub = subsample(train_mag_target, train_mag_recon, max_samples_per_plot)
    train_phase_target_sub, train_phase_recon_sub = subsample(train_phase_target, train_phase_recon, max_samples_per_plot)
    val_mag_target_sub, val_mag_recon_sub = subsample(val_mag_target, val_mag_recon, max_samples_per_plot)
    val_phase_target_sub, val_phase_recon_sub = subsample(val_phase_target, val_phase_recon, max_samples_per_plot)
    
    # Top row: Magnitude
    # Train Magnitude
    axes[0, 0].scatter(train_mag_target_sub, train_mag_recon_sub, alpha=0.3, s=1, color='blue')
    lim_min = min(train_mag_target_sub.min(), train_mag_recon_sub.min())
    lim_max = max(train_mag_target_sub.max(), train_mag_recon_sub.max())
    axes[0, 0].plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='Perfect')
    axes[0, 0].set_xlabel('Target Magnitude')
    axes[0, 0].set_ylabel('Reconstructed Magnitude')
    axes[0, 0].set_title('Train: Magnitude Reconstruction')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Val Magnitude
    axes[0, 1].scatter(val_mag_target_sub, val_mag_recon_sub, alpha=0.3, s=1, color='blue')
    lim_min = min(val_mag_target_sub.min(), val_mag_recon_sub.min())
    lim_max = max(val_mag_target_sub.max(), val_mag_recon_sub.max())
    axes[0, 1].plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='Perfect')
    axes[0, 1].set_xlabel('Target Magnitude')
    axes[0, 1].set_ylabel('Reconstructed Magnitude')
    axes[0, 1].set_title('Val: Magnitude Reconstruction')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bottom row: Phase
    # Train Phase
    axes[1, 0].scatter(train_phase_target_sub, train_phase_recon_sub, alpha=0.3, s=1, color='green')
    lim_min = min(train_phase_target_sub.min(), train_phase_recon_sub.min())
    lim_max = max(train_phase_target_sub.max(), train_phase_recon_sub.max())
    axes[1, 0].plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='Perfect')
    axes[1, 0].set_xlabel('Target Phase')
    axes[1, 0].set_ylabel('Reconstructed Phase')
    axes[1, 0].set_title('Train: Phase Reconstruction')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Val Phase
    axes[1, 1].scatter(val_phase_target_sub, val_phase_recon_sub, alpha=0.3, s=1, color='green')
    lim_min = min(val_phase_target_sub.min(), val_phase_recon_sub.min())
    lim_max = max(val_phase_target_sub.max(), val_phase_recon_sub.max())
    axes[1, 1].plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='Perfect')
    axes[1, 1].set_xlabel('Target Phase')
    axes[1, 1].set_ylabel('Reconstructed Phase')
    axes[1, 1].set_title('Val: Phase Reconstruction')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Magnitude and Phase Reconstruction Scatter Analysis', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.show()
    
    # ========================================================================
    # 2. ERROR ANALYSIS BY MAGNITUDE/PHASE
    # ========================================================================
    print("\\n" + "="*80)
    print("ERROR ANALYSIS BY MAGNITUDE/PHASE")
    print("="*80)
    
    channel_names = ['Magnitude', 'Phase']
    
    for dataset_name, target_struct, recon_struct in [
        ('TRAIN', train_target_struct, train_recon_struct),
        ('VAL', val_target_struct, val_recon_struct)
    ]:
        print(f"\\n{dataset_name} SET:")
        print("-" * 80)
        print(f"{'Channel':<15} {'MAE':<12} {'MSE':<12} {'RMSE':<12} {'R²':<12}")
        print("-" * 80)
        
        for ch_idx, ch_name in enumerate(channel_names):
            ch_target = target_struct[ch_idx].flatten()
            ch_recon = recon_struct[ch_idx].flatten()
            
            error = ch_recon - ch_target
            mae = np.mean(np.abs(error))
            mse = np.mean(error ** 2)
            rmse = np.sqrt(mse)
            
            # R² score
            ss_res = np.sum((ch_target - ch_recon) ** 2)
            ss_tot = np.sum((ch_target - np.mean(ch_target)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            print(f"{ch_name:<15} {mae:<12.6f} {mse:<12.6f} {rmse:<12.6f} {r2:<12.6f}")
    
    # ========================================================================
    # 3. ERROR ANALYSIS BY FREQUENCY BAND
    # ========================================================================
    print("\\n" + "="*80)
    print("ERROR ANALYSIS BY FREQUENCY BAND (25 bands)")
    print("="*80)
    
    # Get actual frequency values from config
    frequencies = np.logspace(np.log10(0.5), np.log10(79.9), 25)
    
    for dataset_name, target_struct, recon_struct in [
        ('TRAIN', train_target_struct, train_recon_struct),
        ('VAL', val_target_struct, val_recon_struct)
    ]:
        print(f"\\n{dataset_name} SET:")
        print("-" * 80)
        print(f"{'Freq Band':<12} {'Freq (Hz)':<12} {'MAE':<12} {'MSE':<12} {'RMSE':<12} {'R²':<12}")
        print("-" * 80)
        
        for freq_idx in range(25):
            # Average over magnitude/phase for this frequency
            freq_target = target_struct[:, freq_idx].flatten()
            freq_recon = recon_struct[:, freq_idx].flatten()
            
            error = freq_recon - freq_target
            mae = np.mean(np.abs(error))
            mse = np.mean(error ** 2)
            rmse = np.sqrt(mse)
            
            ss_res = np.sum((freq_target - freq_recon) ** 2)
            ss_tot = np.sum((freq_target - np.mean(freq_target)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            print(f"Band {freq_idx:<6} {frequencies[freq_idx]:<12.2f} {mae:<12.6f} {mse:<12.6f} {rmse:<12.6f} {r2:<12.6f}")
    
    # ========================================================================
    # 4. ERROR ANALYSIS BY SPATIAL LOCATION (7x5 grid)
    # ========================================================================
    print("\\n" + "="*80)
    print("ERROR ANALYSIS BY SPATIAL LOCATION (7 rows × 5 cols)")
    print("="*80)
    
    for dataset_name, target_struct, recon_struct in [
        ('TRAIN', train_target_struct, train_recon_struct),
        ('VAL', val_target_struct, val_recon_struct)
    ]:
        print(f"\\n{dataset_name} SET:")
        print("-" * 80)
        print(f"{'Location':<12} {'MAE':<12} {'MSE':<12} {'RMSE':<12} {'R²':<12}")
        print("-" * 80)
        
        for row in range(7):
            for col in range(5):
                # Average over magnitude/phase and frequencies for this location
                loc_target = target_struct[:, :, row, col].flatten()
                loc_recon = recon_struct[:, :, row, col].flatten()
                
                error = loc_recon - loc_target
                mae = np.mean(np.abs(error))
                mse = np.mean(error ** 2)
                rmse = np.sqrt(mse)
                
                ss_res = np.sum((loc_target - loc_recon) ** 2)
                ss_tot = np.sum((loc_target - np.mean(loc_target)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                print(f"({row},{col}){'':<6} {mae:<12.6f} {mse:<12.6f} {rmse:<12.6f} {r2:<12.6f}")
    
    # ========================================================================
    # 5. OVERALL STATISTICS
    # ========================================================================
    print("\\n" + "="*80)
    print("OVERALL RECONSTRUCTION STATISTICS")
    print("="*80)
    
    # Flatten back for overall stats
    train_target_flat = train_target_struct.reshape(-1, 640)
    train_recon_flat = train_recon_struct.reshape(-1, 640)
    val_target_flat = val_target_struct.reshape(-1, 640)
    val_recon_flat = val_recon_struct.reshape(-1, 640)
    
    for dataset_name, target, recon in [
        ('TRAIN', train_target_flat, train_recon_flat),
        ('VAL', val_target_flat, val_recon_flat)
    ]:
        error = recon - target
        mae = np.mean(np.abs(error))
        mse = np.mean(error ** 2)
        rmse = np.sqrt(mse)
        
        ss_res = np.sum((target - recon) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"\\n{dataset_name} SET:")
        print(f"  MAE:  {mae:.6f}")
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  R²:   {r2:.6f}")
        print(f"  Target range:  [{target.min():.4f}, {target.max():.4f}]")
        print(f"  Recon range:   [{recon.min():.4f}, {recon.max():.4f}]")
    
    print("="*80 + "\\n")
    
    # ========================================================================
    # 6. VISUALIZATION: ERROR HEATMAPS AND FREQUENCY PROFILES
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Calculate spatial error maps (averaged over magnitude/phase, time, and frequency)
    train_spatial_mae = np.mean(np.abs(train_recon_struct - train_target_struct), axis=(0, 1, 4))
    val_spatial_mae = np.mean(np.abs(val_recon_struct - val_target_struct), axis=(0, 1, 4))
    
    # Calculate frequency error profiles (averaged over space and time)
    # Shape after mean: (2, 25) where dim 0 is mag/phase, dim 1 is frequency
    train_freq_mae = np.mean(np.abs(train_recon_struct - train_target_struct), axis=(2, 3, 4))
    val_freq_mae = np.mean(np.abs(val_recon_struct - val_target_struct), axis=(2, 3, 4))
    
    print(f"\\nFrequency MAE shape (train): {train_freq_mae.shape}")
    print(f"Magnitude MAE values (first 5): {train_freq_mae[0][:5]}")
    print(f"Phase MAE values (first 5):     {train_freq_mae[1][:5]}")
    
    # Plot spatial heatmaps
    im0 = axes[0, 0].imshow(train_spatial_mae, cmap='hot', aspect='auto')
    axes[0, 0].set_title('Train: Spatial MAE (7×5 grid)')
    axes[0, 0].set_xlabel('Column')
    axes[0, 0].set_ylabel('Row')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(val_spatial_mae, cmap='hot', aspect='auto')
    axes[0, 1].set_title('Val: Spatial MAE (7×5 grid)')
    axes[0, 1].set_xlabel('Column')
    axes[0, 1].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Plot frequency profiles with proper x-axis
    freq_indices = np.arange(25)
    
    # Train frequency profile
    axes[1, 0].plot(freq_indices, train_freq_mae[0], label='Magnitude', marker='o', linewidth=2, markersize=4)
    axes[1, 0].plot(freq_indices, train_freq_mae[1], label='Phase', marker='s', linewidth=2, markersize=4)
    axes[1, 0].set_title('Train: MAE by Frequency Band')
    axes[1, 0].set_xlabel('Frequency Band Index (0-24)')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(np.arange(0, 25, 5))
    
    # Val frequency profile
    axes[1, 1].plot(freq_indices, val_freq_mae[0], label='Magnitude', marker='o', linewidth=2, markersize=4)
    axes[1, 1].plot(freq_indices, val_freq_mae[1], label='Phase', marker='s', linewidth=2, markersize=4)
    axes[1, 1].set_title('Val: MAE by Frequency Band')
    axes[1, 1].set_xlabel('Frequency Band Index (0-24)')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(np.arange(0, 25, 5))
    
    plt.suptitle('Spatial and Frequency Error Analysis', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.show()