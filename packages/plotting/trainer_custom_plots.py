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