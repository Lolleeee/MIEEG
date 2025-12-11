import sys
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging
import os
from packages.train.loss import VQAE23Loss
from packages.data_objects.dataset import H5Dataset, TorchH5Dataset
from packages.io.torch_dataloaders import get_data_loaders 
logging.basicConfig(level=logging.INFO)

def load_signal_hdf5(file_path: str):
    """
    Simple loader to check HDF5 file structure and content locally.
    Use this to inspect your saved file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    logging.info(f"Opening {file_path}...")
    
    with h5py.File(file_path, 'r') as f:
        # Check for Kaggle-style expandable datasets
        if 'tensor' in f.keys():
            # This is your Kaggle-optimized file
            num_samples = f['tensor'].shape[0]
            logging.info(f"✅ Found Kaggle-optimized dataset with {num_samples} samples")
            logging.info(f"   Tensor shape: {f['tensor'].shape}")
            logging.info(f"   EEG shape:    {f['eeg'].shape}")
            
            # Load one sample to verify
            sample_tensor = f['tensor'][0]
            sample_eeg = f['eeg'][0]
            
            logging.info(f"   Sample 0 tensor dtype: {sample_tensor.dtype}")
            logging.info(f"   Sample 0 tensor mean:  {np.mean(sample_tensor):.4f}")
            
            return {
                'type': 'kaggle',
                'num_samples': num_samples,
                'tensor_shape': f['tensor'].shape,
                'eeg_shape': f['eeg'].shape
            }
            
        # Check for standard epoch files
        else:
            # This is a standard single-epoch file
            logging.info("✅ Found standard signal file")
            keys = list(f.keys())
            logging.info(f"   Signals available: {keys}")
            
            data = {}
            for k in keys:
                data[k] = f[k][:]
                logging.info(f"   {k}: {data[k].shape}, {data[k].dtype}")
                
            return {
                'type': 'standard',
                'data': data
            }


def znorm(tensor: torch.Tensor, axes: tuple, eps: float = 1e-8) -> torch.Tensor:
    """
    Z-normalize tensor along specified axes.
    
    Args:
        tensor: Input tensor
        axes: Tuple of axes to normalize over (e.g., (0, 1) or (-1,))
        eps: Small constant for numerical stability
        
    Returns:
        Normalized tensor
    """
    mean = tensor.mean(dim=axes, keepdim=True)
    std = tensor.std(dim=axes, keepdim=True)
    return (tensor - mean) / (std + eps)


# Test Script
if __name__ == "__main__":
    # 1. Inspect the file
    file_path = "/media/lolly/SSD/motor_eeg_dataset_optimized.h5"
    
    print("=" * 60)
    print("OVERFITTING TEST - Single Sample")
    print("=" * 60)
    
    # Load dataset
    dataset = TorchH5Dataset(file_path)
    print(f"\nDataset: {len(dataset)} samples")
    
    train, _ ,_  = get_data_loaders(dataset, norm_axes=(0,5), target_norm_axes=(0,2), batch_size=32, max_norm_samples=1000)

    sys.exit()
    # Get single sample
    sample = dataset[0]
    single_input = sample['input'].unsqueeze(0)  # Add batch dim: (1, 2, 30, 7, 5, 80)
    single_target = sample['target'].unsqueeze(0)  # (1, 32, 80)
    
    print(f"Input shape:  {single_input.shape}")
    print(f"Target shape: {single_target.shape}")
    
    # ========================================
    # Z-NORMALIZATION
    # ========================================
    print("\n--- Applying Z-normalization ---")
    
    # Input: (1, 2, 30, 7, 5, 80) - normalize over (channels, freq, time)
    # Keep spatial structure intact, normalize across (1, 2, -1) = (complex, freq, time)
    print("Before norm - Input mean:", single_input.mean().item(), "std:", single_input.std().item())
    single_input = znorm(single_input, axes=(1, 2, -1))  # Normalize over (complex, freq, time)
    print("After norm  - Input mean:", single_input.mean().item(), "std:", single_input.std().item())
    
    # Target: (1, 32, 80) - normalize over (channels, time)
    print("Before norm - Target mean:", single_target.mean().item(), "std:", single_target.std().item())
    single_target = znorm(single_target, axes=(1, 2))  # Normalize over (channels, time)
    print("After norm  - Target mean:", single_target.mean().item(), "std:", single_target.std().item())
    
    # Create model
    from packages.models.vqae_light import VQAE23, VQAE23Config
    
    config = VQAE23Config(
        use_quantizer=False,  # Disable VQ for easier overfitting test
        use_cwt=True
    )
    
    model = VQAE23(config)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"VQ enabled: {config.use_quantizer}")
    
    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING - Overfitting Single Sample")
    print("=" * 60)
    
    num_epochs = 1000
    print_every = 50
    loss_fn = VQAE23Loss(bottleneck_var_weight=0.0, bottleneck_cov_weight=0.0, freq_weight=0.0)  
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward
        outputs = model(single_input)
        
        # Loss
        # loss = torch.nn.functional.mse_loss(
        #     outputs['reconstruction'], 
        #     single_target
        # )

        loss = loss_fn(outputs, 
            {'input': single_input, 'target': single_target}
        )

        
        # Backward
        loss['loss'].backward()
        optimizer.step()
        
        # Print progress
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{num_epochs} | Loss: {loss}")
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Final Loss: {loss['loss'].item():.6f}")
    
    # Check reconstruction quality
    with torch.no_grad():
        outputs = model(single_input)
        recon = outputs['reconstruction']
        
        mse = torch.nn.functional.mse_loss(recon, single_target).item()
        mae = torch.nn.functional.l1_loss(recon, single_target).item()
        
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        
        # Correlation
        recon_flat = recon.flatten()
        target_flat = single_target.flatten()
        corr = torch.corrcoef(torch.stack([recon_flat, target_flat]))[0, 1].item()
        print(f"Correlation: {corr:.4f}")
    
    print("\n✅ Overfitting test complete!")
    
    # Expected behavior:
    # - Loss should decrease significantly (> 10x reduction)
    # - Final loss should be very low (< 0.01)
    # - Correlation should approach 1.0
    # If not, there's a bug in the model architecture