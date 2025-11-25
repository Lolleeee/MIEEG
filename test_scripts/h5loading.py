import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging
import os

from packages.data_objects.dataset import H5Dataset
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


# Test Script
if __name__ == "__main__":
    # 1. Inspect the file
    file_path = "scripts/test_output/TEST/motor_eeg_dataset.h5"  # Update this path!
    
    print("--- Inspecting File ---")
    try:
        info = load_signal_hdf5(file_path)
    except Exception as e:
        print(f"Error inspecting file: {e}")
        exit()
        
    if info['type'] == 'kaggle':
        # 2. Test PyTorch Loading (Simulating training loop)
        print("\n--- Testing PyTorch DataLoader ---")
        dataset = H5Dataset(file_path)
        
        # Test simple access
        sample = dataset[0]
        print(f"Sample 0 tensor shape: {sample['input'].shape}")
        
        train_loader, val_loader, test_loader = get_data_loaders(dataset, batch_size=32, norm_axes=[0,4], target_norm_axes=[0,2])
        
        
        print("\nIterating 1 batch...")
        for batch in train_loader:
            print(f"Batch tensor shape: {batch['input'].shape}")
            print(f"Batch eeg shape:    {batch['target'].shape}")
            print(f"Batch tensor mean: {torch.mean(batch['input']):.4f}, std: {torch.std(batch['input']):.4f}")
            print(f"Batch eeg mean:    {torch.mean(batch['target']):.4f}, std: {torch.std(batch['target']):.4f}")
            break  # Just one batch for test

        print("\n✅ SUCCESS: Dataset is ready for Kaggle training!")

    # Print dataset length and gb size
    print(f"\nDataset length: {len(dataset)} samples")
    file_size_gb = os.path.getsize(file_path) / (1024 **3)
    print(f"File size on disk: {file_size_gb:.2f} GB")

    # Print single sample kb size
    single_sample_size = info['tensor_shape'][1] * info['tensor_shape'][2] * info['tensor_shape'][3] * info['tensor_shape'][4] * 4 / 1024
    print(f"Single sample size in memory: {single_sample_size:.2f} KB")