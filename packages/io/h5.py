import os
import time
import logging
from typing import Dict, Optional, List, Union
import numpy as np
import h5py
from packages.data_objects.signal import GLOBAL_DIM_KEYS, SignalObject
from packages.io.output_packager import _save_package
logging.basicConfig(level=logging.INFO)

def save_signal(
    signals_dict: Dict[str, SignalObject],
    out_path: str,
    out_format: str = "npz",
    separate_epochs: bool = True,
    group_patients: bool = False,
    # HDF5-specific parameters
    use_float16: bool = True,
    compression_level: int = 5,
    kaggle_mode: bool = False,  # Single expandable file vs multiple files
    kaggle_file_path: Optional[str] = None  # Path to existing Kaggle dataset
):
    """
    Saves the signal data to the specified output path in the desired format.
    
    Args:
        signals_dict: Dictionary of signal names to SignalObject instances
        out_path: Output directory path
        out_format: Output format ('npz', 'h5', 'hdf5', 'pkl')
        separate_epochs: If True, save each epoch separately
        group_patients: If True, create patient subdirectories
        use_float16: Convert float32/64 to float16 for HDF5 (2× compression)
        compression_level: GZIP compression level (0-9, higher=smaller+slower)
        kaggle_mode: If True, append to single expandable HDF5 file
        kaggle_file_path: Path to existing Kaggle dataset file (for appending)
    
    Example:
        # Standard mode (multiple files)
        save_signal(signals, '/output', 'h5', separate_epochs=True)
        
        # Kaggle mode (single expandable file)
        save_signal(signals, '/output', 'h5', kaggle_mode=True, 
                   kaggle_file_path='/output/dataset.h5')
    """
    os.makedirs(out_path, exist_ok=True)

    # Check patient, trial, nEpochs consistency
    sample_sig = next(iter(signals_dict.values()))
    assert all(isinstance(sig, SignalObject) for sig in signals_dict.values())
    patient_id = sample_sig.patient
    trial_id = sample_sig.trial
    assert isinstance(patient_id, int) and isinstance(trial_id, int)

    if group_patients:
        os.makedirs(os.path.join(out_path, f"patient{sample_sig.patient}"), exist_ok=True)
        out_path = os.path.join(out_path, f"patient{sample_sig.patient}")
    
    assert all(signal.patient == patient_id for signal in signals_dict.values()), \
        'Found different patients in signal list, please check sync between patients'
    assert all(signal.trial == trial_id for signal in signals_dict.values()), \
        'Found different trials in signal list, please check sync between trials'
    assert all(sample_sig.epochs == sig.epochs for sig in signals_dict.values())
    
    # Handle HDF5 formats
    is_hdf5 = out_format.lower() in ["h5", "hdf5"]
    
    if is_hdf5 and kaggle_mode:
        # Kaggle mode: append to single expandable file
        if kaggle_file_path is None:
            kaggle_file_path = os.path.join(out_path, "dataset.h5")
            # Create file if it doesn't exist
            if not os.path.exists(kaggle_file_path):
                _create_kaggle_hdf5_file(
                    kaggle_file_path,
                    signals_dict,
                    use_float16=use_float16,
                    compression_level=compression_level
                )
        
        # Append samples to Kaggle dataset
        if separate_epochs:
            for idx in range(sample_sig.epochs):
                package = {}
                for sig_name, sig in signals_dict.items():
                    seg_axis = sig.dim_dict[GLOBAL_DIM_KEYS.EPOCHS.value]
                    slices = [slice(None)] * sig.signal.ndim
                    slices[seg_axis] = idx
                    seg_data = sig.signal[tuple(slices)]
                    package[sig_name] = seg_data
                
                _append_to_kaggle_hdf5(
                    kaggle_file_path,
                    package,
                    patient_id=patient_id,
                    trial_id=trial_id,
                    seg_idx=idx,
                    use_float16=use_float16
                )
        else:
            package = {name: sig.signal for name, sig in signals_dict.items()}
            _append_to_kaggle_hdf5(
                kaggle_file_path,
                package,
                patient_id=patient_id,
                trial_id=trial_id,
                seg_idx=None,
                use_float16=use_float16
            )
    
    elif separate_epochs:
        # Standard mode: separate files per epoch
        for idx in range(sample_sig.epochs):
            package = {}
            for sig_name, sig in signals_dict.items():
                seg_axis = sig.dim_dict[GLOBAL_DIM_KEYS.EPOCHS.value]
                slices = [slice(None)] * sig.signal.ndim
                slices[seg_axis] = idx
                seg_data = sig.signal[tuple(slices)]
                package[sig_name] = seg_data

            if is_hdf5:
                _save_hdf5_optimized(
                    out_path,
                    package,
                    patient_id,
                    trial_id,
                    seg_idx=idx,
                    use_float16=use_float16,
                    compression_level=compression_level
                )
            else:
                _save_package(
                    out_path, patient_id, trial_id, package, seg_idx=idx, fmt=out_format
                )
    else:
        # Non-separated epochs
        package = {name: sig.signal for name, sig in signals_dict.items()}
        if is_hdf5:
            _save_hdf5_optimized(
                out_path,
                package,
                patient_id,
                trial_id,
                seg_idx=None,
                use_float16=use_float16,
                compression_level=compression_level
            )
        else:
            _save_package(
                out_path, patient_id, trial_id, package, seg_idx=None, fmt=out_format
            )


def _save_hdf5_optimized(
    out_path: str,
    package: Dict[str, np.ndarray],
    patient_id: int,
    trial_id: int,
    seg_idx: Optional[int] = None,
    use_float16: bool = True,
    compression_level: int = 5
) -> None:
    """
    Save single sample/epoch to HDF5 with full optimization.
    
    Applies: float16 conversion, GZIP compression, shuffle filter, optimal chunking.
    """
    if seg_idx is not None:
        fname = f"patient{patient_id}_trial{trial_id}_seg{seg_idx}.h5"
    else:
        fname = f"patient{patient_id}_trial{trial_id}.h5"
    
    full_path = os.path.join(out_path, fname)
    
    with h5py.File(full_path, 'w') as f:
        # Metadata
        f.attrs['patient_id'] = patient_id
        f.attrs['trial_id'] = trial_id
        if seg_idx is not None:
            f.attrs['segment_idx'] = seg_idx
        f.attrs['compression_level'] = compression_level
        f.attrs['float16_used'] = use_float16
        
        for sig_name, sig_data in package.items():
            # Convert to float16 if requested
            if use_float16 and sig_data.dtype in [np.float32, np.float64]:
                data = sig_data.astype(np.float16)
            else:
                data = sig_data
            
            # Calculate optimal chunks based on data shape
            chunks = _calculate_optimal_chunks(data.shape, sig_name)
            
            f.create_dataset(
                sig_name,
                data=data,
                compression='gzip',
                compression_opts=compression_level,
                chunks=chunks,
                shuffle=True  # Critical for compression!
            )
        
        file_size = os.path.getsize(full_path) / 1024
        logging.info(f"Saved {fname} ({file_size:.2f} KB, float16={use_float16})")


def _calculate_optimal_chunks(
        shape: tuple,
        sig_name: str,
        target_batch_size: int = 32,
        target_chunk_mb: float = 2.0
    ) -> tuple:
        """Calculate optimal chunk size for Kaggle training."""
        
        element_size = 2  # float16
        elements_per_sample = np.prod(shape)
        sample_size_mb = (elements_per_sample * element_size) / (1024**2)
        
        if 'tensor' in sig_name.lower():
            # Wavelet tensor: (2, 30, 7, 5, 80) ≈ 336 KB/sample
            # Use 16 samples (half batch) = 5.4 MB chunks
            chunk_samples = target_batch_size // 2
            return (chunk_samples,) + shape
        
        elif 'eeg' in sig_name.lower() or 'raw' in sig_name.lower():
            # Raw EEG: (32, 80) ≈ 5 KB/sample  
            # Use 128-256 samples = 0.6-1.3 MB chunks
            samples_for_target = int(target_chunk_mb / sample_size_mb)
            chunk_samples = min(max(samples_for_target, 128), 512)
            return (chunk_samples,) + shape
        
        else:
            # Fallback for any other signal type
            samples_for_target = int(target_chunk_mb / sample_size_mb)
            chunk_samples = min(max(samples_for_target, 16), 256)
            return (chunk_samples,) + shape


def _create_kaggle_hdf5_file(
    file_path: str,
    sample_signals: Dict[str, SignalObject],
    use_float16: bool = True,
    compression_level: int = 5,
    target_batch_size: int = 32  # Add this parameter
) -> None:
    """Create empty Kaggle-optimized HDF5 file with expandable datasets."""
    with h5py.File(file_path, 'w') as f:
        f.attrs['creation_time'] = time.time()
        f.attrs['format_version'] = '1.0'
        f.attrs['compression_level'] = compression_level
        f.attrs['float16_used'] = use_float16
        f.attrs['target_batch_size'] = target_batch_size  # Store for reference
        
        for sig_name, sig_obj in sample_signals.items():
            # Get single-epoch shape
            epoch_axis = sig_obj.dim_dict.get(GLOBAL_DIM_KEYS.EPOCHS.value)
            if epoch_axis is not None:
                shape = list(sig_obj.signal.shape)
                shape.pop(epoch_axis)
                shape = tuple(shape)
            else:
                shape = sig_obj.signal.shape
            
            # Determine dtype
            dtype = np.float16 if use_float16 and sig_obj.signal.dtype in [np.float32, np.float64] else sig_obj.signal.dtype
            
            # USE YOUR OPTIMIZED CHUNK CALCULATOR!
            chunk_shape = _calculate_optimal_chunks(
                shape, 
                sig_name, 
                target_batch_size=target_batch_size
            )
            
            # Calculate chunk size for logging
            chunk_mb = (np.prod(chunk_shape) * 2) / (1024**2)
            
            # Create expandable dataset
            f.create_dataset(
                sig_name,
                shape=(0,) + shape,
                maxshape=(None,) + shape,
                dtype=dtype,
                chunks=chunk_shape,
                compression='gzip',
                compression_opts=compression_level,
                shuffle=True
            )
            
            logging.info(
                f"Created '{sig_name}': shape={shape}, dtype={dtype}, "
                f"chunks={chunk_shape[0]} samples ({chunk_mb:.2f} MB/chunk)"
            )
        
        f.attrs['num_samples'] = 0
    
    logging.info(f"Created Kaggle HDF5 file: {file_path}")



def _append_to_kaggle_hdf5(
    file_path: str,
    package: Dict[str, np.ndarray],
    patient_id: int,
    trial_id: int,
    seg_idx: Optional[int] = None,
    use_float16: bool = True
) -> None:
    """
    Append single sample to Kaggle HDF5 file.
    
    This is called for each sample/epoch to build up the full dataset.
    """
    with h5py.File(file_path, 'a') as f:
        current_size = f.attrs['num_samples']
        new_size = current_size + 1
        
        # Append each signal
        for sig_name, sig_data in package.items():
            # Convert to float16 if requested
            if use_float16 and sig_data.dtype in [np.float32, np.float64]:
                data = sig_data.astype(np.float16)
            else:
                data = sig_data
            
            # Resize and append
            f[sig_name].resize((new_size,) + f[sig_name].shape[1:])
            f[sig_name][-1] = data
        
        # Update sample count
        f.attrs['num_samples'] = new_size
        
        if new_size % 100 == 0:  # Log every 100 samples
            file_size = os.path.getsize(file_path) / (1024**2)
            logging.info(f"Kaggle dataset: {new_size} samples, {file_size:.2f} MB")


# Batch append for efficiency (optional, for faster dataset creation)
def batch_append_to_kaggle_hdf5(
    file_path: str,
    packages: List[Dict[str, np.ndarray]],
    patient_ids: List[int],
    trial_ids: List[int],
    seg_ids: List[Optional[int]],
    use_float16: bool = True
) -> None:
    """
    Append multiple samples at once (10-100× faster than single appends).
    
    Use this when processing data in batches.
    """
    with h5py.File(file_path, 'a') as f:
        current_size = f.attrs['num_samples']
        batch_size = len(packages)
        new_size = current_size + batch_size
        
        # Get signal names from first package
        sig_names = list(packages[0].keys())
        
        # Batch append each signal type
        for sig_name in sig_names:
            # Stack all samples for this signal
            batch_data = np.stack([pkg[sig_name] for pkg in packages], axis=0)
            
            # Convert to float16
            if use_float16 and batch_data.dtype in [np.float32, np.float64]:
                batch_data = batch_data.astype(np.float16)
            
            # Resize and write batch
            f[sig_name].resize((new_size,) + f[sig_name].shape[1:])
            f[sig_name][current_size:new_size] = batch_data
        
        
        f.attrs['num_samples'] = new_size
        
        file_size = os.path.getsize(file_path) / (1024**2)
        logging.info(f"Batch appended {batch_size} samples (total: {new_size}, {file_size:.2f} MB)")
