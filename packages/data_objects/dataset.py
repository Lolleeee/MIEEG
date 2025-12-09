import os
import re
import logging
import glob
from typing import Any, Callable, Dict, List, Union, Tuple
import h5py
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Logging Setup
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("dataset")
logger.setLevel(logging.WARNING)

RANDOM_SEED = 42

# File Loader Mapping
FILE_LOADING_FUNCTIONS = {
    ".mat": loadmat,
    ".npy": np.load,
    ".npz": np.load,
    ".pt": torch.load,
}

def _filetype_loader(file_path: str) -> Any:
    _, ext = os.path.splitext(file_path)
    if ext in FILE_LOADING_FUNCTIONS:
        return FILE_LOADING_FUNCTIONS[ext](file_path)
    raise TypeError(f"Unsupported file type: {os.path.basename(file_path)}")

def default_unpack_func(data: Union[Dict[str, Any], np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert input into a standard dict {'input': ..., 'target': ...}.
    """
    package = {}
    if isinstance(data, (dict, np.lib.npyio.NpzFile)):
        assert len(data) <= 2, "Default unpack supports max 2 keys (input, target)."
        for i, value in enumerate(data.values()):
            key = 'input' if i == 0 else 'target'
            package[key] = value
    elif isinstance(data, (np.ndarray, torch.Tensor)):
        package['input'] = data
    else:
        raise TypeError(f'Data type {type(data)} not supported')
    return package

def autoencoder_unpack_func(data: Union[Dict[str, Any], np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Unpack function for autoencoder datasets where input == target.
    """
    package = {}
    if isinstance(data, (dict, np.lib.npyio.NpzFile)):
        assert len(data) == 1, "Autoencoder unpack expects exactly 1 key."
        package['input'] = next(iter(data.values()))
        package['target'] = package['input']
    elif isinstance(data, (np.ndarray, torch.Tensor)):
        package['input'] = data
        package['target'] = data
    else:
        raise TypeError(f'Data type {type(data)} not supported')
    return package
# ============================================================================
# BASE DATASET CLASSES (For raw file folders)
# ============================================================================

class BasicDataset(Dataset):
    def __init__(self, root_folder: str, unpack_func: Callable = None) -> None:
        self.root_folder = root_folder
        self.unpack_func = unpack_func if unpack_func is not None else default_unpack_func
        self.item_list: List[str] = []

        if os.path.exists(root_folder) and os.path.isdir(root_folder):
            self._get_item_list()
            if not self.item_list:
                raise ValueError(f"No files found in: {root_folder}")
        else:
            # Allow initialization for subclasses that don't use root_folder (like H5)
            pass

    def __len__(self) -> int:
        return len(self.item_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item_path = self.item_list[idx]
        try:
            packed_data = _filetype_loader(item_path)
            return self.unpack_func(packed_data)
        except Exception as e:
            logger.error(f"Error loading {item_path}: {e}")
            raise e

    def _get_item_list(self, base_folder: str = None):
        folder = base_folder or self.root_folder
        for item in os.listdir(folder):
            path = os.path.join(folder, item)
            if os.path.isdir(path):
                self._get_item_list(path)
            else:
                self.item_list.append(path)

class FileDataset(BasicDataset):
    """
    Dataset that loads files from folders and extracts Patient/Trial metadata 
    from filenames via Regex. 
    
    Essential for the 'Pre-processing' -> 'HDF5 Generation' step.
    """
    def __init__(
        self,
        root_folder: str,
        unpack_func: Callable[[Any], Any] = None,
        yield_identifiers: bool = False,
    ):
        super().__init__(root_folder, unpack_func)
        self.yield_identifiers = yield_identifiers
    
    def __getitem__(self, idx):
        # Load the actual data (e.g. .mat, .npy) via BasicDataset
        super_data = super().__getitem__(idx)
        
        if self.yield_identifiers:
            file_path = self.item_list[idx]
            patient, trial = self._infer_from_path_identifiers(file_path)
            return patient, trial, super_data
        else:
            return super_data
        
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def _infer_from_path_identifiers(self, file_name):
        patient = self._regex_patient(file_name)
        trial = self._regex_trial(file_name)

        if patient is None:
            logger.warning(f"Patient ID missing in filename: {os.path.basename(file_name)}")
        if trial is None:
            logger.warning(f"Trial ID missing in filename: {os.path.basename(file_name)}")

        return patient, trial
    
    def _regex_patient(self, file_name: str) -> Union[int, None]:
        # Matches "patient12", "p12", etc.
        patterns = [r"patient(\d+)", r"p(\d+)"]
        for pattern in patterns:
            match = re.search(pattern, file_name, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def _regex_trial(self, file_name: str) -> Union[int, None]:
        # Matches "trial12", "t12", etc.
        patterns = [r"trial(\d+)", r"t(\d+)"]
        for pattern in patterns:
            match = re.search(pattern, file_name, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None


class TorchDataset(BasicDataset):
    """Standard Dataset for individual files with Normalization & Augmentation hooks."""
    def __init__(self, root_folder: str, unpack_func: Callable = None, augmentation_func: Callable = None):
        super().__init__(root_folder, unpack_func)
        self._norm_params = None
        self._target_norm_params = None
        self._augmentation_func = augmentation_func

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        
        if self._norm_params is not None:
            data['input'] = self._normalize_item(data['input'], is_target=False)
        
        if self._target_norm_params is not None and 'target' in data:
            data['target'] = self._normalize_item(data['target'], is_target=True)
            
        if self._augmentation_func:
            data = self._augmentation_func(data)
        return data

    def _normalize_item(self, item, is_target=False):
        params = self._target_norm_params if is_target else self._norm_params
        mean, std = params
        
        if isinstance(item, np.ndarray):
            item = torch.from_numpy(item)
        
        # Ensure params are on same device/dtype if needed (usually CPU here)
        return (item - mean) / (std + 1e-10)


class TestTorchDataset(TorchDataset):
    """
    A dummy dataset for testing/debugging.
    - If root_folder is None: Generates random synthetic data (RAM-only).
    - If root_folder is provided: Samples N random files from disk.
    """
    __test__ = False  # Prevent PyTest from collecting this as a test class

    def __init__(
        self,
        root_folder: str = None,
        unpack_func: Union[Callable[[Any], Any], str] = default_unpack_func,
        nsamples: int = 10,
        shape: tuple = (25, 7, 5, 250), # Default shape for random generation
    ):  
        # Initialize base properties manually since we might not call super().__init__
        self._norm_params = None
        self._target_norm_params = None
        self._augmentation_func = None
        self.unpack_func = unpack_func
        self.nsamples = nsamples
        self.shape = shape
        self.root_folder = root_folder
        self.is_synthetic = (root_folder is None)

        if not self.is_synthetic:
            # --- MODE A: Subsample Real Files ---
            # Initialize basic file list logic
            super().__init__(root_folder, unpack_func)
            
            # If requested N is larger than available files, cap it
            if len(self.item_list) < self.nsamples:
                logger.warning(f"Requested {self.nsamples} samples, but found only {len(self.item_list)}. Using all.")
                self.nsamples = len(self.item_list)
            
            # Randomly subsample the file list
            # We use torch.randperm for deterministic randomness if seed is set
            indices = torch.randperm(len(self.item_list))[:self.nsamples]
            self.item_list = [self.item_list[i] for i in indices]
            logging.info(f"TestDataset: Selected {self.nsamples} real files from {root_folder}")
            
        else:
            # --- MODE B: Synthetic Data ---
            # Pre-generate random tensors in RAM
            logging.info(f"TestDataset: Generating {self.nsamples} synthetic samples of shape {shape}")
            self.item_list = self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """Generates a list of random dictionary samples."""
        data_list = []
        for _ in range(self.nsamples):
            # Generate random input
            input_tensor = torch.randn(*self.shape)
            
            # Generate dummy target (assuming classification or regression)
            # Adjust this shape if your target is different (e.g., mask, label)
            # Here assuming a simple label or smaller vector
            target_tensor = torch.randint(0, 2, (1,)).float() 
            
            data_list.append({
                'input': input_tensor,
                'target': target_tensor
            })
        return data_list

    def __getitem__(self, idx):
        if self.is_synthetic:
            # --- Synthetic Mode ---
            # Direct dictionary access (already in RAM)
            data = self.item_list[idx].copy() # Copy to avoid modifying original
            
            # Apply Norm/Augment (Same logic as TorchDataset)
            if self._norm_params is not None:
                data['input'] = self._normalize_item(data['input'], is_target=False)
            
            if self._target_norm_params is not None:
                data['target'] = self._normalize_item(data['target'], is_target=True)
                
            if self._augmentation_func:
                data = self._augmentation_func(data)
                
            return data
            
        else:
            # --- Real File Mode ---
            # Use parent class logic which handles loading from disk
            return super().__getitem__(idx)

# ============================================================================
# OPTIMIZED HDF5 DATASET (For Kaggle/HPC)
# ============================================================================

class TorchH5Dataset(Dataset):
    """
    HDF5 Dataset optimized for PyTorch DataLoader.
    Features: Lazy Loading, LZF support, Multiprocessing safety.
    """
    def __init__(self, h5_path: str, augmentation_func: Callable = None, unpack_func: Callable = None):
        self.h5_path = h5_path
        self._augmentation_func = augmentation_func
        self._unpack_func = unpack_func if unpack_func is not None else default_unpack_func
        self._norm_params = None
        self._target_norm_params = None
        
        # 1. Verify file exists and get length ONCE
        # We do NOT keep the file open here. That kills multiprocessing.
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
            
        with h5py.File(h5_path, 'r') as f:
            # Auto-detect dataset keys if possible, or fallback to defaults
            self.keys = [k for k in f.keys() if k not in ['patient_ids', 'trial_ids', 'segment_ids']]
            # Prioritize 'tensor' as input if present (legacy support)
            if 'tensor' in self.keys:
                self.input_key = 'tensor'
                self.target_key = 'eeg' if 'eeg' in self.keys else None
            else:
                # Fallback: assume first key is input
                self.input_key = self.keys[0]
                self.target_key = self.keys[1] if len(self.keys) > 1 else None
                
            self.length = f[self.input_key].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        # 2. LAZY LOADING: Open file inside the worker process
        if not hasattr(self, 'h5_file') or self.h5_file is None:
            # 'swmr=True' allows concurrent reads if writers exist
            # 'libver=latest' improves performance
            self.h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)
            self.input_dset = self.h5_file[self.input_key]
            self.target_dset = self.h5_file[self.target_key] if self.target_key else None

        # 3. Read Data (Chunk=1 optimization applies here automatically)
        input_data = torch.from_numpy(self.input_dset[idx]).float()
        
        data = self._unpack_func(input_data)
        
        if self.target_dset:
            data['target'] = torch.from_numpy(self.target_dset[idx]).float()

        # 4. Apply Normalization
        if self._norm_params is not None:
            data['input'] = self._normalize_item(data['input'], is_target=False)
            
        if self._target_norm_params is not None and 'target' in data:
            data['target'] = self._normalize_item(data['target'], is_target=True)

        # 5. Apply Augmentation
        if self._augmentation_func:
            data = self._augmentation_func(data)

        return data

    def _normalize_item(self, item, is_target=False):
        params = self._target_norm_params if is_target else self._norm_params
        if params is None: return item
        
        mean, std = params
        return (item - mean) / (std + 1e-10)
        
    def __del__(self):
        # Cleanup file handle if it exists
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()



class TestTorchH5Dataset(TorchH5Dataset):
    """
    Test/Debug version of TorchH5Dataset that supports sampling a subset.
    
    Useful for:
    - Quick prototyping with small data subsets
    - Debugging without processing entire dataset
    - Sanity checks on model/training pipeline
    
    Args:
        h5_path: Path to HDF5 file
        nsamples: Number of samples to use (None = all, int = random subset)
        augmentation_func: Optional augmentation function
        unpack_func: Optional unpack function
        seed: Random seed for reproducible sampling
    """
    __test__ = False  # Prevent PyTest from treating this as a test class
    
    def __init__(
        self,
        h5_path: str,
        nsamples: int = None,
        augmentation_func: Callable = None,
        unpack_func: Callable = None,
        seed: int = 42
    ):
        # Initialize parent class
        super().__init__(h5_path, augmentation_func, unpack_func)
        
        # Store full dataset length
        self.full_length = self.length
        
        # Determine subset indices
        if nsamples is not None:
            if nsamples > self.full_length:
                logging.warning(
                    f"Requested {nsamples} samples but dataset only has {self.full_length}. "
                    f"Using all samples."
                )
                nsamples = self.full_length
            
            # Create random subset indices
            torch.manual_seed(seed)
            self.subset_indices = torch.randperm(self.full_length)[:nsamples].tolist()
            self.length = len(self.subset_indices)
            
            logging.info(
                f"TestTorchH5Dataset: Using {self.length}/{self.full_length} samples from {h5_path}"
            )
        else:
            # Use all samples
            self.subset_indices = None
            logging.info(f"TestTorchH5Dataset: Using all {self.length} samples from {h5_path}")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Map subset index to full dataset index
        if self.subset_indices is not None:
            if idx >= len(self.subset_indices):
                raise IndexError(f"Index {idx} out of range for subset of size {len(self.subset_indices)}")
            actual_idx = self.subset_indices[idx]
        else:
            actual_idx = idx
        
        # Use parent class logic with mapped index
        return super().__getitem__(actual_idx)


class TestTorchH5DatasetContiguous(TorchH5Dataset):
    """
    Alternative version that takes the FIRST N samples instead of random sampling.
    Useful when you want consistent, reproducible subsets without randomness.
    
    Args:
        h5_path: Path to HDF5 file
        nsamples: Number of samples to use from the beginning (None = all)
        augmentation_func: Optional augmentation function
        unpack_func: Optional unpack function
    """
    __test__ = False
    
    def __init__(
        self,
        h5_path: str,
        nsamples: int = None,
        augmentation_func: Callable = None,
        unpack_func: Callable = None
    ):
        super().__init__(h5_path, augmentation_func, unpack_func)
        
        self.full_length = self.length
        
        if nsamples is not None:
            self.length = min(nsamples, self.full_length)
            logging.info(
                f"TestTorchH5DatasetContiguous: Using first {self.length}/{self.full_length} samples"
            )
        else:
            logging.info(f"TestTorchH5DatasetContiguous: Using all {self.length} samples")
    
    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of range for subset of size {self.length}")
        return super().__getitem__(idx)
    

    
# ============================================================================
# HELPER CLASSES
# ============================================================================

class AugmentedDataset(Dataset):
    """
    Wrapper to apply transforms ONLY to specific subsets (e.g. Train).
    Solves the issue where Subset() shares the underlying dataset object.
    """
    def __init__(self, dataset: Dataset, transform: Callable):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return self.transform(sample)

    def __len__(self):
        return len(self.dataset)
