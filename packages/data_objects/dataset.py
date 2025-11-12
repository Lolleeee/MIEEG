import glob
import os
import re
from typing import Any, Callable, Dict, List, Union, Tuple
from scipy.io import loadmat
import numpy as np
import scipy.io as sio
import torch
from dotenv import load_dotenv
from torch.utils.data import Dataset
from tqdm import tqdm

load_dotenv()
import logging

logging.basicConfig(level=logging.WARNING)
file_handler = logging.StreamHandler()
file_handler.setLevel(logging.WARNING)

logger = logging.getLogger("dataset")

logger.setLevel(logging.WARNING)
logger.addHandler(file_handler)
logger.propagate = False

RANDOM_SEED = 42

FILE_LOADING_FUNCTIONS = {
    ".mat": loadmat,
    ".npy": np.load,
    ".npz": np.load,
    ".pt": torch.load,
}

def _filetype_loader(file_path: str) -> Any:
    _, ext = os.path.splitext(file_path)
    if ext in FILE_LOADING_FUNCTIONS:
        file = FILE_LOADING_FUNCTIONS[ext](file_path)
    else:
        raise TypeError(f"Unsupported file type: {os.path.basename(file_path)}")
    return file 

def default_unpack_func(data: Dict[str, Any] | np.array | torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Default unpack function that converts input data into a dictionary with keys 'input' and optionally 'target'.
        If the input is a dictionary with one or two keys, the first key's value is assigned to 'input' and the second (if present) to 'target'.

        Note: In the current implementation it's assumed that if only one key is present, it corresponds to  both 'input' and 'target'.
        """
        package = {}
        
        if isinstance(data, dict) or isinstance(data, np.lib.npyio.NpzFile):

            assert len(data) <= 2, "Default unpack function only supports dictionaries with one or two keys. First will be 'input', second (optional) will be 'target'."
            
            for i, value in enumerate(data.values()):
                if i == 0:
                    package['input'] = value
                elif i == 1:
                    package['target'] = value

        elif isinstance(data, np.array) or isinstance(data, torch.Tensor):
            package['input'] = data

        else:
            raise TypeError(f'data type {type(data)} currently not supported')
        
        return package

class BasicDataset():
    def __init__(self, root_folder: str, unpack_func: Callable[[Any], Any] = None) -> None:

        self.root_folder: str = root_folder
        self.unpack_func: Callable[[Any], Any] = unpack_func if unpack_func is not None else default_unpack_func

        self.item_list: List[str] = []

        assert os.path.exists(root_folder), f"Root folder does not exist: {root_folder}"
        if os.path.isdir(root_folder):
            self._get_item_list()
            if len(self.item_list) == 0:
                raise ValueError(f"No files found in directory: {root_folder}")

    def __len__(self) -> int:
        return len(self.item_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item_path = self.item_list[idx]
        try:
            packed_data = _filetype_loader(item_path)
        except Exception as e:
            raise TypeError(f"Error loading {item_path}: {e}")
        
        try:
            unpacked_data = self.unpack_func(packed_data)

            return unpacked_data
        except Exception as e:
            raise ValueError(f"Error unpacking data object: {e}")
    # Recursively gather all file paths inside root_folder
    def _get_item_list(self, base_folder: str = None):
        if base_folder is None:
            base_folder = self.root_folder

        for item in os.listdir(base_folder):
            item_path = os.path.join(base_folder, item)
            if os.path.isdir(item_path):
                self._get_item_list(item_path)
            else:
                self.item_list.append(item_path)


class FileDataset(BasicDataset):
    def __init__(
        self,
        root_folder : str,
        unpack_func: Callable[[Any], Any] = None,
        yield_identifiers: bool = False,
    ):
        super().__init__(root_folder, unpack_func)
        self.yield_identifiers = yield_identifiers
    
    def __getitem__(self, idx):
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
            logging.warning(
                "Patient number not provided and could not be inferred from file name."
            )

        if trial is None:
            logging.warning(
                "Trial number not provided and could not be inferred from file name."
            )

        return patient, trial
    
    def _regex_patient(self, file_name: str) -> Union[int, None]:
        patterns = [r"patient(\d+)", r"p(\d+)"]
        for pattern in patterns:
            match = re.search(pattern, file_name, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def _regex_trial(self, file_name: str) -> Union[int, None]:
        patterns = [r"trial(\d+)", r"t(\d+)"]
        for pattern in patterns:
            match = re.search(pattern, file_name, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None




class TorchDataset(Dataset, BasicDataset):
    def __init__(
        self,
        root_folder: str,
        unpack_func: Callable = None,

    ):
        BasicDataset.__init__(self, root_folder, unpack_func)
        self._norm_params = None
        self._target_norm_params = None

    def __getitem__(self, idx):
        data = BasicDataset.__getitem__(self, idx)

        if self._norm_params is not None:
            data['input'] = self._normalize_item(data['input'])

        if self._target_norm_params is not None and 'target' in data:
            data['target'] = self._normalize_item(data['target'], is_target=True)

        return data

    def _normalize_item(self, item, is_target=False):
        if is_target:
            mean = self._target_norm_params[0]  
            std = self._target_norm_params[1]   
        else:
            mean = self._norm_params[0]  
            std = self._norm_params[1]   
        if isinstance(item, np.ndarray):
            item = torch.from_numpy(item)
        try:
            item = (item - mean) / (std + 1e-10)
            return item.float()
        except Exception as e:
            raise ValueError(f"Error normalizing data: {e}")


class TestTorchDataset(TorchDataset):
    __test__ = False  # prevent pytest from collecting this class as a test case
    def __init__(
        self,
        root_folder: str = None,
        unpack_func: Union[Callable[[Any], Any], str] = default_unpack_func,
        nsamples: int = 10,
        shape: tuple = (25, 7, 5, 250),
    ):  
        self._norm_params = None
        self.nsamples = nsamples
        self.shape = shape

        if root_folder is not None:
            super().__init__(root_folder, unpack_func)
        else: 
            self.item_list = self._get_random_item_list()
        
        if len(self.item_list) < self.nsamples:
            self.nsamples = len(self.item_list)
            logger.warning(f"Requested {self.nsamples} samples, but only found {len(self.item_list)} files: taking {len(self.item_list)} files.")
        
        torch.manual_seed(RANDOM_SEED)
        indices = torch.randperm(len(self.item_list))[:self.nsamples]
        self.item_list = [self.item_list[i] for i in indices]
        logging.info(f"Sampling {self.nsamples} items.")
        
    def _get_random_item_list(self):
        torch.manual_seed(RANDOM_SEED)
        item_list = [torch.randn(self.shape) for _ in range(self.nsamples)]
        assert all(isinstance(item, torch.Tensor) for item in item_list)
        return item_list

    def __getitem__(self, idx):
        # When item_list contains tensors (not file paths), return directly
        if isinstance(self.item_list[idx], torch.Tensor):
            data = self.item_list[idx]
            
            if self._norm_params is not None:
                data = self._normalize_item(data)
            
            return data.float()
        else:
            return super().__getitem__(idx)