import glob
import os
import re
from typing import Any, Callable, Dict, Union, Tuple
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

def general_unpack_func(data: Dict[str, Any]) -> Any:
        data = data["data"]
        return data

class BasicDataset():
    def __init__(self, root_folder: str, unpack_func: Callable[[Any], Any] = None):

        self.root_folder = root_folder

        if unpack_func is None:
            self.unpack_func = general_unpack_func
        else:
            self.unpack_func = unpack_func

        self.item_list = []

        if root_folder is not None and os.path.isdir(root_folder):
            self._get_item_list()
            if len(self.item_list) == 0:
                raise ValueError(f"No files found in directory: {root_folder}")

    def __len__(self):
        return len(self.item_list)
    
    def __getitem__(self, idx):
        item_path = self.item_list[idx]
        try:
            packed_data = _filetype_loader(item_path)

            if self.unpack_func:
                unpacked_data = self.unpack_func(packed_data)
            else:
                unpacked_data = packed_data

            return unpacked_data
        except Exception as e:
            raise TypeError(f"Error loading {item_path}: {e}")

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
        unpack_func: Union[Callable[[Any], Any], str] = None,
    ):
        BasicDataset.__init__(self, root_folder, unpack_func)
        self._norm_params = None

    def __getitem__(self, idx):
        data = BasicDataset.__getitem__(self, idx)
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        if self._norm_params is not None:
            data = self._normalize_item(data)
        

        return data.float()

    def _normalize_item(self, item):
        mean = self._norm_params[0]  
        std = self._norm_params[1]   

        try:
            item = (item - mean) / (std + 1e-10)
            return item
        except Exception as e:
            raise ValueError(f"Error normalizing data: {e}")


class TestTorchDataset(TorchDataset):
    __test__ = False  # prevent pytest from collecting this class as a test case
    def __init__(
        self,
        root_folder: str = None,
        unpack_func: Union[Callable[[Any], Any], str] = general_unpack_func,
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