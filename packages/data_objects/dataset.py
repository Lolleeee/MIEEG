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

def _filetype_loader(file_path):
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


class FileLoader(BasicDataset):
    def __init__(
        self,
        root_folder,
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
        chunk_size: int = None
    ):
        BasicDataset.__init__(self, root_folder, unpack_func)
        self._norm_params = None
        self.chunk_size = chunk_size

    def __getitem__(self, idx):
        data = BasicDataset.__getitem__(self, idx)
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        if self._norm_params is not None:
            data = self._normalize_item(data)

        if self.chunk_size is not None:
            chunked_data = self._get_chunks(data)
        return chunked_data.float()

    def _normalize_item(self, item):
        mean = self._norm_params[0]  
        std = self._norm_params[1]   

        try:
            item = (item - mean) / (std + 1e-10)
            return item
        except Exception as e:
            raise ValueError(f"Error normalizing data: {e}")
    def _get_chunks(self, data):
        if data.shape[-1] < self.chunk_size:
                raise ValueError(f"Data length {data.shape[-1]} is smaller than chunk size {self.chunk_size}.")
            
        start_idx = 0
        
        n_chunks = data.shape[-1] // self.chunk_size
        total_len = n_chunks * self.chunk_size
        if n_chunks == 0:
            raise ValueError(f"Data length {data.shape[-1]} is smaller than chunk size {self.chunk_size}.")

        cropped = data[..., :total_len].squeeze(0)
        k = cropped.dim()  # original number of dims of cropped
        reshaped = cropped.reshape(*cropped.shape[:-1], n_chunks, self.chunk_size)
        
        data = reshaped.permute(k - 1, *range(0, k - 1), k)
        return data

class CustomTestDataset(Dataset, BasicDataset):
    def __init__(
        self,
        root_folder: str = None,
        unpack_func: Union[Callable[[Any], Any], str] = general_unpack_func,
        nsamples: int = 10,
        shape: tuple = (25, 7, 5, 250),
    ):  
        BasicDataset.__init__(self, root_folder, unpack_func)
        self.nsamples = nsamples
        self.shape = shape
        
        

        if self.root_folder is not None and os.path.isdir(self.root_folder):
            if len(self.item_list) < self.nsamples:
                self.nsamples = len(self.item_list)
                logger.warning(f"Requested {self.nsamples} samples, but only found {len(self.item_list)} files: taking {len(self.item_list)} files.")
            self.item_list = np.random.choice(self.item_list, self.nsamples, replace=False)
            self.use_files = True
        else:
            logging.warning("No root_folder or file_type provided. Using random data generation.")
            self.use_files = False

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        if self.use_files:
            data = BasicDataset.__getitem__(self, idx)
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            return data.float()
        else:
            np.random.seed(RANDOM_SEED + idx)
            data = np.random.randn(*self.shape).astype(np.float32)
            return torch.from_numpy(data).float()


