import glob
import os
import re
from typing import Any, Callable, Dict, Union

import numpy as np
import scipy.io as sio
import torch
from dotenv import load_dotenv
from torch.utils.data import Dataset

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


def unpack_func(data: Dict[str, Any]) -> Any:
    data = data["data"]
    return data


class Dataset(Dataset):
    def __init__(
        self,
        root_folder: str,
        unpack_func: Union[Callable[[Any], Any], str] = None,
        file_type: str = "npz",
    ):
        self.root_folder = root_folder
        self.file_type = file_type
        self.unpack_func = unpack_func
        self.file_list = self._gather_files()

        if self.file_list is None or len(self.file_list) == 0:
            logging.warning(f"No files found in {root_folder} with type {file_type}")

    def _gather_files(self):
        # Check if there are subdirectories (patient folders)
        subdirs = any(
            os.path.isdir(os.path.join(self.root_folder, item))
            for item in os.listdir(self.root_folder)
        )
        if subdirs:
            # If there are subdirectories, search recursively
            pattern = os.path.join(self.root_folder, "**", f"*.{self.file_type}")
            return glob.glob(pattern, recursive=True)
        if self.file_type == "npz":
            return glob.glob(os.path.join(self.root_folder, "*.npz"))
        elif self.file_type == "mat":
            return glob.glob(os.path.join(self.root_folder, "*.mat"))
        elif self.file_type == "npy":
            return glob.glob(os.path.join(self.root_folder, "*.npy"))
        elif self.file_type == "pt":
            return glob.glob(os.path.join(self.root_folder, "*.pt"))
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = self._load_file(file_path)
        if isinstance(self.unpack_func, Callable):
            data = self.unpack_func(data)
        elif isinstance(self.unpack_func, str) and self.unpack_func == "dict":
            data = unpack_func(data)

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        return data

    def _load_file(self, file_path: str):
        if self.file_type == "npz":
            return np.load(file_path)
        elif self.file_type == "mat":
            return sio.loadmat(file_path)
        elif self.file_type == "npy":
            return np.load(file_path)
        elif self.file_type == "pt":
            return torch.load(file_path)
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")

    @classmethod
    def get_test_dataset(
        cls,
        root_folder: str,
        nsamples=1,
        unpack_func: Union[Callable[[Any], Any], str] = None,
        file_type: str = "npz",
    ):
        instance = cls(
            root_folder=root_folder, unpack_func=unpack_func, file_type=file_type
        )
        random_indices = np.random.choice(len(instance), size=nsamples, replace=False)

        instance.file_list = [instance.file_list[i] for i in random_indices]
        return instance
