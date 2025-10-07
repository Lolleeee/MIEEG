import logging
import os
import re
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Union

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Subset

from packages.data_objects.dataset import RANDOM_SEED

logging.basicConfig(level=logging.WARNING)
file_handler = logging.StreamHandler()
file_handler.setLevel(logging.WARNING)
logger = logging.getLogger("input_loader")
logger.setLevel(logging.WARNING)
logger.addHandler(file_handler)
logger.propagate = False


"""
    A class to load data files from a specified directory structure.
    Supports loading .mat files organized in patient-specific folders.
    Parameters:
    - root_folder: Root directory containing patient folders
    - folder_structure: Structure of folders ('patient' for patient-specific folders like "P1", "P2", etc. that contain files to be loaded)
    - file_type: Type of files to load (currently supports 'mat')
    - unpack_func: Optional function to unpack loaded data (e.g., extract specific variables from .mat files)
"""


class FileLoader:
    def __init__(
        self,
        root_folder,
        unpack_func: Callable[[Any], Any] = None,
        folder_structure: str = "patient",
        file_type: str = "mat",
    ):
        self.root_folder = root_folder
        self.folder_structure = folder_structure
        self.file_type = file_type
        self.unpack_func = unpack_func

    def load_data(self):
        if self.folder_structure == "patient":
            return self._iter_patient_files()
        else:
            raise ValueError(f"Unsupported folder structure: {self.folder_structure}")

    def _iter_patient_files(self):
        for patient_name in os.listdir(self.root_folder):
            patient_folder = os.path.join(self.root_folder, patient_name)
            if os.path.isdir(patient_folder):
                for file in os.listdir(patient_folder):
                    file_path = os.path.join(patient_folder, file)
                    try:
                        packed_data = self._data_loader(file_path)

                        if packed_data is not None:
                            logger.info(f"Loaded file: {file_path}")

                        self._infer_from_path_identifiers(file_path)

                        if self.unpack_func:
                            unpacked_data = self.unpack_func(packed_data)
                        else:
                            unpacked_data = packed_data

                        if unpacked_data is not None:
                            yield self.patient, self.trial, unpacked_data
                        else:
                            continue
                    except Exception as e:
                        logger.warning(f"Error loading {file_path}: {e}")

    def _iter_other_folder_structure(self):
        pass

    def _data_loader(self, file_path):
        if file_path.endswith(".mat"):
            return loadmat(file_path)
        if file_path.endswith(".npy"):
            return np.load(file_path)
        else:
            logger.warning(
                f"Skipping unsupported file type: {os.path.basename(file_path)}"
            )
            return None  # Skip unsupported file types/extensions

    def _infer_from_path_identifiers(self, file_name):
        self.patient = self._regex_patient(file_name)
        self.trial = self._regex_trial(file_name)

        if not hasattr(self, "patient"):
            logging.warning(
                "Patient number not provided and could not be inferred from file name."
            )
            self.patient = None
        if not hasattr(self, "trial"):
            logging.warning(
                "Trial number not provided and could not be inferred from file name."
            )
            self.trial = None

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


def get_data_loaders(
    dataset: Dataset,
    batch_size: int = 32,
    sets_size: dict = {"train": 0.6, "val": 0.2, "test": 0.2},
    num_workers: int = 4,
) -> DataLoader:
    indices = np.arange(len(dataset))

    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)

    train_size = int(sets_size["train"] * len(dataset))
    val_size = int(sets_size["val"] * len(dataset))
    if "test" not in sets_size:
        test_size = len(dataset) - train_size - val_size
    else:
        test_size = int(sets_size["test"] * len(dataset))

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]

    if "test" not in sets_size:
        test_idx = indices[train_size + val_size :]
    else:
        test_idx = indices[train_size + val_size : train_size + val_size + test_size]

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader, test_loader


def get_cv_loaders_with_static_test(
    dataset, batch_size=8, n_splits=5, test_size=0.2, num_workers=4
):
    indices = np.arange(len(dataset))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
    n_test = int(test_size * len(dataset))
    test_idx = indices[:n_test]
    cv_idx = indices[n_test:]

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(test_idx),
        num_workers=num_workers,
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    for train_idx, val_idx in kf.split(cv_idx):
        # Map fold indices back to the original dataset indices
        train_sampler = SubsetRandomSampler(cv_idx[train_idx])
        val_sampler = SubsetRandomSampler(cv_idx[val_idx])
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers
        )
        yield train_loader, val_loader, test_loader


def get_test_loader(dataset: Dataset, batch_size=8, num_workers=4):
    indices = np.arange(len(dataset))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(indices),
        num_workers=num_workers,
    )
    return test_loader
