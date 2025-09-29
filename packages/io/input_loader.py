import os
from scipy.io import loadmat
from typing import Callable, Any, Dict, List
import logging
import re
from enum import Enum, auto
import numpy as np
logging.basicConfig(level=logging.WARNING)
file_handler = logging.StreamHandler()
file_handler.setLevel(logging.WARNING)
logger = logging.getLogger('input_loader')
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
    def __init__(self, root_folder, unpack_func: Callable[[Any], Any] = None, folder_structure: str = 'patient', file_type: str = 'mat'):
        self.root_folder = root_folder
        self.folder_structure = folder_structure
        self.file_type = file_type
        self.unpack_func = unpack_func
        
    def load_data(self):
        if self.folder_structure == 'patient':
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
                            
                        if self.unpack_func:
                            unpacked_data = self.unpack_func(packed_data)
                        else:
                            unpacked_data = packed_data

                        if unpacked_data is not None:
                            
                            

                            yield patient_name, file, unpacked_data
                        else:
                            continue  
                    except Exception as e:
                        logger.warning(f"Error loading {file_path}: {e}")
    # def add other loading methods if dataset structure is different
    
    def _data_loader(self, file_path):
        if file_path.endswith('.mat'):
            return loadmat(file_path)
        if file_path.endswith('.npy'):
            return np.load(file_path)
        else:
            logger.warning(f"Skipping unsupported file type: {os.path.basename(file_path)}")
            return None  # Skip unsupported file types/extensions
        

