import os
from scipy.io import loadmat
from typing import Callable, Any, Dict
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

class DIM_DICT_KEYS(Enum):
    ROWS = 'rows'
    COLS = 'cols'
    FREQUENCIES = 'frequencies'
    TIME = 'time'
    EPOCHS = 'epochs'


"""
    A class to load data files from a specified directory structure.
    Supports loading .mat files organized in patient-specific folders.
    Parameters:
    - root_folder: Root directory containing patient folders
    - folder_structure: Structure of folders ('patient' for patient-specific folders like "P1", "P2", etc. that contain files to be loaded)
    - file_type: Type of files to load (currently supports 'mat')
"""
class FileLoader:
    def __init__(self, root_folder, unpack_func: Callable[[Any], Any], folder_structure: str = 'patient', file_type: str = 'mat'):
        self.root_folder = root_folder
        self.folder_structure = folder_structure
        self.file_type = file_type
        
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
        
"""
Class used to create a object of a signal in which data and metadata are stored.
Info like patient number and trial number are extracted from the file name if not provided.
To guide the data into processing functions the meaning of each dimension can be provided via a dictionary.
Parameters:
- unpacked_data: numpy array containing data
- dim_dict: Dictionary mapping dimension indices to their meanings (e.g., {'rows': 0, 'cols': 1, 'frequencies': 2, 'time': 3, 'epochs': 4})
- patient_number: Optional patient identifier
- trial_number: Optional trial identifier
- file_name: Optional file name to extract patient and trial identifiers if not provided 
"""
class SignalObject:
    def __init__(self, unpacked_data: np.ndarray, dim_dict: Dict[str, int], patient_number: int = None, trial_number: int = None, file_name: str = None):
        self.signal = unpacked_data

        self._infer_from_path_identifiers(patient_number, trial_number, file_name)

        self._check_dim_dict_dimensions(dim_dict)
        self.dim_dict = dim_dict

    def _infer_from_path_identifiers(self, patient_number, trial_number, file_name):

        if patient_number is not None:
            self.patient = patient_number
        if trial_number is not None:
            self.trial = trial_number
            
        if file_name is not None:
            patient_match = re.search(r'patient(\d+)', file_name, re.IGNORECASE)
            trial_match = re.search(r'trial(\d+)', file_name, re.IGNORECASE)

            if patient_match:
                self.patient = int(patient_match.group(1))
            if trial_match:
                self.trial = int(trial_match.group(1))

    def _check_dim_dict_dimensions(self, dim_dict: Dict):
        if dim_dict is None:
            raise ValueError("Dimension dictionary must be provided.")
        
        possible_keys = {key.value for key in DIM_DICT_KEYS}
        for key in dim_dict.keys():
            if key not in possible_keys:
                raise ValueError(f"Invalid dimension key: {key}. Must be one of {possible_keys}.")
            
        if len(set(dim_dict.values())) != len(dim_dict.values()):
            raise ValueError("Dimension indices must be unique.")
        


class EegSignal(SignalObject):
    def __init__(self
