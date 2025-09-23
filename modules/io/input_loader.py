import os
from scipy.io import loadmat


"""
    A class to load data files from a specified directory structure.
    Supports loading .mat files organized in patient-specific folders.
    Parameters:
    - root_folder: Root directory containing patient folders
    - folder_structure: Structure of folders ('patient' for patient-specific folders like "P1", "P2", etc. that contain files to be loaded)
    - file_type: Type of files to load (currently supports 'mat')
"""
class FileLoader:
    def __init__(self, root_folder, folder_structure: str = 'patient', file_type: str = 'mat'):
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
                        data = self._data_loader(file_path)
                        if data is not None:
                            yield patient_name, file, data
                        else:
                            continue  # Skip unsupported file types/extensions
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

    
    def _data_loader(self, file_path):
        if file_path.endswith('.mat'):
            return loadmat(file_path)
        else:
            print(f"Skipping unsupported file type: {os.path.basename(file_path)}")
            return None  # Skip unsupported file types/extensions