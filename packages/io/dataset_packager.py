import os
import zipfile
from dotenv import load_dotenv
import random
import shutil
import logging

from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
from typing import List
import re
load_dotenv()
os.getenv("DATASET_FOLDER")

def get_subset(patient_structure: bool = True, base_folder: str = None, out_folder: str = None, testset_path: str = None, sample_ratio: float = 0.1, selected_patients: List[str] = None, testset_others: bool = False):
    """
    This function scans the DATASET_FOLDER.
    Identifies if the data is organized in patients folders or directly in trial files.
    Saves a zipped subset of the data in the OUT_FOLDER.
    Subset files are selected randomly and can be filtered by patient IDs.
    Subset composition can be controlled by specifying the percentage of files to include from each patient.
    Testset_others is a option to create a test set from the files that were not selected in the subset.
    """
    if base_folder is None:
        base_folder = os.getenv("DATASET_FOLDER")
    if out_folder is None:
        out_folder = os.getenv("OUT_FOLDER")

    os.makedirs(out_folder, exist_ok=True)

    if testset_path is None:
        testset_path = os.path.join(os.path.dirname(out_folder), f"{os.path.basename(out_folder)}_testset")
    os.makedirs(testset_path, exist_ok=True)

    all_selected_files = []

    if patient_structure is False and selected_patients is not None:
        logging.warning("selected_patients is specified but patient_structure is False. Ignoring selected_patients.")

    if patient_structure:
        logging.info("Processing Data organized in patient folders.")
        patient_folders = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
        patient_folders = _filter_patients(patient_folders, selected_patients)
        
        for patient in patient_folders:
            patient_path = os.path.join(base_folder, patient)
            selected_files = _save_subset(patient_path, out_folder, sample_ratio, testset_path, testset_others)

            
            all_selected_files.extend(selected_files)
    else:
        logging.info("Data organized directly in trial files.")
        selected_files = _save_subset(base_folder, out_folder, sample_ratio, testset_path, testset_others)
        all_selected_files.extend(selected_files)


    _cut_zip(out_folder, testset_path, testset_others)


def _filter_patients(patient_folders, selected_patients):
    if selected_patients is not None:
        # Build regex patterns for each selected patient identifier
        patterns = [re.compile(rf".*{re.escape(str(pid))}.*", re.IGNORECASE) for pid in selected_patients]
        filtered_folders = [
            pf for pf in patient_folders
            if any(pattern.search(pf) for pattern in patterns)
        ]
        logging.info(f"Filtered patient folders by pattern: {filtered_folders}")
        return filtered_folders
    return patient_folders

def _save_subset(data_folder, out_folder, sample_ratio, testset_path, testset_others=False):
    trial_files = [
                f
                for f in os.listdir(data_folder)
                if os.path.isfile(os.path.join(data_folder, f))
            ]
    num_files_to_copy = max(1, int(sample_ratio * len(trial_files)))
    selected_files = random.sample(trial_files, num_files_to_copy)
    non_selected_files = list(set(trial_files) - set(selected_files))

    for file in selected_files:
        shutil.move(
            os.path.join(data_folder, file), os.path.join(out_folder, file)
        )
    if testset_others:
        for file in non_selected_files:
            shutil.move(
                os.path.join(data_folder, file), os.path.join(testset_path, file)
            )
    return selected_files, non_selected_files

def _cut_zip(out_folder, testset_path=None, testset_others=False):
    logging.info(f"Zipping subset folder {out_folder}...")
    parent_dir = os.path.dirname(out_folder)
    archive_path = os.path.join(parent_dir, os.path.basename(out_folder))
    shutil.make_archive(base_name=archive_path, format='zip', root_dir=out_folder)

    if testset_others:
        logging.info(f"Zipping test set folder {testset_path}...")
        shutil.make_archive(base_name=testset_path, format='zip', root_dir=os.path.dirname(testset_path))
        logging.info(f"Test set saved and zipped at {testset_path}.zip")

    logging.info(f"Subset saved and zipped at {archive_path}.zip")

if __name__ == "__main__":
    get_subset(sample_ratio=1, testset_others=False, base_folder="/media/lolly/SSD/WAYEEGGAL_dataset/WAYEEG_processsed", out_folder="/media/lolly/SSD/WAYEEGGAL_dataset/0.69subset_250_eeg_wav", patient_structure=True)