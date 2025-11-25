import os
import sys
from typing import Dict
import copy
import numpy as np
from dotenv import load_dotenv
import torch
import tqdm

from packages.data_objects.signal import EegSignal
from packages.data_objects.dataset import FileDataset
from packages.processing import misc, tensor_reshape, wavelet
from scripts import debug_constants
import h5py
from packages.io.h5 import _create_kaggle_hdf5_file, batch_append_to_kaggle_hdf5, save_signal
import logging

logging.basicConfig(level=logging.INFO)

def main():
    load_dotenv()
    base_folder = "/media/lolly/SSD/MotorImagery_Preprocessed/P40"
    out_path = "scripts/test_output/TEST"
    
    # HDF5 Kaggle dataset path
    dataset_path = os.path.join(out_path, "motor_eeg_dataset.h5")
    os.makedirs(out_path, exist_ok=True)
    
    def unpack(input: Dict):
        eeg_data = np.array(input["out_eeg"])
        return eeg_data

    loader = FileDataset(
        root_folder=base_folder, yield_identifiers=True, unpack_func=unpack
    )
    
    # Frequency bins (30 total)
    frequencies = np.concatenate([
        np.linspace(1, 4, 3),        # Delta: 3 bins
        np.linspace(4, 8, 5)[1:],    # Theta: 4 bins
        np.linspace(8, 13, 8)[1:],   # Alpha: 7 bins
        np.linspace(13, 30, 10)[1:], # Beta: 9 bins
        np.linspace(30, 80, 8)[1:]   # Gamma: 7 bins
    ])
    frequencies = tuple(frequencies.tolist())
    
    # Track if this is first iteration (to create file)
    first_iteration = True
    
    # Batch accumulator for efficient writing
    batch_packages = []
    batch_patient_ids = []
    batch_trial_ids = []
    batch_seg_ids = []
    BATCH_SIZE = 100  # Write every 100 samples
    
    for patient, trial, eeg_data in tqdm.tqdm(loader):
        
        # Your existing processing pipeline
        EEG = EegSignal(
            unpacked_data=eeg_data,
            fs=160,
            dim_dict={"channels": 0, "time": 1},
            patient=patient,
            trial=trial,
            electrode_schema=debug_constants.CHANNELS_32,
        )
        
        EEG_Raw = copy.deepcopy(EEG)
        
        # Wavelet transform
        WaveletEEG = wavelet.eeg_wavelet_transform(
            EEG, bandwidth=[1, 80], freqs=frequencies
        )
        WaveletEEG = misc.get_magnitude_and_phase(WaveletEEG)
        WaveletEEG = tensor_reshape.reshape_to_spatial(
            WaveletEEG, debug_constants.SPATIAL_DOMAIN_MATRIX_32
        )
        
        # Segmentation (creates ~24 epochzs)
        WaveletEEG = tensor_reshape.segment_signal(WaveletEEG, window=80, overlap=40)
        EEG_Raw = tensor_reshape.segment_signal(EEG_Raw, window=80, overlap=40)
        
        # Reorder dimensions for network input
        WaveletEEG.reorder_signal_dimensions(
            ["epochs", "complex", "frequencies", "rows", "cols", "time"]
        )
        
        # Convert to float16 BEFORE creating SignalObject dict
        # (this saves memory during processing)
        WaveletEEG.signal = WaveletEEG.signal.astype(np.float16)
        EEG_Raw.signal = EEG_Raw.signal.astype(np.float16)
        out = {'tensor': WaveletEEG, 'eeg': EEG_Raw}
        save_signal(
            out,
            out_path,
            out_format="h5",
            separate_epochs=True,
            group_patients=True,
            use_float16=True,
            compression_level=5,
            kaggle_mode=True,
            kaggle_file_path=dataset_path
        )
            
if __name__ == "__main__":
    main()