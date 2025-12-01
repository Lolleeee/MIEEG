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
from packages.io.h5 import benchmark_loading_speed, save_signal, verify_optimization
import logging

logging.basicConfig(level=logging.INFO)

# ============================================================================
# KAGGLE ENVIRONMENT SETTINGS - UPDATED
# ============================================================================
KAGGLE_SETTINGS = {
    'target_batch_size': 64,     
    'compression_level': None,     # Set to None because we use LZF now
    'use_float16': True,
    'max_dataset_size_gb': 200,
}

def main():
    load_dotenv()
    base_folder = "/media/lolly/SSD/MotorImagery_Preprocessed/"
    out_path = "/media/lolly/SSD/motor_eeg_dataset"
    dataset_path = os.path.join(out_path, "motor_eeg_dataset_kaggle_optimized.h5")
    os.makedirs(out_path, exist_ok=True)
    
    def unpack(input: Dict):
        return np.array(input["out_eeg"])

    loader = FileDataset(root_folder=base_folder, yield_identifiers=True, unpack_func=unpack)
    
    # Frequencies setup (unchanged)
    frequencies = np.concatenate([
        np.linspace(1, 4, 3), np.linspace(4, 8, 5)[1:], 
        np.linspace(8, 13, 8)[1:], np.linspace(13, 30, 10)[1:], 
        np.linspace(30, 80, 8)[1:]
    ])
    frequencies = tuple(frequencies.tolist())
    
    total_samples = 0
    
    for patient, trial, eeg_data in tqdm.tqdm(loader):
        # ... (Processing pipeline same as before) ...
        EEG = EegSignal(
            unpacked_data=eeg_data, fs=160, dim_dict={"channels": 0, "time": 1},
            patient=patient, trial=trial, electrode_schema=debug_constants.CHANNELS_32,
        )
        
        EEG_Raw = copy.deepcopy(EEG)
        
        WaveletEEG = wavelet.eeg_wavelet_transform(EEG, bandwidth=[1, 80], freqs=frequencies)
        WaveletEEG = misc.get_magnitude_and_phase(WaveletEEG)
        WaveletEEG = tensor_reshape.reshape_to_spatial(WaveletEEG, debug_constants.SPATIAL_DOMAIN_MATRIX_32)
        
        WaveletEEG = tensor_reshape.segment_signal(WaveletEEG, window=80, overlap=0)
        EEG_Raw = tensor_reshape.segment_signal(EEG_Raw, window=80, overlap=0)
        
        WaveletEEG.reorder_signal_dimensions(["epochs", "complex", "frequencies", "rows", "cols", "time"])
        WaveletEEG.signal = WaveletEEG.signal.astype(np.float16)
        EEG_Raw.signal = EEG_Raw.signal.astype(np.float16)
        
        out = {'tensor': WaveletEEG, 'eeg': EEG_Raw}
        
        save_signal(
            out,
            out_path=out_path,
            out_format='h5',
            kaggle_mode=True,
            kaggle_file_path=dataset_path,
            use_float16=KAGGLE_SETTINGS['use_float16'],
            compression_level=None, # LZF ignores level
            target_batch_size=KAGGLE_SETTINGS['target_batch_size'],
            batch_append=True,
            separate_epochs=True
        )
        
        total_samples += WaveletEEG.epochs

    # ====================================================================
    # VERIFICATION
    # ====================================================================
    print("\n" + "="*70)
    print("DATASET CREATION COMPLETE")
    final_size_mb = os.path.getsize(dataset_path) / (1024**2)
    print(f"Size: {final_size_mb:.2f} MB")
    verify_optimization(dataset_path)
    benchmark_loading_speed(dataset_path)
    print("="*70 + "\n")

if __name__ == "__main__":
    main()