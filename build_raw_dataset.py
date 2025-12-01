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
from packages.io.h5 import save_signal
from packages.processing import misc, tensor_reshape, wavelet
from scripts import debug_constants

KAGGLE_SETTINGS = {
    'target_batch_size': 64,     
    'compression_level': None,     # Set to None because we use LZF now
    'use_float16': True,
    'max_dataset_size_gb': 200,
}

def main():
    load_dotenv()
    base_folder = "/media/lolly/SSD/MotorImagery_Preprocessed"
    out_path = "/media/lolly/SSD/motor_eeg_dataset"
    dataset_path = os.path.join(out_path, "motor_eeg_dataset.h5")
    def unpack(input: Dict):
        eeg_data = np.array(input["out_eeg"])
        return eeg_data

    loader = FileDataset(
        root_folder=base_folder, yield_identifiers=True, unpack_func=unpack)
    
    frequencies = np.concatenate([
    np.linspace(1, 4, 3),       # Delta: 3 bins (keep all)
    np.linspace(4, 8, 5)[1:],   # Theta: 4 bins (skip first)
    np.linspace(8, 13, 8)[1:],  # Alpha: 7 bins (skip first)  
    np.linspace(13, 30, 10)[1:],# Beta: 9 bins (skip first)
    np.linspace(30, 80, 8)[1:]  # Gamma: 7 bins (skip first)
])
    frequencies = tuple(frequencies.tolist())

    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [0.4, 79.9], btype='bandpass', fs=160, output='sos')
    
    for patient, trial, eeg_data in tqdm.tqdm(loader):
        
        EEG = EegSignal(
            unpacked_data=eeg_data,
            fs=160,
            dim_dict={"channels": 0, "time": 1},
            patient=patient,
            trial=trial,
            electrode_schema=debug_constants.CHANNELS_32,
        )

        EEG = tensor_reshape.segment_signal(EEG, window=640, overlap=0)
        EEG.signal = sosfiltfilt(sos, EEG.signal, axis=-1)
        EEG.signal = EEG.signal.astype(np.float16)

        out = {'eeg': EEG}
        
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
        


if __name__ == "__main__":
    main()
    