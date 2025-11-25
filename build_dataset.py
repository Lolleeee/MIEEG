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
from packages.io.output_packager import save_signal
from packages.processing import misc, tensor_reshape, wavelet
from scripts import debug_constants


def main():
    load_dotenv()
    base_folder ="/media/lolly/SSD/WAYEEGGAL_dataset/WAYEEG_preprocessed/P1" #os.getenv("BASE_FOLDER")
    out_path = "scripts/test_output/TEST"#os.getenv("OUT_FOLDER")

    def unpack(input: Dict):
        eeg_data = np.array(input["trial_eeg"])
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

    for patient, trial, eeg_data in tqdm.tqdm(loader):
        
        EEG = EegSignal(
            unpacked_data=eeg_data,
            fs=160,
            dim_dict={"channels": 0, "time": 1},
            patient=patient,
            trial=trial,
            electrode_schema=debug_constants.CHANNELS_32,
        )

        EEG_Raw = copy.deepcopy(EEG)

        WaveletEEG = wavelet.eeg_wavelet_transform(EEG, bandwidth=[1, 80], freqs=frequencies)

        WaveletEEG = misc.get_magnitude_and_phase(WaveletEEG)

        WaveletEEG = tensor_reshape.reshape_to_spatial(
            WaveletEEG, debug_constants.SPATIAL_DOMAIN_MATRIX_32
        )  # Move time axis to front

        WaveletEEG = tensor_reshape.segment_signal(WaveletEEG, window=80, overlap=40)

        EEG_Raw = tensor_reshape.segment_signal(EEG_Raw, window=80, overlap=40)

        WaveletEEG.reorder_signal_dimensions(
            ["epochs", "complex", "frequencies", "rows", "cols", "time"]
        )

        # KIN = KinematicSignal(unpacked_data=kin_data, fs=250, dim_dict={"position": 0, "time": 1}, patient=patient, trial=trial)
        # print(KIN.signal.shape, KIN.dim_dict)
        # KIN = sensor_data.window_delta_value(KIN, window=250//2, offset=250//2, dim='time')
        # print(KIN.signal.shape, KIN.dim_dict)
        #out = {'tensor': WaveletEEG, 'eeg': EEG_Raw}

        WaveletEEG.signal = WaveletEEG.signal.astype(np.float16)
        EEG_Raw.signal = EEG_Raw.signal.astype(np.float16)


        out = {'tensor': WaveletEEG, 'eeg': EEG_Raw}
        save_signal(
            out,
            out_path=out_path,
            out_format="npz",
            separate_epochs=True,
            group_patients=False,
        )
        sys.exit()


if __name__ == "__main__":
    main()
    