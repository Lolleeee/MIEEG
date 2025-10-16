import os
from typing import Dict

import numpy as np
from dotenv import load_dotenv

from packages.data_objects.signal import EegSignal
from packages.data_objects.dataset import FileLoader
from packages.io.output_packager import save_signal
from packages.processing import misc, tensor_reshape, wavelet
from test import debug_constants


def main():
    load_dotenv()
    base_folder = os.getenv("BASE_FOLDER")
    out_path = os.getenv("OUT_FOLDER")

    def unpack(input: Dict):
        eeg_data = np.array(input["trial_eeg"])
        return eeg_data

    loader = FileLoader(
        root_folder=base_folder, yield_identifiers=True, unpack_func=unpack)

        

    for patient, trial, eeg_data in loader:

        EEG = EegSignal(
            unpacked_data=eeg_data,
            fs=250,
            dim_dict={"channels": 0, "time": 1},
            patient=patient,
            trial=trial,
            electrode_schema=debug_constants.CHANNELS_32,
        )

        EEG = wavelet.eeg_wavelet_transform(EEG, bandwidth=[1, 100], freq_samples=25)

        EEG = misc.absolute_values(EEG)

        EEG = tensor_reshape.reshape_to_spatial(
            EEG, debug_constants.SPATIAL_DOMAIN_MATRIX_32
        )  # Move time axis to front

        EEG = tensor_reshape.segment_signal(EEG, window=64, overlap=0)

        EEG._reorder_signal_dimensions(
            ["epochs", "frequencies", "rows", "cols", "time"]
        )

        # KIN = KinematicSignal(unpacked_data=kin_data, fs=250, dim_dict={"position": 0, "time": 1}, patient=patient, trial=trial)
        # print(KIN.signal.shape, KIN.dim_dict)
        # KIN = sensor_data.window_delta_value(KIN, window=250//2, offset=250//2, dim='time')
        # print(KIN.signal.shape, KIN.dim_dict)

        save_signal(
            EEG,
            out_path=out_path,
            out_format="npz",
            separate_epochs=True,
            group_patients=True,
        )


if __name__ == "__main__":
    main()
    