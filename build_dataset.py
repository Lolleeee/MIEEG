from enum import Enum
import os
import glob
from typing import Dict
import numpy as np
import scipy.io as sio
import re
from dotenv import load_dotenv
import logging 

from packages.io.input_loader import FileLoader

from packages.data_objects.signal import EegSignal, KinematicSignal
from packages.processing import sensor_data
from packages.processing import wavelet
from packages.processing import misc
from packages.processing import tensor_reshape
from packages.test import debug_constants
from packages.io.output_packager import save_signal
load_dotenv() 
base_folder = os.getenv("BASE_FOLDER")
out_path = os.getenv("OUT_PATH")

def unpack(input: Dict):
    eeg_data = np.array(input["trial_eeg"])

    kin_data = np.array(input['trial_kin'])

    return eeg_data, kin_data


file_loader = FileLoader(root_folder=base_folder, folder_structure='patient', file_type='mat', unpack_func=unpack).load_data()

for patient, trial, (eeg_data, kin_data) in file_loader:

    EEG = EegSignal(unpacked_data=eeg_data, fs=250, dim_dict={"channels": 0, "time": 1}, patient=patient, trial=trial, electrode_schema=debug_constants.CHANNELS_32)

    EEG = wavelet.eeg_wavelet_transform(EEG, bandwidth=[1, 100], freq_samples=50)

    EEG = misc.absolute_values(EEG)

    EEG = misc.normalize_values(EEG, ['channels', 'time']) #TODO should I normalize also across channels? Maybe only through time

    EEG = tensor_reshape.reshape_to_spatial(EEG, debug_constants.SPATIAL_DOMAIN_MATRIX_32)  # Move time axis to front

    EEG = tensor_reshape.segment_signal(EEG, window=250, overlap=200)

    EEG._reorder_signal_dimensions(['epochs', 'frequencies', 'rows', 'cols', 'time'])


    # KIN = KinematicSignal(unpacked_data=kin_data, fs=250, dim_dict={"position": 0, "time": 1}, patient=patient, trial=trial)
    # print(KIN.signal.shape, KIN.dim_dict)
    # KIN = sensor_data.window_delta_value(KIN, window=250//2, offset=250//2, dim='time')
    # print(KIN.signal.shape, KIN.dim_dict)
    

    save_signal(EEG, out_path=out_path, out_format='npz', separate_epochs=True, group_patients=True)