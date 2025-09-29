from enum import Enum
import os
import glob
import numpy as np
import scipy.io as sio
import re

import logging 

CHANNELS = np.array([
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
    'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
    'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10'
])

from packages.io.input_loader import FileLoader

from packages.data_objects.signal import EegSignal, SignalObject

from packages.processing.wavelet import eeg_wavelet_transform


base_folder = "/media/lolly/Bruh/WAYEEGGAL_dataset/WAYEEG_preprocessed"
out_path = "/media/lolly/Bruh/WAYEEGGAL_dataset/WAYEEG_processed"
file_loader = FileLoader(root_folder=base_folder, folder_structure='patient', file_type='mat').load_data()

for patient_name, file_name, mat in file_loader:
    
    eeg_data = np.array(mat["trial_eeg"])

    kin_data = np.array(mat['trial_kin'])

    kin_data = kin_data[(3,7,11), :]
    
    EEG = EegSignal(unpacked_data=eeg_data, fs = 250, dim_dict={"channels": 0, "time": 1}, file_name=file_name)
    
    EEG = eeg_wavelet_transform(EEG, bandwidth=[1, 100], freq_samples=50)

    print(EEG.signal.shape)

    # spatial_eeg_tensor = reshape_to_spatial(eeg_tensor)

    # segmented_eeg_tensor, segmented_sensor_data = segment_data(eeg_data=spatial_eeg_tensor, sensor_data=kin_data, window=250, overlap=200, axis=-1, segment_sensor_signal=True)

    # displacements = window_delta_displacement(segmented_sensor_data, window= 250//2, offset=250//2)

    # save_tensors(out_path, patient_name, trial_id, eeg_data, sensor_data = None, out_format = 'npz', segmented = True)
    
    # print(f"Processing {patient_name} - {file_name} with EEG shape: {eeg_data.shape}")