import os
import glob
import numpy as np
import scipy.io as sio
import re

import logging 

from packages.processing.wavelet import wavelet_transform
from packages.processing.tensor_reshape import reshape_to_spatial, segment_data
from packages.processing.sensor_data import window_delta_displacement
from packages.plotting.napari_plots import plot_spatial_eeg_tensor   
from packages.io.input_loader import FileLoader
from packages.io.output_packager import save_tensors

base_folder = "/media/lolly/Bruh/WAYEEGGAL_dataset/WAYEEG_preprocessed"
out_path = "/media/lolly/Bruh/WAYEEGGAL_dataset/WAYEEG_processed"
file_loader = FileLoader(root_folder=base_folder, folder_structure='patient', file_type='mat').load_data()

for patient_name, file_name, mat in file_loader:

    match = re.search(r"trial(\d+)", file_name)
    if match:
        trial_id = int(match.group(1))

    eeg_data = np.array(mat["trial_eeg"])
    
    kin_data = np.array(mat['trial_kin'])

    kin_data = kin_data[(3,7,11), :]
    
    eeg_tensor, _ = wavelet_transform(eeg_data, bandwidth=[1, 100], fs=250, num_samples=50, norm_out = True, abs_out=True)

    spatial_eeg_tensor = reshape_to_spatial(eeg_tensor)

    segmented_eeg_tensor, segmented_sensor_data = segment_data(eeg_data=spatial_eeg_tensor, sensor_data=kin_data, window=250, overlap=200, axis=-1, segment_sensor_signal=True)

    displacements = window_delta_displacement(segmented_sensor_data, window= 250//2, offset=250//2)

    save_tensors(out_path, patient_name, trial_id, eeg_data, sensor_data = None, out_format = 'npz', segmented = True)
    
    print(f"Processing {patient_name} - {file_name} with EEG shape: {eeg_data.shape}")