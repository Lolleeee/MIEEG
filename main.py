import os
import glob
import numpy as np
import scipy.io as sio
from modules.processing.wavelet import wavelet_transform
from modules.processing.tensor_reshape import reshape_to_spatial, segment_eeg_data
from modules.plotting.napari_plots import plot_spatial_eeg_tensor   

base_folder = "/media/lolly/Bruh/WAYEEGGAL_dataset/WAYEEG_preprocessed"

# Load and preprocess EEG data for all patients
eeg_data_list = []
patient_folders = sorted(glob.glob(os.path.join(base_folder, "*")))

for patient_folder in patient_folders:
    eeg_files = glob.glob(os.path.join(patient_folder, "*.mat"))
    for eeg_file in eeg_files:
        mat_contents = sio.loadmat(eeg_file)
        
        eeg_data = mat_contents['trial_eeg']
        print(eeg_data.shape)