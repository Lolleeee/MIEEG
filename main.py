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

        eeg_tensor, _ = wavelet_transform(eeg_data, bandwidth=[1, 100], fs=250, num_samples=50, norm_out = True, abs_out=True)

        spatial_eeg_tensor = reshape_to_spatial(eeg_tensor)

        segmented_eeg_tensor = segment_eeg_data(spatial_eeg_tensor, window=250, overlap=200, axis=-1)

        # save each segment of the segmented tensor
        for segment_idx in range(segmented_eeg_tensor.shape[0]):
            segment = segmented_eeg_tensor[segment_idx]
            save_path = os.path.join(patient_folder, f"segmented_tensor_segment{segment_idx+1}.npy")
            np.save(save_path, segment)
        break