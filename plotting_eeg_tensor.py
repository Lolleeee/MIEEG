import os
import glob
import numpy as np
import scipy.io as sio
import plotly.graph_objects as go
import re
from packages.processing.wavelet import wavelet_transform
from packages.processing.tensor_reshape import reshape_to_spatial, segment_data
from packages.plotting.napari_plots import plot_spatial_eeg_tensor
base_folder = "/media/lolly/Bruh/WAYEEGGAL_dataset/WAYEEG_preprocessed"
patient = "P1"
idx = [10]  # Indices of trials to plot

patient_id = int(patient.strip("P"))
folder_path = os.path.join(base_folder, patient)
files = glob.glob(os.path.join(folder_path, f"HS_P{patient_id}_trial*.mat"))
files.sort(key=lambda f: int(re.search(r'trial(\d+)', f).group(1)))
selected_files = [files[i] for i in idx]

fig = go.Figure()

for i, file in enumerate(selected_files):
    mat = sio.loadmat(file, squeeze_me=True, struct_as_record=False)
    eeg = np.array(mat["trial_eeg"])

eeg_tensor, _ = wavelet_transform(eeg, bandwidth=[1, 100], fs=250, num_samples=50, norm_out = True)

spatial_eeg_tensor = reshape_to_spatial(eeg_tensor)

segmented_eeg_tensor = segment_data(spatial_eeg_tensor, window=100, overlap=90, axis=-1)

plot_spatial_eeg_tensor(segmented_eeg_tensor)