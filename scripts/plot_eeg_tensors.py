import os
from packages.plotting.napari_plots import raw_plot_spatial_eeg_tensor
import copy
import numpy as np
from dotenv import load_dotenv
import torch
import tqdm

from packages.data_objects.dataset import FileDataset




base_folder = "/media/lolly/SSD/MotorImagery_Preprocessed/"

def unpack(input):
    return np.array(input["out_eeg"])

loader = FileDataset(root_folder=base_folder, yield_identifiers=True, unpack_func=unpack)

# Frequencies setup (unchanged)
frequencies = np.concatenate([
    np.linspace(1, 4, 3), np.linspace(4, 8, 5)[1:], 
    np.linspace(8, 13, 8)[1:], np.linspace(13, 30, 10)[1:], 
    np.linspace(30, 80, 8)[1:]
])
frequencies = tuple(frequencies.tolist())

patient, trial, eeg_data = next(iter(loader))

from packages.models.wavelet_head import CWTHead
model = CWTHead(
    frequencies=frequencies,
    fs=160,
    num_channels=32,
    bandwidth=1.0,
    trainable=False
)
out = model(torch.tensor(eeg_data).unsqueeze(0).float())  # (1, 2, 1, 7, 5, time)
out = out[0, 0, ...]
print("Output shape:", out.shape)
# print(signal[:,0,0,:])
raw_plot_spatial_eeg_tensor(out.detach().numpy())
