import glob
import logging
import os
import re
from enum import Enum
from typing import Dict

import numpy as np
import scipy.io as sio
from dotenv import load_dotenv

from packages.data_objects.signal import EegSignal, KinematicSignal
from packages.io.output_packager import save_signal
from packages.data_objects.dataset import CustomTestDataset
from packages.plotting.napari_plots import raw_plot_spatial_eeg_tensor
from packages.processing import misc, sensor_data, tensor_reshape, wavelet
from packages.test import debug_constants, test_data_objects

load_dotenv()

dataset_path = "/media/lolly/Bruh/WAYEEGGAL_dataset/WAYEEG_autoencoder"

dataset = CustomTestDataset(root_folder=dataset_path, unpack_func='dict', nsamples=1, file_type='npz')
signal = dataset[0]
print(signal[:,0,0,:])
#signal = EegSignal(unpacked_data = signal, fs=250, dim_dict={'frequencies': 0, 'rows': 1, 'cols': 2, 'time': 3}, patient=1, trial=1)
raw_plot_spatial_eeg_tensor(signal.detach().numpy())
