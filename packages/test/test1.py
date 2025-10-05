from enum import Enum
import os
import glob
from typing import Dict
import numpy as np
import scipy.io as sio
import re
from dotenv import load_dotenv
import logging 

from packages.plotting.napari_plots import raw_plot_spatial_eeg_tensor
from packages.test import test_objects, debug_constants

from packages.data_objects.signal import EegSignal, KinematicSignal
from packages.processing import sensor_data
from packages.processing import wavelet
from packages.processing import misc
from packages.processing import tensor_reshape
from packages.test import debug_constants
from packages.io.output_packager import save_signal
load_dotenv() 

out_path = "packages/test/test_output"

EEG = EegSignal.random(shape=(32, 1000), fs=250, electrode_schema=debug_constants.CHANNELS_32, dim_dict={'channels': 0, 'time': 1})

EEG = wavelet.eeg_wavelet_transform(EEG, bandwidth=[1, 100], freq_samples=50)

EEG = misc.absolute_values(EEG)

EEG = misc.normalize_values(EEG, ['channels', 'time']) 

EEG = tensor_reshape.reshape_to_spatial(EEG, debug_constants.SPATIAL_DOMAIN_MATRIX_32)  # Move time axis to front

EEG = tensor_reshape.segment_signal(EEG, window=250, overlap=200)
# KIN = KinematicSignal(unpacked_data=kin_data, fs=250, dim_dict={"position": 0, "time": 1}, patient=patient, trial=trial)
# print(KIN.signal.shape, KIN.dim_dict)
# KIN = sensor_data.window_delta_value(KIN, window=250//2, offset=250//2, dim='time')
# print(KIN.signal.shape, KIN.dim_dict)
EEG._reorder_signal_dimensions(['epochs', 'frequencies', 'rows', 'cols', 'time'])

save_signal(EEG, out_path=out_path, out_format='npz', separate_epochs=True)

# raw_plot_spatial_eeg_tensor(EEG.signal)