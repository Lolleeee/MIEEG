
# Place all your existing code inside this function
import os
import torch
from modules.basics.FileIO import Dataset, WayEEG, split_dataset
from modules.basics.PreProcessing import SignalResample, SignalFilter, PipeProcess
from modules.basics.Processing import SignalEpoch, SignalZNorm
from modules.wavelet.Processing import SignalCWT, Domain2D
import numpy as np
import dotenv
from modules.dataset_sinth.SetProd import SaveManager, target_processing
dotenv.load_dotenv(override=True)

dataset = Dataset()
struct = next(iter(dataset))
data = WayEEG(struct)
domain = Domain2D(data, electrodes=list(data.electrodes))

params = [
    {'max_freq': 3, 'min_freq': 0.5, 'freq_resolution': 0.5},
    {'max_freq': 8, 'min_freq': 3, 'freq_resolution': 1},
    {'max_freq': 13, 'min_freq': 8, 'freq_resolution': 1},
    {'max_freq': 30, 'min_freq': 13, 'freq_resolution': 1},
    {'max_freq': 50, 'min_freq': 30, 'freq_resolution': 1},
]

pipeline = PipeProcess()
pipeline.add_step([
    SignalResample(new_fs=120, original_fs=500),
    SignalFilter(filter_type='bandpass', highcut=50),
    SignalCWT(wavelet_params=params, fs=120),
    SignalEpoch(epoch_length=1, overlap=0.7, axis=-1),
    SignalZNorm(axes=1)
])

for struct in dataset:
    manager = SaveManager(dataset_iter=dataset, save_format='npy')
    data = WayEEG(struct)
    eeg = data.eeg

    piped_tensors = pipeline.run(eeg)

    piped_tensors = piped_tensors.transpose([1,0,2,3])
    domain_tensors = domain.transform(piped_tensors)

    for tensor in domain_tensors:
        manager.save(output=tensor)

