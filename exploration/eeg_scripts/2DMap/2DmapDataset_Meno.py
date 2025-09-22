import os

import torch

from modules.basics.FileIO import Dataset, WayEEG, split_dataset
from modules.basics.PreProcessing import SignalResample, SignalFilter, PipeProcess
from modules.basics.Processing import SignalEpoch
from modules.wavelet.Processing import SignalCWT, Domain2D
import numpy as np
from modules.dataset_sinth.Regression import pipe_ts_target
import dotenv
from modules.dataset_sinth.SetProd import SaveManager, target_processing
import pywt

dotenv.load_dotenv(override=True)

dataset = Dataset()
struct = next(iter(dataset))
data = WayEEG(struct)
domain = Domain2D(data, electrodes=list(data.electrodes))

params = [
    {'max_freq': 3, 'min_freq': 0.5, 'freq_resolution': 0.5},
    {'max_freq': 8, 'min_freq': 3, 'freq_resolution': 1},
    {'max_freq': 13, 'min_freq': 8, 'freq_resolution': 1},
    {'max_freq': 30, 'min_freq': 13, 'freq_resolution': 2},
    {'max_freq': 50, 'min_freq': 30, 'freq_resolution': 2},
]
for struct in dataset:
    manager = SaveManager(dataset_iter=dataset)
    data = WayEEG(struct)
    targets = np.array([data.x_pos, data.y_pos, data.z_pos]).T  # must be shape(N*3)
    targets = SignalFilter(fs=500, filter_type='bandpass', lowcut=0.001, highcut=2).apply(targets)
    eeg = data.eeg
    pipeline = PipeProcess()
    pipeline.add_step([
        SignalResample(new_fs=120, original_fs=500),
        SignalFilter(filter_type='bandpass', highcut=52),
        SignalCWT(wavelet_params=params, fs=120),
        SignalEpoch(epoch_length=0.333, overlap=0.9, axis=-1)
    ])

    piped_tensors = pipeline.run(eeg)
    piped_tensors = piped_tensors.transpose([1, 0, 2, 3])
    domain_tensors = domain.transform(piped_tensors)

    piped_targets = pipe_ts_target(targets, pipeline, 'zero', time_offset=0.1)

    filtered_piped_targets = target_processing(piped_targets, trajectory=False, velocity=True)

    for target, tensor in zip(filtered_piped_targets, domain_tensors):
        target = torch.from_numpy(target).to(torch.bfloat16)
        tensor = torch.from_numpy(tensor).to(torch.bfloat16)
        output = {"Tensor": tensor, "Targets": target}
        manager.save(output=output)
