from modules.basics.FileIO import Dataset, DatasetProfile, WayEEG, split_dataset
import modules.basics.ExploratoryAnalysis as exp
from modules.basics.PreProcessing import SignalICA, SignalResample, SignalFilter, PipeProcess
from modules.basics.Processing import SignalEpoch, SignalMashup, SignalPrint, SignalAbs
from modules.wavelet.ExploratoryAnalysis import WtTestPlots
from modules.wavelet.Processing import SignalCWT, Domain2D
import os
import torch
import numpy as np
from modules.dataset_sinth.Regression import pipe_ts_target
import dotenv
from modules.dataset_sinth.SetProd import SaveManager, target_processing
dotenv.load_dotenv(override=True)
dataset = Dataset()
target_list=[]
for i, dict in enumerate(dataset):
    target = dict['Targets']
    target_list.append(target)

import numpy as np
import torch
import napari
targets = torch.tensor(target_list)
# Assuming `valid_targets` is your filtered data tensor of shape (frames, 66, 3)
trajectory_data = targets  # Your filtered targets data

# Center each frame at the origin by subtracting the mean position of each frame
centered_trajectory_data = trajectory_data - trajectory_data.mean(dim=1, keepdim=True)

# Convert the centered data to numpy for visualization
centered_trajectory_data = centered_trajectory_data.numpy()
# Add a time axis: reshape to (frames, points, 4) where the first column is time
n_frames, n_points, _ = centered_trajectory_data.shape
time_data = np.repeat(np.arange(n_frames)[:, None], n_points, axis=0).reshape(-1, 1)
points_data = centered_trajectory_data.reshape(-1, 3)

# Combine time axis and point coordinates into a single array of shape (frames*points, 4)
points_with_time = np.hstack([time_data, points_data])

# Initialize Napari viewer
viewer = napari.Viewer(ndisplay=3)

# Add the points layer with time as the first axis
viewer.add_points(
    points_with_time,
    size=0.1,
    ndim=4,  # Enable 3D + time (t, z, y, x)
    face_color="cyan",
    name="Centered Trajectory",
)

# Start the Napari viewer
napari.run()



