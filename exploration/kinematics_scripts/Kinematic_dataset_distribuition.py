"""
Loads all kinematic trials for a given patient, computes velocity, and stores them in a list.
for each trial, a window of 250 samples is taken with a certain overlap (e.g., 125 samples).
For each window the relative difference in position is computed and stored.
"""
import os
import glob
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------ USER PARAMETERS ------
base_folder = "/media/lolly/Bruh/WAYEEGGAL_dataset/WAYEEG_preprocessed"
patient = 'P1'
window_size = 250
overlap = 250//10
th = 1.5  # threshold for minimum displacement to consider


all_trials = []

patient_id = int(patient.strip("P"))
folder_path = os.path.join(base_folder, patient)
files = glob.glob(os.path.join(folder_path, f"HS_P{patient_id}_trial*.mat"))

for path in files:
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    kin = np.array(mat["trial_kin"])
    x = kin[3, :]
    y = kin[7, :]
    z = kin[11, :]
    n_points = len(x)
    windows = []
    for start in range(0, n_points - window_size + 1, window_size - overlap):

        end = start + window_size - 1
        dx = x[end] - x[start]
        dy = y[end] - y[start]
        dz = z[end] - z[start]
        displacement = np.array([dx, dy, dz])
        
        if np.linalg.norm(displacement) > th:
            windows.append(displacement)
    all_trials.append(windows)

all_displacements = np.vstack(all_trials)

# Plot the distribution of relative differences in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    all_displacements[:, 0], 
    all_displacements[:, 1], 
    all_displacements[:, 2], 
    alpha=0.5, 
    s=8  # smaller points
)
ax.set_title(f'3D Distribution of Relative Differences in Position for {patient}')
ax.set_xlabel('Delta X')
ax.set_ylabel('Delta Y')
ax.set_zlabel('Delta Z')
plt.show()

# Calculate the norm of each displacement
displacement_norm = np.linalg.norm(all_displacements, axis=1)

# Plot histograms for each dimension and the norm
fig, axs = plt.subplots(1, 4, figsize=(24, 5))

axs[0].hist(all_displacements[:, 0], bins=50, color='tab:blue', alpha=0.7)
axs[0].set_title('Distribution of ΔX')
axs[0].set_xlabel('ΔX')
axs[0].set_ylabel('Count')

axs[1].hist(all_displacements[:, 1], bins=50, color='tab:orange', alpha=0.7)
axs[1].set_title('Distribution of ΔY')
axs[1].set_xlabel('ΔY')
axs[1].set_ylabel('Count')

axs[2].hist(all_displacements[:, 2], bins=50, color='tab:green', alpha=0.7)
axs[2].set_title('Distribution of ΔZ')
axs[2].set_xlabel('ΔZ')
axs[2].set_ylabel('Count')

axs[3].hist(displacement_norm, bins=50, color='tab:purple', alpha=0.7)
axs[3].set_title('Distribution of |Δ| (Norm)')
axs[3].set_xlabel('|Δ|')
axs[3].set_ylabel('Count')

plt.tight_layout()
plt.show()

#save figure
fig.savefig(f'Kinematic_Distribution_{patient}.png', dpi=300)

#######################################################

# Normalize the displacements for each coordinate
mean_disp_x = np.mean(all_displacements, axis=0)
std_disp_x = np.std(all_displacements, axis=0)
normalized_displacements = (all_displacements[:,0] - mean_disp_x) / std_disp_x

mean_disp_y = np.mean(all_displacements, axis=0)
std_disp_y = np.std(all_displacements, axis=0)
normalized_displacements = (all_displacements[:,1] - mean_disp_y) / std_disp_y

mean_disp_z = np.mean(all_displacements, axis=0)
std_disp_z = np.std(all_displacements, axis=0)
normalized_displacements = (all_displacements[:,2] - mean_disp_z) / std_disp_z

# Plot the distribution of relative differences in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    normalized_displacements[:, 0], 
    normalized_displacements[:, 1], 
    normalized_displacements[:, 2], 
    alpha=0.5, 
    s=8  # smaller points
)
ax.set_title(f'3D Distribution of Relative Differences in Position for {patient}')
ax.set_xlabel('Delta X')
ax.set_ylabel('Delta Y')
ax.set_zlabel('Delta Z')
plt.show()

# Calculate the norm of each displacement
normalized_displacement_norm = np.linalg.norm(normalized_displacements, axis=1)

# Plot histograms for each dimension and the norm
fig, axs = plt.subplots(1, 4, figsize=(24, 5))

axs[0].hist(normalized_displacements[:, 0], bins=50, color='tab:blue', alpha=0.7)
axs[0].set_title('Distribution of ΔX')
axs[0].set_xlabel('ΔX')
axs[0].set_ylabel('Count')

axs[1].hist(normalized_displacements[:, 1], bins=50, color='tab:orange', alpha=0.7)
axs[1].set_title('Distribution of ΔY')
axs[1].set_xlabel('ΔY')
axs[1].set_ylabel('Count')

axs[2].hist(normalized_displacements[:, 2], bins=50, color='tab:green', alpha=0.7)
axs[2].set_title('Distribution of ΔZ')
axs[2].set_xlabel('ΔZ')
axs[2].set_ylabel('Count')

axs[3].hist(normalized_displacement_norm, bins=50, color='tab:purple', alpha=0.7)
axs[3].set_title('Distribution of |Δ| (Norm)')
axs[3].set_xlabel('|Δ|')
axs[3].set_ylabel('Count')

plt.tight_layout()
plt.show()
