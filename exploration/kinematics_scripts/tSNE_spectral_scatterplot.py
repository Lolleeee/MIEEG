"""
python_tsne_eeg_features.py

This script processes EEG .mat files stored in a folder structure like:
    Folder/P%d/HS_P%d_trial%d.mat

Each .mat contains two structs: EEG and kin. We take EEG.trial_eeg (32 x samples),
compute relative band powers, build feature vectors, reduce them with t-SNE,
and plot them with colors distinguishing patients.

Dependencies:
    pip install numpy scipy scikit-learn matplotlib

Usage:
    Adjust `base_folder` to point to your dataset folder.
    Run: python python_tsne_eeg_features.py
"""

import os
import glob
import numpy as np
import scipy.io as sio
import scipy.signal as signal
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ---------------------- USER PARAMETERS ----------------------
base_folder = "/media/lolly/Bruh/WAYEEGGAL_dataset/WAYEEG_preprocessed" 
fs = 250        # used if EEG does not contain a sampling rate
max_freq = 45           # max frequency to consider for relative power
bands = [(1,4),(4,8),(8,13),(13,30),(30,45)]
band_names = ["delta","theta","alpha","beta","gamma"]

# ---------------------- FUNCTIONS ----------------------

def compute_relative_bandpowers(x, fs):
    """Compute relative band power for a 1D signal."""
    f, pxx = signal.welch(x, fs=fs, nperseg=min(len(x),2*fs))
    idx = f <= max_freq
    f, pxx = f[idx], pxx[idx]
    total_power = np.trapezoid(pxx, f)
    rels = []
    for lo,hi in bands:
        idxb = (f >= lo) & (f < hi)
        band_power = np.trapezoid(pxx[idxb], f[idxb])
        rels.append(band_power/total_power if total_power > 0 else 0)
    return rels

# ---------------------- MAIN LOOP ----------------------
all_features = []
all_labels = []
file_list = []

patients = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder,d)) and d.startswith("P")]

for pf in patients:
    try:
        patient_id = int(pf.strip("P"))
    except:
        patient_id = pf
    folder_path = os.path.join(base_folder,pf)
    files = glob.glob(os.path.join(folder_path,f"HS_P{patient_id}_trial*.mat"))
    if not files:
        files = glob.glob(os.path.join(folder_path,"HS_P*_trial*.mat"))
    for path in files:
        mat = sio.loadmat(path,squeeze_me=True,struct_as_record=False)

        data = np.array(mat["trial_eeg"])
        
        feat = []
        for ch in range(data.shape[0]):
            x = data[ch,:].astype(float)
            x = x - np.mean(x)
            rels = compute_relative_bandpowers(x, fs)
            feat.extend(rels)
        all_features.append(feat)
        all_labels.append(patient_id)
        file_list.append(path)

all_features = np.array(all_features)
all_labels = np.array(all_labels)

if all_features.size == 0:
    raise RuntimeError("No valid EEG trials found.")

# ---------------------- FEATURE SCALING & t-SNE ----------------------
scaler = StandardScaler()
features_z = scaler.fit_transform(all_features)
tsne = TSNE(n_components=2, random_state=1, init='pca')
mappedX = tsne.fit_transform(features_z)

# ---------------------- PLOT ----------------------
unique_pats = np.unique(all_labels)
colors = plt.cm.tab10(np.linspace(0,1,len(unique_pats)))

plt.figure(figsize=(9,6))
for i, pid in enumerate(unique_pats):
    idx = all_labels == pid
    plt.scatter(mappedX[idx,0], mappedX[idx,1], s=36, color=colors[i], label=f"P{pid}", edgecolor='k')
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE: relative band-power features (each point = trial)")
plt.legend(loc="best", bbox_to_anchor=(1.05,1))
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------- SAVE ----------------------
# np.savez("tsne_features.npz", all_features=all_features, features_z=features_z, labels=all_labels, mappedX=mappedX, file_list=file_list, bands=bands, band_names=band_names)
# print(f"Done. Processed {all_features.shape[0]} trials from {len(unique_pats)} patients. Results saved to tsne_features.npz")