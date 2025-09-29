import numpy as np
import time
import os
from typing import Dict

def save_tensors(out_path: str, patient_id: str, trial_id: str, eeg_data: np.ndarray, sensor_data: np.ndarray = None, out_format: str = 'npz', segmented: bool = True):
    
    os.makedirs(out_path, exist_ok=True)

    if segmented:
        if sensor_data is not None:

            for idx, (e, s) in enumerate(zip(eeg_data, sensor_data)):
                package = _package_builder(e, s)
                _save_package(out_path, patient_id, trial_id, package, seg_idx=idx)
        else:

            for idx, e in enumerate(eeg_data):
                package = _package_builder(e, np.array([]))
                _save_package(out_path, patient_id, trial_id, package, seg_idx=idx)


def _package_builder(eeg_data: np.ndarray, sensor_data: np.ndarray):
    if sensor_data.size > 0:
        return {'eeg': eeg_data, 'sens': sensor_data}
    else:
        return {'eeg': eeg_data}


def _save_package(out_path: str, patient_id: str, trial_id: str, package: Dict, seg_idx=None, fmt: str = "npz"):
    # Build filename
    if seg_idx is not None:
        fname = f"patient{patient_id}_trial{trial_id}_seg{seg_idx}.{fmt}"
    else:
        fname = f"patient{patient_id}_trial{trial_id}_{int(time.time())}.{fmt}"

    # Join with output directory
    full_path = os.path.join(out_path, fname)

    # Save
    if fmt == 'npy':
        np.save(full_path, package)
    elif fmt == 'npz':
        np.savez(full_path, **package)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


    