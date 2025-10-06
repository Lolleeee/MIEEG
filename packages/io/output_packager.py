import os
import time
from typing import Dict

import numpy as np

from packages.data_objects.signal import GLOBAL_DIM_KEYS, SignalObject


def save_signal(
    signal: SignalObject,
    out_path: str,
    out_format: str = "npz",
    separate_epochs: bool = True,
    group_patients: bool = False,
):
    """Saves the signal data to the specified output path in the desired format."""
    os.makedirs(out_path, exist_ok=True)
    if signal.patient is None or signal.trial is None:
        raise ValueError(
            "Signal object must have patient and trial information to be saved."
        )
    if separate_epochs and GLOBAL_DIM_KEYS.EPOCHS.value not in signal.dim_dict:
        raise ValueError(
            "Signal object does not contain segments dimension for separate epoch saving."
        )

    patient_id = signal.patient
    trial_id = signal.trial
    if group_patients:
        os.makedirs(os.path.join(out_path, f"patient{patient_id}"), exist_ok=True)
        out_path = os.path.join(out_path, f"patient{patient_id}")

    if separate_epochs and GLOBAL_DIM_KEYS.EPOCHS.value in signal.dim_dict:
        seg_axis = signal.dim_dict[GLOBAL_DIM_KEYS.EPOCHS.value]
        for idx in range(signal.signal.shape[seg_axis]):
            slices = [slice(None)] * signal.signal.ndim
            slices[seg_axis] = idx
            seg_data = signal.signal[tuple(slices)]

            package = {"data": seg_data}
            _save_package(
                out_path, patient_id, trial_id, package, seg_idx=idx, fmt=out_format
            )
    else:
        package = {"data": signal.signal}
        _save_package(
            out_path, patient_id, trial_id, package, seg_idx=None, fmt=out_format
        )


# Deprecated function, kept for reference
def dep_save_tensors(
    out_path: str,
    patient_id: str,
    trial_id: str,
    eeg_data: np.ndarray,
    sensor_data: np.ndarray = None,
    out_format: str = "npz",
    segmented: bool = True,
):
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
        return {"eeg": eeg_data, "sens": sensor_data}
    else:
        return {"eeg": eeg_data}


def _save_package(
    out_path: str,
    patient_id: str,
    trial_id: str,
    package: Dict,
    seg_idx=None,
    fmt: str = "npz",
):
    # Build filename
    if seg_idx is not None:
        fname = f"patient{patient_id}_trial{trial_id}_seg{seg_idx}.{fmt}"
    else:
        fname = f"patient{patient_id}_trial{trial_id}_{int(time.time())}.{fmt}"

    # Join with output directory
    full_path = os.path.join(out_path, fname)

    # Save
    if fmt == "npy":
        np.save(full_path, package)
    elif fmt == "npz":
        np.savez(full_path, **package)
        print(full_path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
