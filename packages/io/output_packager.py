import logging
import os
from signal import signal
import sys
import time
from typing import Dict, List, Tuple, Union
import numpy as np

from packages.data_objects.signal import GLOBAL_DIM_KEYS, SignalObject

logging.basicConfig(level=logging.INFO)

def save_signal(
    signals_dict: Dict[str, SignalObject],
    out_path: str,
    out_format: str = "npz",
    separate_epochs: bool = True,
    group_patients: bool = False,
):
    """Saves the signal data to the specified output path in the desired format."""
    os.makedirs(out_path, exist_ok=True)

    # Check patient, trial, nEpochs consistency
    sample_sig = next(iter(signals_dict.values()))
    assert all(isinstance(sig, SignalObject) for sig in signals_dict.values())
    patient_id = sample_sig.patient
    trial_id = sample_sig.trial
    assert isinstance(patient_id, int) and isinstance(trial_id, int)

    if group_patients:
        os.makedirs(os.path.join(out_path, f"patient{sample_sig.patient}"), exist_ok=True)
        out_path = os.path.join(out_path, f"patient{sample_sig.patient}")
    
    assert all(signal.patient == patient_id for signal in signals_dict.values()), 'Found different patients in signal list, please check sync between patients'
    assert all(signal.trial == trial_id for signal in signals_dict.values()), 'Found different trials in signal list, please check sync between trials'
    assert all(sample_sig.epochs == sig.epochs for sig in signals_dict.values())
    
    if separate_epochs:
        package = {}
        for idx in range(sample_sig.epochs):
            for sig_name, sig in signals_dict.items():

                seg_axis = sig.dim_dict[GLOBAL_DIM_KEYS.EPOCHS.value]

                slices = [slice(None)] * sig.signal.ndim
                slices[seg_axis] = idx
                seg_data = sig.signal[tuple(slices)]
                
                package[sig_name] = seg_data

            _save_package(
                out_path, patient_id, trial_id, package, seg_idx=idx, fmt=out_format
            )
    else:
        package = signals_dict
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
        logging.info(f"Saved to {full_path}")
    else:
        raise ValueError(f"Unsupported format: {fmt}")


