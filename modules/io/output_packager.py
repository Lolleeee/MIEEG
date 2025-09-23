import numpy as np


def package_eeg_data(eeg_data: np.ndarray) -> np.ndarray:
    return None


class OutputPackager:
    def __init__(self, eeg_data: np.ndarray):
        self.eeg_data = eeg_data

    def package(self) -> np.ndarray:
        packaged_data = package_eeg_data(self.eeg_data)
        return packaged_data
