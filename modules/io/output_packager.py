import numpy as np


def package_eeg_data(eeg_data: np.ndarray) -> np.ndarray:
    return None


class OutputPackager:
    def __init__(self, eeg_data, sensor_data, out_format: str = 'npz'):
        self.eeg_data = eeg_data
        self.sensor_data = sensor_data
        self.package = self.package_builder(self.eeg_data, self.sensor_data)
        self.format = out_format

    def package_builder(self, eeg_data, sensor_data):
        return {'eeg': eeg_data, 'sens': sensor_data}
    
    def _save_package(self, filename):
        if self.format == 'npy':
            np.save(filename, self.package)
        elif self.format == 'npz':
            np.savez(filename, **self.package)
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def _iterate_segments(self, )