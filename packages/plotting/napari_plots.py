import napari
import numpy as np
from packages.data_objects.signal import EegSignal, GLOBAL_DIM_KEYS
from typing import List


def raw_plot_spatial_eeg_tensor(eeg_tensor: np.ndarray) -> None:
    """
    WARNING: This function is experimental and may not work as intended.
    Plot the EEG tensor using napari with automated 3D volume view.
    Parameters:
    - eeg_tensor: 4D numpy array of shape (rows, cols, frequencies, samples) or 5D if segmented with shape (segments, rows, cols, frequencies, samples)
    """

    if eeg_tensor.ndim == 4:
        # Reorder: put frequencies as z, rows as y, cols as x, keep time as last axis
        eeg_tensor = np.moveaxis(eeg_tensor, 2, 0)  # shape: (freq, rows, cols, time)
        
        viewer = napari.Viewer()

        # Add as image volume with time slider
        layer = viewer.add_image(
            eeg_tensor,
            name='EEG Tensor',
            colormap='jet',
            contrast_limits=[np.min(eeg_tensor), np.max(eeg_tensor)],
            rgb=False,
            multiscale=False,
        )

        # Switch to 3D display
        viewer.dims.ndisplay = 2
        viewer.camera.angles = (30, 30, 0)
        viewer.camera.zoom = 1.5
        # Set interpolation for 3D rendering
        layer.interpolation3d = 'nearest'

        napari.run()

    if eeg_tensor.ndim == 5:
        # Combine segments and frequencies into one dimension for 3D visualization
        eeg_tensor = np.moveaxis(eeg_tensor, 3, 1)  # shape: (segments, freq, rows, cols, time)

        viewer = napari.Viewer()

        # Add as image volume with time slider
        layer = viewer.add_image(
            eeg_tensor,
            name='EEG Tensor',
            colormap='jet',
            contrast_limits=[np.min(eeg_tensor), np.max(eeg_tensor)],
            rgb=False,
            multiscale=False,
        )

        # Switch to 3D display
        viewer.dims.ndisplay = 3
        viewer.camera.angles = (30, 30, 0)
        viewer.camera.zoom = 1.5
        # Set interpolation for 3D rendering
        layer.interpolation3d = 'nearest'

        napari.run()
        
