import numpy as np

SPATIAL_DOMAIN_MATRIX_32 = np.array(
    [
        [None, "Fp1", None, "Fp2", None],
        ["F7", "F3", "Fz", "F4", "F8"],
        ["FC5", "FC1", "Cz", "FC2", "FC6"],
        ["T7", "C3", "CP1", "C4", "T8"],
        ["TP9", "CP5", "CP2", "CP6", "TP10"],
        ["P7", "P3", "Pz", "P4", "P8"],
        ["PO9", "O1", "Oz", "O2", "PO10"],
    ]
)
CHANNELS_32 = [
    'Fp1', 'Fp2',           # Frontal pole
    'F7', 'F3', 'Fz', 'F4', 'F8',  # Frontal
    'FC5', 'FC1', 'FC2', 'FC6',    # Frontal-central
    'T7', 'C3', 'Cz', 'C4', 'T8',  # Central/temporal
    'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',  # Central-parietal/temporal-parietal
    'P7', 'P3', 'Pz', 'P4', 'P8',  # Parietal
    'PO9', 'O1', 'Oz', 'O2', 'PO10'  # Parietal-occipital/occipital
]

CHANNELS_32 = np.array(
    [
        "Fp1",
        "Fp2",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "FC5",
        "FC1",
        "FC2",
        "FC6",
        "T7",
        "C3",
        "Cz",
        "C4",
        "T8",
        "TP9",
        "CP5",
        "CP1",
        "CP2",
        "CP6",
        "TP10",
        "P7",
        "P3",
        "Pz",
        "P4",
        "P8",
        "PO9",
        "O1",
        "Oz",
        "O2",
        "PO10",
    ]
)



if __name__ == "__main__":
    matrix = SPATIAL_DOMAIN_MATRIX_32
    import torch

    new_order_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 9, 10, 11, 12, 18, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    
    row_chan = CHANNELS_32[new_order_indices]
    matrix = row_chan.reshape(matrix.shape)
    print(matrix)

    