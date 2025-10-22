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
    shape = matrix.shape
    num = shape[0] * shape[1]
    matrix_tensor = torch.arange(num, dtype=torch.float32).reshape(shape)
    combined_channels = matrix_tensor.reshape(matrix_tensor.shape[0] * matrix_tensor.shape[1]) 
    print(matrix_tensor)
    print(combined_channels)

    x = np.load("/home/lolly/Projects/MIEEG/test/test_output/TEST_SAMPLE_FOLDER/TEST_SAMPLE2 (Copy).npz")
    x = x['data']
    x = torch.Tensor(x)
    from packages.plotting.reconstruction_plots import plot_reconstruction_slices
    x = x[:,:,:, 32:64]
    print(x.shape)
    plot_reconstruction_slices(x, x, n_channels=6)
    