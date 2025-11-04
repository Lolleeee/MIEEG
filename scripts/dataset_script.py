from packages.data_objects.dataset import TorchDataset, TestTorchDataset
from packages.io.torch_dataloaders import get_data_loaders, _calc_norm_params
dataset = TorchDataset(root_folder="test/test_output")
# dataset.calculate_global_normalization_params()
# print(dataset.global_mean, dataset.global_std)

train_loader, val_loader, test_loader = get_data_loaders(dataset, sets_size={"train": 1, "val": 0}, norm_axes=(0, 2, 3, 4))
