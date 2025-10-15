from dotenv import load_dotenv
import torch
from packages.data_objects.dataset import CustomTestDataset, TorchDataset
from packages.io.file_loader import get_data_loaders
from packages.plotting.tensor_plots import plot_dimension_distribuitions
from packages.processing.misc import calculate_global_normalization_params
load_dotenv()

dataset_path = "/media/lolly/Bruh/WAYEEGGAL_dataset/0.5subset_datanooverlap"

dataset = TorchDataset(root_folder=dataset_path)

train_loader, _, _ = get_data_loaders(dataset, sets_size={"train": 1, "val": 0})





out = calculate_global_normalization_params(train_loader)
print(f"Mean: {out['mean']}, Std: {out['std']}")
# plot_dimension_distribuitions(next(iter(train_loader))[0], dim_labels=['frequencies', 'rows', 'cols', 'time'])