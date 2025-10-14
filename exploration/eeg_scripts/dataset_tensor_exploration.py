from dotenv import load_dotenv
from packages.data_objects.dataset import CustomTestDataset, TorchDataset
from packages.io.file_loader import get_data_loaders
from packages.plotting.tensor_plots import plot_dimension_distribuitions

load_dotenv()

dataset_path = "/media/lolly/Bruh/WAYEEGGAL_dataset/WAYEEG_Autoencoder"

dataset = TorchDataset(root_folder=dataset_path)

train_loader, _, _ = get_data_loaders(dataset, sets_size={"train": 1, "val": 0}, norm_axes=(0, 2, 3, 4))

plot_dimension_distribuitions(next(iter(train_loader))[0], dim_labels=['frequencies', 'rows', 'cols', 'time'])