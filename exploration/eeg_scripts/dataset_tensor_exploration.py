from dotenv import load_dotenv
from packages.data_objects.dataset import CustomTestDataset
from packages.plotting.tensor_plots import plot_dimension_distribuitions

load_dotenv()

dataset_path = "/media/lolly/Bruh/WAYEEGGAL_dataset/WAYEEG_Autoencoder"

dataset = CustomTestDataset(root_folder=dataset_path, nsamples=1)

signal = dataset[100]

plot_dimension_distribuitions(signal, dim_labels=['frequencies', 'rows', 'cols', 'time'])