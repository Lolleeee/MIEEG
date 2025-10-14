
from packages.models.autoencoder_plus import Conv3DAEP
from packages.plotting.reconstruction_plots import plot_reconstruction_distribution
from packages.train.training import train_model
from packages.train.loss import VaeLoss, CustomMSELoss
from packages.train.helpers import BackupManager, EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from packages.io.file_loader import get_data_loaders
import torch
import os
from packages.data_objects.dataset import TorchDataset, CustomTestDataset
from dotenv import load_dotenv
# model = Conv3DAE(in_channels=25, embedding_dim=128, hidden_dims=[64, 128, 256], use_convnext=False)


from packages.models.autoencoder_plus import Conv3DAEP
model = Conv3DAEP(in_channels=25, latent_dim=128)

load_dotenv()
dataset_path = "/media/lolly/Bruh/WAYEEGGAL_dataset/0.4subset_data"
# Dummy training loop
optimizer = torch.optim.AdamW
criterion = CustomMSELoss()
mae = torch.nn.L1Loss

config = {
    'batch_size': 2,
    'lr': 1e-4,
    'epochs': 100,
    # 'EarlyStopping' : {'patience': 10000, 'min_delta': 0.0},
    # 'BackupManager': {'backup_interval': 100000, 'backup_path': './model_backups'},
    # 'ReduceLROnPlateau': {'patience': 25, 'factor': 0.5},
    'history_plot': {'plot_type': 'extended', 'save_path': './training_history'}
}

metrics = {}

dataset = TorchDataset(root_folder=dataset_path, precision=torch.float16)




train_loader, val_loader, _ = get_data_loaders(dataset, sets_size={'train': 0.75, 'val': 0.25, 'test': 0}, norm_axes=(0,2,3,4))

os.abort()
print("\nStarting dummy training loop...")

model = train_model(model, train_loader=train_loader, val_loader=val_loader, loss_criterion=criterion, optimizer=optimizer, config=config, metrics=metrics)
