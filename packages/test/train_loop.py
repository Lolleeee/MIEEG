
from packages.plotting.reconstruction_plots import plot_reconstruction_distribution
from packages.train.training import train_model
from packages.train.loss import VaeLoss
from packages.train.helpers import BackupManager, EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from packages.io.input_loader import get_data_loaders
import torch
import os
from packages.data_objects.dataset import Dataset, CustomTestDataset
from dotenv import load_dotenv
# model = Conv3DAE(in_channels=25, embedding_dim=128, hidden_dims=[64, 128, 256], use_convnext=False)
from packages.models.autoencoder import Conv3DAE
model = Conv3DAE(in_channels=25, embedding_dim=128)
print(model)

load_dotenv()
dataset_path = "/media/lolly/Bruh/WAYEEGGAL_dataset/WAYEEG_autoencoder_subset"#os.getenv("DATASET_PATH")
# Dummy training loop
optimizer = torch.optim.AdamW
criterion = torch.nn.MSELoss()# VaeLoss(beta=4)
mae = torch.nn.L1Loss

config = {
    'batch_size': 7,
    'lr': 1e-3,
    'epochs': 100,
    'weight_decay': 1e-4,
    'backup_interval': 10,
    'EarlyStopping' : {'patience': 100, 'min_delta': 0.0},
    'BackupManager': {'backup_interval': 10, 'backup_path': './model_backups'},
    'ReduceLROnPlateau': {'patience': 15, 'factor': 0.1},
    'history_plot': {'plot_type': 'extended', 'save_path': './training_history'}
}

metrics = {}

dataset = CustomTestDataset(root_folder=dataset_path, unpack_func='dict', nsamples=10)


train_loader, val_loader, _ = get_data_loaders(dataset, sets_size={'train': 0.7, 'val': 0.3, 'test': 0})

print("\nStarting dummy training loop...")
model.train()
train_model(model, train_loader=train_loader, val_loader=val_loader, loss_criterion=criterion, optimizer=optimizer, config=config, metrics=metrics)

