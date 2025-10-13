
from packages.plotting.reconstruction_plots import plot_reconstruction_distribution
from packages.train.training import train_model
from packages.train.loss import VaeLoss
from packages.train.helpers import BackupManager, EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from packages.io.file_loader import get_data_loaders
import torch
import os
from packages.data_objects.dataset import Dataset, CustomTestDataset
from dotenv import load_dotenv
# model = Conv3DAE(in_channels=25, embedding_dim=128, hidden_dims=[64, 128, 256], use_convnext=False)
from packages.models.variational_autoencoder_convnext import Conv3DVAE
model = Conv3DVAE(in_channels=25, latent_dim=128, hidden_dims=[64, 128, 256], use_convnext=True)
print(model)

load_dotenv()
dataset_path = "/media/lolly/Bruh/WAYEEGGAL_dataset/WAYEEG_processed"
# Dummy training loop
optimizer = torch.optim.AdamW
criterion =  VaeLoss(beta=1)
mae = torch.nn.L1Loss

config = {
    'batch_size': 1,
    'lr': 1e-4,
    'epochs': 100,
    # 'EarlyStopping' : {'patience': 10000, 'min_delta': 0.0},
    # 'BackupManager': {'backup_interval': 100000, 'backup_path': './model_backups'},
    # 'ReduceLROnPlateau': {'patience': 25, 'factor': 0.5},
    'history_plot': {'plot_type': 'extended', 'save_path': './training_history'}
}

metrics = {}

dataset = CustomTestDataset(root_folder=dataset_path, file_type='npz', unpack_func='dict', nsamples=2)


train_loader, val_loader, _ = get_data_loaders(dataset, sets_size={'train': 0.5, 'val': 0.5, 'test': 0})

print("\nStarting dummy training loop...")
model.train()
model = train_model(model, train_loader=train_loader, val_loader=val_loader, loss_criterion=criterion, optimizer=optimizer, config=config, metrics=metrics)

# TODO add mse to VAE metrics to compare with AE