from packages.models.autoencoder import Conv3DAutoencoder
from packages.train.training import train_model
from packages.io.input_loader import get_data_loaders
import torch
import os
from packages.data_objects.dataset import Dataset
from dotenv import load_dotenv
model = Conv3DAutoencoder(in_channels=50, embedding_dim=256)
load_dotenv()
dataset_path = os.getenv("DATASET_PATH")
# Dummy training loop
optimizer = torch.optim.AdamW
criterion = torch.nn.MSELoss
mae = torch.nn.L1Loss

config = {
    'batch_size': 32,
    'lr': 1e-3,
    'epochs': 5,
    'backup_interval': 10,
    'EarlyStopping' : {'patience': 1, 'min_delta': 0.1},
    'BackupManager': {'backup_interval': 10, 'backup_path': './model_backups'},
    'ReduceLROnPlateau': {'mode': 'min', 'patience': 1, 'factor': 0.1},
    'history_plot': {'plot_type': 'extended', 'save_path': './training_history'}
}

metrics = {'MAE': mae}

dataset = Dataset.get_test_dataset(root_folder=dataset_path, unpack_func='dict', nsamples=4)

train_loader, val_loader, _ = get_data_loaders(dataset, sets_size={'train': 0.5, 'val': 0.5, 'test': 0})

print("\nStarting dummy training loop...")
model.train()
train_model(model, train_loader=train_loader, val_loader=val_loader, loss_criterion=criterion, optimizer=optimizer, config=config, metrics=metrics)