
from packages.models.autoencoder import basicConv3DAE
from packages.plotting.reconstruction_plots import plot_reconstruction_distribution
from packages.train.training import train_model
from packages.train.loss import VaeLoss, CustomMSELoss, PerceptualLoss
from packages.train.helpers import BackupManager, EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from packages.io.file_loader import get_data_loaders
import torch
import os
from packages.data_objects.dataset import TorchDataset, CustomTestDataset
from dotenv import load_dotenv
# model = Conv3DAE(in_channels=25, embedding_dim=128, hidden_dims=[64, 128, 256], use_convnext=False)

from packages.models.vqae import VQVAE, SequenceProcessor
from packages.train.loss import VQVAELoss, SequenceVQVAELoss

model = SequenceProcessor(chunk_shape=(50, 7, 5, 25), use_quantizer=False)





load_dotenv()
dataset_path = "/media/lolly/Bruh/WAYEEGGAL_dataset/0.5subset_datanooverlap"
# Dummy training loop
optimizer = torch.optim.AdamW
criterion = SequenceVQVAELoss(
    recon_loss_type='mse',
    recon_weight=1.0
)
mae = torch.nn.L1Loss

config = {
    'batch_size': 2,
    'lr': 1e-3,
    'epochs': 100,
    # 'EarlyStopping' : {'patience': 10000, 'min_delta': 0.0},
    # 'BackupManager': {'backup_interval': 100000, 'backup_path': './model_backups'},
    # 'ReduceLROnPlateau': {'patience': 25, 'factor': 0.5},
    'history_plot': {'plot_type': 'extended', 'save_path': './training_history'},
    'grad_clip': 1.0,
    'use_amp': False,
    'grad_logging_interval': None   
}

metrics = {}
    
# dataset = CustomTestDataset(root_folder=dataset_path, nsamples=10)
dataset = TorchDataset("/home/lolly/Desktop/test", chunk_size=25)

train_loader, val_loader, _ = get_data_loaders(dataset, sets_size={'train': 0.01, 'val': 0.1}, batch_size=32)
print(next(iter(train_loader)).shape)
print("\nStarting dummy training loop...")
print(model)

from packages.plotting.tensor_plots import plot_dimension_distribuitions
#plot_dimension_distribuitions(next(iter(train_loader))[0, 0, ...])
model = train_model(model, train_loader=train_loader, val_loader=val_loader, loss_criterion=criterion, optimizer=optimizer, config=config, metrics=metrics)
