
from copy import copy
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

model = SequenceProcessor(chunk_shape=(25, 7, 5, 64), embedding_dim=64, codebook_size=512, use_quantizer=True)
model.chunk_ae = VQVAE(
    in_channels = 25,
    input_spatial=(7, 5, 64),
    embedding_dim=64,
    codebook_size=512,
    use_skip_connections=False,
    num_downsample_stages=3)



load_dotenv()
# Dummy training loop
optimizer = torch.optim.AdamW
criterion = SequenceVQVAELoss(
    recon_loss_type='mse',
    recon_weight=1.0
)
mae = torch.nn.L1Loss

config = {
    'lr': 1e-5,
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
dataset = TorchDataset("/home/lolly/Projects/MIEEG/test/test_output/TEST_SAMPLE_FOLDER", chunk_size=64)


# train_loader, val_loader, _ = get_data_loaders(dataset, sets_size={'train': 0.5, 'val': 0.5}, norm_axes=(0, 1, 5), batch_size=1, norm_params=(29, 69))
train_loader, val_loader, _ = get_data_loaders(dataset, sets_size={'train': 0.5, 'val': 0.5}, batch_size=1)

print("\nStarting dummy training loop...")

from packages.train.testing import autoencoder_test_plots

#model = train_model(model, train_loader=train_loader, val_loader=val_loader, loss_criterion=criterion, optimizer=optimizer, config=config, metrics=metrics)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = next(iter(train_loader)).to(device)
x = x[0, 0, ...].unsqueeze(0).unsqueeze(0)
print(f"x shape: {x.shape}")
# normalize x to [0,1] using min/max computed over dims 2,3,4 (per-sample, per-channel)
#eps = 1e-8
#min_vals = x.amin(dim=(2, 3, 4), keepdim=True)
#max_vals = x.amax(dim=(2, 3, 4), keepdim=True)
#x = (x - min_vals) / (max_vals - min_vals + eps)

#criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for i in range(1000):
    out = model(x)
    #out = out[0]
    loss = criterion(out, x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 50 == 0:
        print(i, loss.item())
from packages.plotting.reconstruction_plots import plot_reconstruction_slices
import numpy as np

out = model(x)

rec = out[0]
rec = rec.detach().cpu().numpy()
orig = x.detach().cpu().numpy()

plot_reconstruction_slices(orig, rec, n_channels=6)

#autoencoder_test_plots(model, val_loader, nsamples=5)