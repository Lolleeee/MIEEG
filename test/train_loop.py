
from copy import copy
import sys
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

from packages.models.vqae import SequenceProcessor, VQVAE
from packages.models.vqae_skip import SkipConnectionScheduler
from packages.models.vqae_skip import VQVAE as VQVAESkip
from packages.train.loss import VQVAELoss, SequenceVQVAELoss

model = SequenceProcessor(chunk_shape=(25, 7, 5, 64), embedding_dim=64, codebook_size=512, use_quantizer=False)
model.chunk_ae = VQVAESkip(
    in_channels=25,
    input_spatial=(7, 5, 64),
    embedding_dim=64,
    codebook_size=512,
    num_downsample_stages=3,
    use_quantizer=False,
    use_skip_connections=True,
    skip_strength=1
)

model.chunk_ae = VQVAE(
    in_channels=25,
    input_spatial=(7, 5, 64),
    embedding_dim=64,
    codebook_size=512,
    num_downsample_stages=3,
    use_quantizer=False,
    use_skip_connections=True,
)


load_dotenv()
# Dummy training loop
optimizer = torch.optim.AdamW
criterion = SequenceVQVAELoss(
    recon_loss_type='mse',
    recon_weight=1.0
)
mae = torch.nn.L1Loss

config = {
    #'weight_decay': 1e-4,
    'epochs': 100,
    #'EarlyStopping' : {'patience': 20, 'min_delta': 0.01},
    #'BackupManager': {'backup_interval': 10, 'backup_path': './model_backups'},
    #'ReduceLROnPlateau': {'mode': 'min', 'patience': 5, 'factor': 0.0},
    'history_plot': {'plot_type': 'extended', 'save_path': './training_history'},
    'grad_clip': 1.0,
    'use_amp': False,
    'grad_logging_interval' : None,
    'asym_lr': None
}

metrics = {}
    
# dataset = CustomTestDataset(root_folder=dataset_path, nsamples=10)
dataset = TorchDataset("test/test_output/TEST_SAMPLE_FOLDER", chunk_size=64)


# train_loader, val_loader, _ = get_data_loaders(dataset, sets_size={'train': 0.5, 'val': 0.5}, norm_axes=(0, 1, 5), batch_size=1, norm_params=(29, 69))
train_loader, val_loader, _ = get_data_loaders(dataset, sets_size={'train': 0.5, 'val': 0.5}, batch_size=1)

print("\nStarting dummy training loop...")

from packages.train.testing import autoencoder_test_plots

model = train_model(model, train_loader=train_loader, val_loader=val_loader, loss_criterion=criterion, optimizer=optimizer, config=config, metrics=metrics)
sys.exit(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = next(iter(train_loader)).to(device)
x = x[0, ...].unsqueeze(0) 
print(f"x shape: {x.shape}")

losses = []
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for i in range(500):
    out = model(x)
    #out = out[0]
    loss = criterion(out, x)
    optimizer.zero_grad()
    loss.backward()
    if i % 50 == 0:
        print(i, loss.item())

    optimizer.step()

    losses.append(loss.item())

    # plot and save loss curve
import matplotlib.pyplot as plt
plt.figure()
plt.plot(losses, marker='.', linewidth=1, label='train_loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.legend()
plt.show()
    
from packages.plotting.reconstruction_plots import plot_reconstruction_slices
out = model(x)

rec = out[0]
rec = rec.detach().cpu().numpy()
orig = x.detach().cpu().numpy()

plot_reconstruction_slices(orig, rec, n_channels=6)

#autoencoder_test_plots(model, val_loader, nsamples=5)