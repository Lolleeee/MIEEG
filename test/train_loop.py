
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

model = SequenceProcessor(chunk_shape=(25, 7, 5, 32), embedding_dim=64, codebook_size=1024, use_quantizer=False)
model.chunk_ae = VQVAESkip(
    in_channels=25,
    input_spatial=(7, 5, 32),
    embedding_dim=64,
    codebook_size=512,
    num_downsample_stages=3,
    use_quantizer=False,
    use_skip_connections=False,
    skip_strength=0.1,
    commitment_cost=0.5,
    decay=0.9999
)
model_dict = torch.load('model_backups/model_epoch_10.pt', map_location='cpu')
model.load_state_dict(model_dict, strict=True)
sys.exit(0)
load_dotenv()
# Dummy training loop
optimizer = torch.optim.AdamW
criterion = SequenceVQVAELoss(
    recon_loss_type='perceptual',
    recon_weight=1,
    perceptual_weight=0.2,
    bottleneck_cov_weight=1,
    bottleneck_var_weight=2,
    min_variance=0.15
)

metrics = {}
    
# dataset = CustomTestDataset(root_folder=dataset_path, nsamples=10)
dataset = TorchDataset("test/test_output/", chunk_size=32)

train_loader, val_loader, _ = get_data_loaders(dataset, sets_size={'train': 0.01, 'val': 0.3}, batch_size=32)

config = {
    'lr': 1e-3,
    'epochs': 300,
    #'EarlyStopping' : {'patience': 20, 'min_delta': 0.01},
    'BackupManager': {'backup_interval': 10, 'backup_path': './model_backups'},
    'ReduceLROnPlateau': {'mode': 'min', 'patience': 40, 'factor': 0.0},
    'history_plot': {'plot_type': 'extended', 'save_path': './training_history'},
    'grad_clip': 1.0
}
train_model(model, train_loader=train_loader, val_loader=val_loader, loss_criterion=criterion, optimizer=optimizer, config=config, metrics={})
sys.exit(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = next(iter(train_loader)).to(device)
print(f"x shape: {x.shape}")

# Correct training loop
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []
print(f"Current quantizer bool: {model.chunk_ae.use_quantizer}")
for i in range(500):
    if i == 2510000:
        model.chunk_ae.use_quantizer = False
        print(f"Quantizer enabled at iteration {i}, Current quantizer bool: {model.chunk_ae.use_quantizer}")
    optimizer.zero_grad()
    
    # Forward pass
    out = model(x)  # Get both outputs
    loss = criterion(out, x)
    
    # Backward pass
    loss.backward()
    
    # Clip gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Update
    optimizer.step()
    
    if i % 50 == 0:
        print(f"{i}: loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")
        print(criterion.last_losses)
    losses.append(loss.item())

        

import matplotlib.pyplot as plt
plt.figure()
plt.plot(losses, marker='.', linewidth=1, label='train_loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.legend()
plt.show()
    
from packages.plotting.reconstruction_plots import plot_reconstruction_slices, plot_reconstruction_performance
out = model(x)

rec = out[0]
rec = rec.detach().cpu().numpy()
orig = x.detach().cpu().numpy()
plot_reconstruction_slices(orig[0, 0,...], rec[0,0,...], n_channels=6)
plot_reconstruction_performance(orig[0, 0,...], rec[0,0,...])
#autoencoder_test_plots(model, val_loader, nsamples=5)