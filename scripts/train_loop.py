
from packages.io.torch_dataloaders import get_data_loaders
import torch
from packages.data_objects.dataset import TorchDataset, TestTorchDataset
from dotenv import load_dotenv
from packages.models.vqae_23 import VQAE, VQAE23Config
import sys
from packages.train.loss import VQAE23Loss
from dataclasses import dataclass

from packages.train.seed import _set_seed
_set_seed(seed=42)
@dataclass
class Config(VQAE23Config):
    """Configuration for the VQ-VAE model."""
    use_quantizer: bool = False  # Whether to use vector quantization
    # Data shape parameters
    num_freq_bands: int = 25          # F: Number of frequency bands
    spatial_rows: int = 7              # R: Spatial grid rows
    spatial_cols: int = 5              # C: Spatial grid cols
    time_samples: int = 250            # T: Time samples per clip
    chunk_dim: int = 25                # ChunkDim: Time chunk length
    orig_channels: int = 32            # Original EEG channels (R*C or separate)
    
    # Encoder parameters
    encoder_2d_channels: list = None   # [32, 64] - 2D conv channels
    encoder_3d_channels: list = None   # [64, 128, 256] - 3D conv channels
    embedding_dim: int = 128           # Final embedding dimension
    
    # VQ parameters
    codebook_size: int = 512           # Number of codebook vectors
    commitment_cost: float = 0.25      # Beta for commitment loss
    ema_decay: float = 0.99            # EMA decay for codebook updates
    epsilon: float = 1e-5              # Small constant for numerical stability
    
    # Decoder parameters
    decoder_channels: list = None


model = VQAE(Config())

dataset = TorchDataset(root_folder="/media/lolly/SSD/WAYEEGGAL_dataset/0.69subset_250_eeg_wav")
train_loader, val_loader, test_loader = get_data_loaders(dataset, sets_size={'train': 0.6, 'val': 0.2}, batch_size=32, norm_axes=(0,4), target_norm_axes=(0, 2))

sys.exit(0)
load_dotenv()

criterion = VQAE23Loss(
    recon_loss_type='mse',
    recon_weight=1,
    perceptual_weight=0.2,
    bottleneck_cov_weight=1,
    bottleneck_var_weight=2,
    min_variance=0.15
)
criterion.disable_validation()
metrics = {}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = next(iter(train_loader))


# Correct training loop
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []

for i in range(300):
    optimizer.zero_grad()
    
    # Forward pass
    out = model(x['input'])  # Get both outputs
    loss = criterion(out, x)
    
    # Backward pass
    loss['loss'].backward()
    
    # Clip gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Update
    optimizer.step()
    
    if i % 50 == 0:
        print(f"{i}: loss={loss['loss'].item():.4f}")
    losses.append(loss['loss'].item())

print(x['target'].shape)
print(out['reconstruction'].shape)

import matplotlib.pyplot as plt

# Number of samples to plot
N = 4

fig, axes = plt.subplots(N, 1, figsize=(12, 3*N))
if N == 1:
    axes = [axes]

for i in range(N):
    axes[i].plot(x['target'][0, i,...].detach().cpu().numpy(), label='Target', alpha=0.7)
    axes[i].plot(out['reconstruction'][0, i,...].detach().cpu().numpy(), label='Reconstruction', alpha=0.7)
    axes[i].set_title(f'Sample {i+1}')
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Amplitude')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()