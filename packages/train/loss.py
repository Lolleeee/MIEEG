import sys
import torch 
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict, Union, Tuple, Any, List
from abc import abstractmethod
import logging
from packages.train.F_torch_wrappers import TorchLoss


logging.basicConfig(level=logging.INFO)


class TorchMSELoss(TorchLoss):
    def __init__(self):
        """MSE Loss with automatic validation"""
        super().__init__(expected_model_output_keys=['reconstruction'], 
                         expected_loss_keys=['loss'])
        self.name = "TorchMSELoss"
        self.function = nn.MSELoss(reduction='mean')
    def _compute_loss(self, outputs: Dict, inputs: torch.Tensor) -> torch.Tensor:
        rec = outputs['reconstruction']
        loss = self.function(rec, inputs)
        return {'loss': loss}

class TorchL1Loss(TorchLoss):
    """L1 Loss with optional embedding regularization"""
    
    def __init__(self, emb_loss=0, scale=1.0):
        super().__init__(expected_model_output_keys=['reconstruction', 'embeddings'], 
        expected_loss_keys=['loss'])
        self.emb_loss = emb_loss
        self.scale = scale
    
    def _compute_loss(self, outputs: Dict, inputs: torch.Tensor) -> torch.Tensor:
        # Get reconstruction
        reconstruction = self._get_main_output(outputs)
        
        # Base L1 loss
        loss = F.l1_loss(reconstruction, inputs)

        embedding = outputs['embeddings']
        # Optional embedding regularization
        if self.emb_loss > 0 and embedding is not None:
            L1_norm = torch.mean(torch.abs(embedding))
            loss += self.emb_loss * L1_norm
        
        return {'loss': loss * self.scale}


class SequenceVQVAELoss(TorchLoss):
    """
    Loss function for SequenceProcessor with bottleneck regularization.
    """
    def __init__(
        self,
        recon_loss_type='mse',
        recon_weight=1.0,
        perceptual_weight=0.1,
        bottleneck_var_weight=1.0,      # NEW: Force variance in embeddings
        bottleneck_cov_weight=0.5,      # NEW: Force decorrelation
        min_variance=0.1                # NEW: Minimum per-dimension variance
    ):
        super().__init__(expected_model_output_keys=['reconstruction', 'embeddings', 'vq_loss'], 
                         expected_loss_keys=['loss', 'recon_loss', 'vq_loss', 'bottleneck_loss', 'bottleneck_var_loss', 'bottleneck_cov_loss'])
        self.name = "SequenceVQVAELoss"
        self.recon_loss_type = recon_loss_type
        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
        self.bottleneck_var_weight = bottleneck_var_weight
        self.bottleneck_cov_weight = bottleneck_cov_weight
        self.min_variance = min_variance
        self.last_losses = None

    def _perceptual_loss(self, x, x_recon):
        """Perceptual loss for sequence of chunks"""
        x_flat = x.view(-1, *x.shape[2:])
        x_recon_flat = x_recon.view(-1, *x_recon.shape[2:])
        
        def compute_gradients(tensor):
            dx = tensor[:, :, 1:, :, :] - tensor[:, :, :-1, :, :]
            dy = tensor[:, :, :, 1:, :] - tensor[:, :, :, :-1, :]
            dt = tensor[:, :, :, :, 1:] - tensor[:, :, :, :, :-1]
            return dx, dy, dt
        
        x_grads = compute_gradients(x_flat)
        recon_grads = compute_gradients(x_recon_flat)
        
        loss = sum(F.l1_loss(g1, g2) for g1, g2 in zip(x_grads, recon_grads))
        return loss / 3.0

    def _bottleneck_variance_loss(self, embeddings):
        """
        Penalize low variance in embeddings to prevent collapse.
        
        Args:
            embeddings: (B * num_chunks, embedding_dim) flattened embeddings
        """
        # Compute per-dimension variance
        z_mean = embeddings.mean(dim=0)
        z_var = ((embeddings - z_mean) ** 2).mean(dim=0)
        
        # Penalize variance below threshold
        var_loss = torch.mean(torch.relu(self.min_variance - z_var))
        
        return var_loss
    
    def _bottleneck_decorrelation_loss(self, embeddings):
        """
        Penalize correlation between embedding dimensions.
        Forces dimensions to be independent.
        
        Args:
            embeddings: (B * num_chunks, embedding_dim) flattened embeddings
        """
        # Center the embeddings
        z_mean = embeddings.mean(dim=0)
        z_centered = embeddings - z_mean
        
        # Compute correlation matrix
        z_std = z_centered.std(dim=0, keepdim=True) + 1e-8
        z_normalized = z_centered / z_std
        
        correlation = torch.mm(z_normalized.t(), z_normalized) / embeddings.size(0)
        
        # Penalize off-diagonal correlations
        identity = torch.eye(embeddings.size(1), device=embeddings.device)
        decorr_loss = torch.mean((correlation - identity) ** 2)
        
        return decorr_loss

    def _compute_loss(self, outputs, chunks):
        """
        Compute total loss for sequence.
        
        Args:
            outputs: (chunks_recon, vq_loss, indices) from model
            chunks: Original chunks (B, num_chunks, C, H, W, T)
            embeddings: (B, num_chunks, embedding_dim) - REQUIRED for bottleneck reg
            
        Returns:
            loss: Total loss (scalar)
            loss_dict: Dictionary of individual loss components
        """
        chunks_recon = outputs['reconstruction']
        embeddings = outputs['embeddings']
        vq_loss = outputs['vq_loss']
        
        # Reconstruction loss
        if self.recon_loss_type == 'mse':
            recon_loss = F.mse_loss(chunks_recon, chunks)
        elif self.recon_loss_type == 'l1':
            recon_loss = F.l1_loss(chunks_recon, chunks)
        elif self.recon_loss_type == 'perceptual':
            recon_loss = F.mse_loss(chunks_recon, chunks)
            perc_loss = self._perceptual_loss(chunks, chunks_recon)
            recon_loss = recon_loss + self.perceptual_weight * perc_loss
        else:
            raise ValueError(f"Unknown recon_loss_type: {self.recon_loss_type}")
        
        # Bottleneck regularization (if embeddings provided)
        bottleneck_loss = 0.0
        if embeddings is not None:
            # Flatten embeddings: (B, num_chunks, D) -> (B*num_chunks, D)
            embeddings_flat = embeddings.view(-1, embeddings.size(-1))
            
            # Variance loss: prevent dimensions from collapsing to zero
            var_loss = self._bottleneck_variance_loss(embeddings_flat)
            
            # Decorrelation loss: prevent dimensions from being correlated
            decorr_loss = self._bottleneck_decorrelation_loss(embeddings_flat)
            
            bottleneck_loss = (
                self.bottleneck_var_weight * var_loss +
                self.bottleneck_cov_weight * decorr_loss
            )
        
        # Total loss
        total_loss = (
            self.recon_weight * recon_loss + 
            vq_loss + 
            bottleneck_loss
        )
        
        return {'loss': total_loss, 'recon_loss': recon_loss, 'vq_loss': vq_loss, 'bottleneck_loss': bottleneck_loss, 'bottleneck_var_loss': var_loss, 'bottleneck_cov_loss': decorr_loss}
    


class VQAE23Loss(TorchLoss):
    """
    Enhanced VQ-VAE Loss for Motor EEG.
    Combines:
    1. Time-domain reconstruction (MSE)
    2. Band-weighted Frequency-domain reconstruction (Spectral loss)
    3. VQ commitment/codebook loss
    4. Bottleneck regularization (variance + decorrelation)
    """
    def __init__(
        self,
        recon_weight: float = 1.0,
        freq_weight: float = 0.5,       # Weight for frequency loss
        bottleneck_var_weight: float = 0.1,
        bottleneck_cov_weight: float = 0.1,
        fs: int = 160,
        time_samples: int = 80,
        motor_bands: dict = None
    ):
        super().__init__(
            expected_model_output_keys=['reconstruction', 'embeddings', 'vq_loss'], 
            expected_loss_keys=[
                'loss', 'recon_loss', 'freq_loss', 'vq_loss', 
                'bottleneck_loss', 'bottleneck_var_loss', 'bottleneck_cov_loss'
            ]
        )
        self.name = "VQAE23Loss"
        self.recon_weight = recon_weight
        self.freq_weight = freq_weight
        self.bottleneck_var_weight = bottleneck_var_weight
        self.bottleneck_cov_weight = bottleneck_cov_weight
        
        # Frequency configuration
        self.fs = fs
        self.n_fft = time_samples
        freq_res = fs / time_samples
        
        # Motor-relevant bands (Hz)
        # Weight alpha (8-13) and beta (13-30) higher for motor tasks
        self.motor_bands = motor_bands or {
            'delta': (1, 4, 0.5),
            'theta': (4, 8, 0.8),
            'alpha': (8, 13, 2.0),   # Critical for motor
            'beta': (13, 30, 2.0),   # Critical for motor
            'gamma': (30, 45, 1.0)
        }
        
        # Precompute FFT bins for each band
        self.band_bins = {}
        for band, (low, high, weight) in self.motor_bands.items():
            low_bin = int(low / freq_res)
            high_bin = int(high / freq_res)
            # Ensure valid indices
            if high_bin > (time_samples // 2 + 1):
                high_bin = time_samples // 2 + 1
            if low_bin < high_bin:
                self.band_bins[band] = (low_bin, high_bin, weight)

    def _bottleneck_decorrelation_loss(self, embeddings):
        """Penalize correlation between embedding dimensions."""
        # Only compute if batch size > 1
        if embeddings.size(0) <= 1:
            return torch.tensor(0.0, device=embeddings.device)
        
        z_mean = embeddings.mean(dim=0)
        z_centered = embeddings - z_mean
        
        # Use unbiased=False to avoid division by zero warning
        z_std = z_centered.std(dim=0, unbiased=False, keepdim=True) + 1e-8
        z_normalized = z_centered / z_std
        
        correlation = torch.mm(z_normalized.t(), z_normalized) / embeddings.size(0)
        identity = torch.eye(embeddings.size(1), device=embeddings.device)
        return torch.mean((correlation - identity) ** 2)

    def _bottleneck_variance_loss(self, embeddings, eps=1e-4):
        """Encourage variance in embedding dimensions (prevent collapse)."""
        # Only compute if batch size > 1
        if embeddings.size(0) <= 1:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Use unbiased=False to avoid division by zero warning
        std = torch.sqrt(embeddings.var(dim=0, unbiased=False) + eps)
        
        # Hinge loss: penalty if std < 1
        return torch.mean(torch.relu(1 - std))

    def _frequency_loss(self, recon, target):
        """
        More stable spectral loss: Magnitude difference (not Power) + Log-Cosh.
        Handles single samples gracefully.
        
        Args:
            recon: Reconstructed signal (B, Channels, Time)
            target: Target signal (B, Channels, Time)
            
        Returns:
            Frequency loss (scalar tensor)
        """
        # Handle single sample case - return zero if no bands defined
        if recon.size(0) <= 1 and len(self.band_bins) == 0:
            return torch.tensor(0.0, device=recon.device)
        
        # FFT
        recon_fft = torch.fft.rfft(recon, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        
        # Use MAGNITUDE (amplitude), not Power (amplitude^2)
        # Power ^2 explodes gradients for large values and vanishes for small ones
        recon_mag = torch.abs(recon_fft)
        target_mag = torch.abs(target_fft)
        
        total_freq_loss = 0.0
        total_weight = 0.0
        
        for band, (low, high, weight) in self.band_bins.items():
            # Ensure valid band range
            if low >= high or high > recon_mag.size(-1):
                continue
            
            # Extract band
            recon_band = recon_mag[..., low:high]
            target_band = target_mag[..., low:high]
            
            # Skip if band is empty
            if recon_band.numel() == 0:
                continue
            
            # Log-Cosh loss: behaves like L2 for small errors, L1 for large errors
            # Much smoother gradients than MSE on Log
            diff = recon_band - target_band
            
            # Clamp to avoid overflow in cosh (cosh(x) explodes for x > ~88)
            diff = torch.clamp(diff, -10, 10)
            
            band_loss = torch.log(torch.cosh(diff) + 1e-8).mean()
            
            total_freq_loss += weight * band_loss
            total_weight += weight
        
        # Handle case where no valid bands were processed
        if total_weight < 1e-8:
            return torch.tensor(0.0, device=recon.device)
            
        return total_freq_loss / total_weight

    def _compute_loss(self, outputs: Dict, batch: dict) -> Dict:
        assert 'target' in batch, "Batch must contain 'target' for VQAE23Loss"

        target = batch['target']  # (B, 32, 80)
        recon = outputs['reconstruction']
        embeddings = outputs['embeddings']
        vq_loss = outputs.get('vq_loss', torch.tensor(0., device=target.device))

        # 1. Time-domain Reconstruction
        recon_loss = F.mse_loss(recon, target)
        
        # 2. Frequency-domain Reconstruction (Motor-Weighted)
        freq_loss = self._frequency_loss(recon, target)
          
        # 3. Bottleneck Regularization
        bottleneck_loss = torch.tensor(0., device=target.device)
        var_loss = torch.tensor(0., device=target.device)
        decorr_loss = torch.tensor(0., device=target.device)
        
        if embeddings is not None:
            # Flatten: (B, num_chunks, D) -> (B*num_chunks, D)
            # Or if embeddings are (B, D), just keep as is
            if embeddings.dim() == 3:
                embeddings_flat = embeddings.view(-1, embeddings.size(-1))
            else:
                embeddings_flat = embeddings
            
            var_loss = self._bottleneck_variance_loss(embeddings_flat)
            decorr_loss = self._bottleneck_decorrelation_loss(embeddings_flat)
            
            bottleneck_loss = (
                self.bottleneck_var_weight * var_loss +
                self.bottleneck_cov_weight * decorr_loss
            )
        
        # Total Loss
        total_loss = (
            self.recon_weight * recon_loss + 
            self.freq_weight * freq_loss +
            vq_loss + 
            bottleneck_loss
        )

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'freq_loss': freq_loss,
            'vq_loss': vq_loss,
            'bottleneck_loss': bottleneck_loss,
            'bottleneck_var_loss': var_loss,
            'bottleneck_cov_loss': decorr_loss
        }
