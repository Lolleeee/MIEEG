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

    def _compute_loss(self, outputs: Dict, inputs: Dict) -> Dict:
        rec = outputs['reconstruction']
        target = inputs['target']
        loss = self.function(rec, target)

        return {'loss': loss}

class TorchL1Loss(TorchLoss):
    """SmoothL1 (Huber-like) loss with optional embedding regularization."""

    def __init__(self, emb_loss=0.0, scale=1.0, beta=1.0, reduction="mean"):
        super().__init__(
            expected_model_output_keys=["reconstruction", "embeddings"],
            expected_loss_keys=["loss"],
        )
        self.emb_loss = emb_loss
        self.scale = scale
        self.beta = beta
        self.reduction = reduction

    def _compute_loss(self, outputs: Dict, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        reconstruction = outputs["reconstruction"]
        target = inputs["target"]

        # SmoothL1 uses squared error when |err| < beta and L1 otherwise. [web:227]
        loss = F.smooth_l1_loss(
            reconstruction, target, beta=self.beta, reduction=self.reduction
        )  # [web:227]

        embedding = outputs.get("embeddings", None)
        if self.emb_loss > 0 and embedding is not None:
            l1_norm = torch.mean(torch.abs(embedding))
            loss = loss + self.emb_loss * l1_norm

        return {"loss": loss * self.scale}


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
    Fixed version with absolute magnitude calibration and running average auto-balancing.
    """
    def __init__(
        self,
        mse_weight: float = 1.0,
        magnitude_weight: float = 1.0,
        phase_weight: float = 0.3,
        power_weight: float = 0.5,
        bottleneck_var_weight: float = 0.1,
        bottleneck_cov_weight: float = 0.1,
        fs: int = 160,
        n_fft: int = 512,
        phase_magnitude_threshold: float = 0.01,
        auto_balance: bool = True,
        warmup_batches: int = 100,
        momentum: float = 0.9
    ):
        super().__init__(
            expected_model_output_keys=['reconstruction', 'embeddings', 'vq_loss'], 
            expected_loss_keys=[
                'loss', 'mse_loss', 'magnitude_loss', 'phase_loss', 
                'power_loss', 'vq_loss', 'bottleneck_loss'
            ]
        )
        self.name = "OptimalVQAELoss"
        
        self.mse_weight = mse_weight
        self.magnitude_weight = magnitude_weight
        self.phase_weight = phase_weight
        self.power_weight = power_weight
        self.bottleneck_var_weight = bottleneck_var_weight
        self.bottleneck_cov_weight = bottleneck_cov_weight
        
        self.fs = fs
        self.n_fft = n_fft
        self.phase_magnitude_threshold = phase_magnitude_threshold
        self.auto_balance = auto_balance
        self.warmup_batches = warmup_batches
        self.momentum = momentum
        
        # Running average of loss scales - will be moved to device on first use
        self.register_buffer('loss_scale_mse', torch.tensor(1.0))
        self.register_buffer('loss_scale_magnitude', torch.tensor(1.0))
        self.register_buffer('loss_scale_phase', torch.tensor(1.0))
        self.register_buffer('loss_scale_power', torch.tensor(1.0))
        self.register_buffer('num_batches_seen', torch.tensor(0, dtype=torch.long))
        
        self.warmup_complete = False

    def _magnitude_loss(self, recon, target):
        """Relative spectral shape (scale-invariant)."""
        recon_fft = torch.fft.rfft(recon, n=self.n_fft, dim=-1)
        target_fft = torch.fft.rfft(target, n=self.n_fft, dim=-1)
        
        recon_psd = torch.abs(recon_fft) ** 2
        target_psd = torch.abs(target_fft) ** 2
        
        recon_psd_norm = recon_psd / (recon_psd.sum(dim=-1, keepdim=True) + 1e-8)
        target_psd_norm = target_psd / (target_psd.sum(dim=-1, keepdim=True) + 1e-8)
        
        recon_log = torch.log(recon_psd_norm + 1e-10)
        target_log = torch.log(target_psd_norm + 1e-10)
        
        return F.l1_loss(recon_log, target_log)

    def _power_loss(self, recon, target):
        """Absolute power/energy constraint."""
        recon_fft = torch.fft.rfft(recon, n=self.n_fft, dim=-1)
        target_fft = torch.fft.rfft(target, n=self.n_fft, dim=-1)
        
        recon_power = torch.sum(torch.abs(recon_fft) ** 2, dim=-1)
        target_power = torch.sum(torch.abs(target_fft) ** 2, dim=-1)
        
        return F.l1_loss(torch.log(recon_power + 1e-8), 
                        torch.log(target_power + 1e-8))

    def _phase_loss(self, recon, target):
        """Phase coherence."""
        recon_fft = torch.fft.rfft(recon, n=self.n_fft, dim=-1)
        target_fft = torch.fft.rfft(target, n=self.n_fft, dim=-1)
        
        target_psd = torch.abs(target_fft) ** 2
        target_power_total = target_psd.sum(dim=-1, keepdim=True) + 1e-8
        target_psd_norm = target_psd / target_power_total
        phase_mask = (target_psd_norm > self.phase_magnitude_threshold).float()
        
        recon_mag = torch.abs(recon_fft) + 1e-8
        target_mag = torch.abs(target_fft) + 1e-8
        
        complex_corr = torch.real(
            (recon_fft * torch.conj(target_fft)) / (recon_mag * target_mag)
        )
        
        phase_error = (1.0 - complex_corr) * phase_mask
        
        return phase_error.sum() / (phase_mask.sum() + 1e-8)

    def _bottleneck_regularization(self, embeddings):
        """Combined variance + decorrelation."""
        if embeddings is None or embeddings.size(0) <= 1:
            return torch.tensor(0.0, device=embeddings.device if embeddings is not None else 'cpu')
        
        std = torch.sqrt(embeddings.var(dim=0, unbiased=False) + 1e-4)
        var_loss = torch.mean(torch.relu(1.0 - std))
        
        z_mean = embeddings.mean(dim=0, keepdim=True)
        z_centered = embeddings - z_mean
        z_std = z_centered.std(dim=0, unbiased=False, keepdim=True) + 1e-8
        z_normalized = z_centered / z_std
        
        correlation = torch.mm(z_normalized.t(), z_normalized) / embeddings.size(0)
        identity = torch.eye(embeddings.size(1), device=embeddings.device)
        decorr_loss = torch.mean((correlation - identity) ** 2)
        
        return self.bottleneck_var_weight * var_loss + self.bottleneck_cov_weight * decorr_loss

    def _update_running_scales(self, mse_loss, magnitude_loss, phase_loss, power_loss):
        """
        Update running averages of loss scales using exponential moving average.
        Ensures all tensors are on the same device.
        """
        # Get scalar values and ensure they're on the right device
        mse_val = mse_loss.detach()
        mag_val = magnitude_loss.detach()
        phase_val = phase_loss.detach()
        power_val = power_loss.detach()
        
        # Move buffers to same device as losses (on first call)
        if self.loss_scale_mse.device != mse_val.device:
            self.loss_scale_mse = self.loss_scale_mse.to(mse_val.device)
            self.loss_scale_magnitude = self.loss_scale_magnitude.to(mse_val.device)
            self.loss_scale_phase = self.loss_scale_phase.to(mse_val.device)
            self.loss_scale_power = self.loss_scale_power.to(mse_val.device)
            self.num_batches_seen = self.num_batches_seen.to(mse_val.device)
        
        if self.num_batches_seen == 0:
            # First batch: initialize with current values
            self.loss_scale_mse.copy_(mse_val)
            self.loss_scale_magnitude.copy_(mag_val)
            self.loss_scale_phase.copy_(phase_val)
            self.loss_scale_power.copy_(power_val)
        else:
            # Exponential moving average
            self.loss_scale_mse.mul_(self.momentum).add_(mse_val, alpha=1 - self.momentum)
            self.loss_scale_magnitude.mul_(self.momentum).add_(mag_val, alpha=1 - self.momentum)
            self.loss_scale_phase.mul_(self.momentum).add_(phase_val, alpha=1 - self.momentum)
            self.loss_scale_power.mul_(self.momentum).add_(power_val, alpha=1 - self.momentum)
        
        self.num_batches_seen += 1
        
        # Mark warmup as complete
        if self.num_batches_seen >= self.warmup_batches and not self.warmup_complete:
            self.warmup_complete = True
            print("=" * 70)
            print(f"Loss scale warmup complete after {self.warmup_batches} batches!")
            print(f"Final running averages:")
            print(f"  MSE scale:       {self.loss_scale_mse.item():.4f}")
            print(f"  Magnitude scale: {self.loss_scale_magnitude.item():.4f}")
            print(f"  Phase scale:     {self.loss_scale_phase.item():.4f}")
            print(f"  Power scale:     {self.loss_scale_power.item():.4f}")
            print(f"\nEffective weights (weight / scale):")
            print(f"  MSE:       {self.mse_weight / self.loss_scale_mse.item():.4f}")
            print(f"  Magnitude: {self.magnitude_weight / self.loss_scale_magnitude.item():.4f}")
            print(f"  Phase:     {self.phase_weight / self.loss_scale_phase.item():.4f}")
            print(f"  Power:     {self.power_weight / self.loss_scale_power.item():.4f}")
            print("=" * 70)

    def _compute_loss(self, outputs: dict, batch: dict) -> dict:
        """Main loss computation with running average auto-balancing."""
        target = batch['target']
        recon = outputs['reconstruction']
        embeddings = outputs.get('embeddings', None)
        vq_loss = outputs.get('vq_loss', torch.tensor(0.0, device=target.device))
        
        # Compute individual losses (raw values)
        mse_loss = F.mse_loss(recon, target)
        magnitude_loss = self._magnitude_loss(recon, target)
        power_loss = self._power_loss(recon, target)
        phase_loss = self._phase_loss(recon, target)
        
        # Update running averages during warmup (training only)
        if self.auto_balance and self.training and self.num_batches_seen < self.warmup_batches:
            self._update_running_scales(mse_loss, magnitude_loss, phase_loss, power_loss)
        
        # Apply scale normalization + user weights
        if self.auto_balance:
            # Use running averages for normalization
            eps = 1e-6
            mse_term = (self.mse_weight / (self.loss_scale_mse + eps)) * mse_loss
            mag_term = (self.magnitude_weight / (self.loss_scale_magnitude + eps)) * magnitude_loss
            phase_term = (self.phase_weight / (self.loss_scale_phase + eps)) * phase_loss
            power_term = (self.power_weight / (self.loss_scale_power + eps)) * power_loss
        else:
            # No auto-balancing, use raw weights
            mse_term = self.mse_weight * mse_loss
            mag_term = self.magnitude_weight * magnitude_loss
            phase_term = self.phase_weight * phase_loss
            power_term = self.power_weight * power_loss
        
        # Bottleneck regularization
        if embeddings is not None and embeddings.dim() == 3:
            embeddings = embeddings.reshape(-1, embeddings.size(-1))
        
        bottleneck_loss = self._bottleneck_regularization(embeddings)
        
        # Total loss
        total_loss = (
            mse_term +
            mag_term +
            power_term +
            phase_term +
            vq_loss +
            bottleneck_loss
        )
        
        return {
            'loss': total_loss,
            'mse_loss': mse_loss,
            'magnitude_loss': magnitude_loss,
            'power_loss': power_loss,
            'phase_loss': phase_loss,
            'vq_loss': vq_loss,
            'bottleneck_loss': bottleneck_loss
        }

    

class CWTMSE(TorchLoss):
    def __init__(self):
        """MSE Loss with automatic validation"""
        super().__init__(expected_model_output_keys=['reconstruction', 'target'], 
                         expected_loss_keys=['loss'])
        self.name = "CustomMSE"
        self.function = nn.MSELoss(reduction='mean')
    def _compute_loss(self, outputs: Dict, inputs: torch.Tensor) -> torch.Tensor:
        rec = outputs['reconstruction']
        target = outputs['target']
        loss = self.function(rec, target)

        return {'loss': loss}


# WARNING: DOESN?T WORK IF HEAD HAS LEARNABLE PARAMETERS
from packages.models.wavelet_head import CWTHead
class CWTLoss(TorchLoss):
    def __init__(self):
        """Continuous Wavelet Transform Loss
        Uses CWT head to compute loss in time-frequency domain.
        """
        super().__init__(expected_model_output_keys=['reconstruction'], 
                         expected_loss_keys=['loss'])
        self.name = "CWTLoss"
        self.function = nn.MSELoss(reduction='mean')
        self.cwt_head = CWTHead(fs=160, frequencies=np.logspace(np.log10(0.5), np.log10(79.9), 25))
    def _compute_loss(self, outputs: dict, batch: dict) -> dict:
        rec = outputs['reconstruction']
        target = batch['target']
        
        rec_cwt = self.cwt_head(rec)
        target_cwt = self.cwt_head(target)
        
        loss = self.function(rec_cwt, target_cwt)

        return {'loss': loss}