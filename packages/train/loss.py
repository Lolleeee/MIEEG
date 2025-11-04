import sys
import torch 
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from typing import NamedTuple

class TorchLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name: str = "BaseLoss"

    def forward(self, outputs, targets):
        raise NotImplementedError("This method should be overridden by subclasses")




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
        super().__init__()
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

    def forward(self, outputs, chunks):
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
        chunks_recon, vq_loss, _, embeddings = outputs
        
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
        
        # Return detailed loss breakdown for monitoring
        self.last_losses = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'vq': vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss,
            'bottleneck': bottleneck_loss.item() if isinstance(bottleneck_loss, torch.Tensor) else 0        ,
            'bottleneck_var': var_loss.item() if embeddings is not None else 0,
            'bottleneck_cov': decorr_loss.item() if embeddings is not None else 0
        }
        
        return total_loss
    


'''
From here all losses are deprecated and might not work properly
'''   
class VaeLoss(TorchLoss):
    def __init__(self, beta=1.0):
        self.beta: float = beta
        self.shape_checked: bool = None

    def __call__(self, outputs, inputs):
        recon_x, mu, logvar = outputs
        return self.compute(recon_x, inputs, mu, logvar)
    
    def compute(self, recon_x, x, mu, logvar):
        '''
        VAE loss = Reconstruction loss + KL divergence (disentangled)
        
        Args:
            recon_x: Reconstructed input
            x: Original input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            beta: Weight for KL divergence term (default: 1.0)
        '''
        if self.shape_checked is None:
            assert recon_x.shape == x.shape, f"Reconstructed shape {recon_x.shape} doesn't match input shape {x.shape}!"
            self.shape_checked = True
            
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kld = torch.mean(kld)

        return recon_loss + self.beta * kld
    

def mask_outputs(x, matrix):
    mask_np = np.where(matrix == None, 0.0, 1.0)
    mask = torch.tensor(mask_np, dtype=torch.float32)
    mask = mask.view(1, 1, *mask.shape, 1)
    device = x.device
    x = x * mask.to(device)
    return x

class TorchMSELoss(TorchLoss):
    def __init__(self, emb_loss=0, masked: bool = False, matrix=None, scale = 1.0):
        super(TorchMSELoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.emb_loss = emb_loss
        self.masked = masked
        self.matrix = matrix
        self.scale = scale
        
    def forward(self, outputs, inputs):
        if isinstance(outputs, tuple):
            reconstruction = outputs[0]  
            embedding = outputs[1]
        else:
            reconstruction = outputs  
            embedding = None

        if self.masked:
            reconstruction = mask_outputs(reconstruction, self.matrix)

        loss = self.mse_loss(reconstruction, inputs)

        if embedding is not None:
            L1_norm = torch.mean(torch.abs(embedding))
            loss += self.emb_loss * L1_norm

        if torch.isnan(loss):
            print(outputs, inputs)
            sys.exit(0)

        return loss * self.scale


class PerceptualLoss(TorchLoss):
    """Feature-based loss that provides stronger encoder gradients."""
    def __init__(self, model, feature_weight=0.5, pixel_weight=0.5):
        super().__init__()
        self.model = model
        self.feature_weight = feature_weight
        self.pixel_weight = pixel_weight

    def forward(self, model_output, target):
        if isinstance(model_output, tuple):
            reconstruction, rec_features = model_output
        else:
            reconstruction = model_output
            rec_features = self.model.encode(reconstruction)

        # Compute target features (no gradient)
        with torch.no_grad():
            target_features = self.model.encode(target)

        # Compute feature and pixel losses
        feature_loss = F.mse_loss(rec_features, target_features)
        pixel_loss = F.mse_loss(reconstruction, target)

        total_loss = (
            self.feature_weight * feature_loss +
            self.pixel_weight * pixel_loss
        )
        return total_loss
    

class TorchL1Loss(TorchLoss):
    def __init__(self, emb_loss=0, masked: bool = False, matrix=None, scale = 1.0):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.emb_loss = emb_loss
        self.masked = masked
        self.matrix = matrix
        self.scale = scale
        
    def forward(self, outputs, inputs):
        if isinstance(outputs, tuple):
            reconstruction = outputs[0]  
            embedding = outputs[1]
        else:
            reconstruction = outputs  
            embedding = None

        if self.masked:
            reconstruction = mask_outputs(reconstruction, self.matrix)

        loss = self.l1_loss(reconstruction, inputs)

        if embedding is not None and self.emb_loss > 0:
            L1_norm = torch.mean(torch.abs(embedding))
            loss += self.emb_loss * L1_norm

        if torch.isnan(loss):
            print(outputs, inputs)
            sys.exit(0)

        return loss * self.scale

class VQVAELoss(TorchLoss):
    """
    Complete loss function for VQ-VAE.
    
    Components:
    1. Reconstruction loss (MSE, L1, or perceptual)
    2. VQ commitment loss (already computed in VectorQuantizer)
    3. Optional perceptual loss for better quality
    
    Usage:
        criterion = VQVAELoss(recon_loss_type='mse', recon_weight=1.0)
        
        # During training:
        reconstruction, vq_loss, indices = model(input)
        loss = criterion(input, reconstruction, vq_loss)
    """
    def __init__(
        self,
        recon_loss_type='mse',  # 'mse', 'l1', or 'perceptual'
        recon_weight=1.0,
        perceptual_weight=0.1
    ):
        super().__init__()
        
        self.recon_loss_type = recon_loss_type
        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
        
    def _perceptual_loss(self, x, x_recon):
        """
        Perceptual loss based on gradient similarity.
        Helps preserve high-frequency details in EEG signals.
        """
        # Compute gradients
        def compute_gradients(tensor):
            dx = tensor[:, :, 1:, :, :] - tensor[:, :, :-1, :, :]
            dy = tensor[:, :, :, 1:, :] - tensor[:, :, :, :-1, :]
            dt = tensor[:, :, :, :, 1:] - tensor[:, :, :, :, :-1]
            return dx, dy, dt
        
        x_grads = compute_gradients(x)
        recon_grads = compute_gradients(x_recon)
        
        loss = sum(F.l1_loss(g1, g2) for g1, g2 in zip(x_grads, recon_grads))
        return loss / 3.0
    
    def forward(self, outputs ,x):
        """
        Compute total loss.
        
        Args:
            x: Original input (B, C, H, W, T)
            x_recon: Reconstructed output (B, C, H, W, T)
            vq_loss: VQ commitment loss from model
            
        Returns:
            loss: Total loss (scalar)
        """
        # Reconstruction loss
        x_recon, vq_loss, _ = outputs
        if self.recon_loss_type == 'mse':
            recon_loss = F.mse_loss(x_recon, x)
        elif self.recon_loss_type == 'l1':
            recon_loss = F.l1_loss(x_recon, x)
        elif self.recon_loss_type == 'perceptual':
            recon_loss = F.mse_loss(x_recon, x)
            perc_loss = self._perceptual_loss(x, x_recon)
            recon_loss = recon_loss + self.perceptual_weight * perc_loss
        else:
            raise ValueError(f"Unknown recon_loss_type: {self.recon_loss_type}")
        
        # Total loss
        total_loss = self.recon_weight * recon_loss + vq_loss
        
        return total_loss