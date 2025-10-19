import sys
import torch 
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class VaeLoss:
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

class CustomMSELoss(torch.nn.Module):
    def __init__(self, emb_loss=0, masked: bool = False, matrix=None, scale = 1.0):
        super(CustomMSELoss, self).__init__()
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



import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceptualLoss(nn.Module):
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
    

class CustomL1Loss(nn.Module):
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

        if embedding is not None:
            L1_norm = torch.mean(torch.abs(embedding))
            loss += self.emb_loss * L1_norm

        if torch.isnan(loss):
            print(outputs, inputs)
            sys.exit(0)

        return loss * self.scale
    

class SequenceVQVAELoss(nn.Module):
    """
    Loss function for SequenceProcessor.
    Handles sequences of chunks.
    
    Usage:
        criterion = SequenceVQVAELoss(recon_loss_type='mse')
        
        # During training:
        chunks = ...  # (B, num_chunks, 25, 7, 5, 32)
        reconstruction, vq_loss, indices = seq_processor(chunks)
        loss = criterion(chunks, reconstruction, vq_loss)
    """
    def __init__(
        self,
        recon_loss_type='mse',
        recon_weight=1.0,
        perceptual_weight=0.1
    ):
        super().__init__()
        
        self.recon_loss_type = recon_loss_type
        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
    
    def _perceptual_loss(self, x, x_recon):
        """Perceptual loss for sequence of chunks"""
        # Flatten sequence dimension for processing
        batch_size, num_chunks = x.shape[:2]
        x_flat = x.view(-1, *x.shape[2:])
        x_recon_flat = x_recon.view(-1, *x_recon.shape[2:])
        
        # Compute gradients
        def compute_gradients(tensor):
            dx = tensor[:, :, 1:, :, :] - tensor[:, :, :-1, :, :]
            dy = tensor[:, :, :, 1:, :] - tensor[:, :, :, :-1, :]
            dt = tensor[:, :, :, :, 1:] - tensor[:, :, :, :, :-1]
            return dx, dy, dt
        
        x_grads = compute_gradients(x_flat)
        recon_grads = compute_gradients(x_recon_flat)
        
        loss = sum(F.l1_loss(g1, g2) for g1, g2 in zip(x_grads, recon_grads))
        return loss / 3.0

    def forward(self, outputs, chunks):
        """
        Compute total loss for sequence.
        
        Args:
            chunks: Original chunks (B, num_chunks, 25, 7, 5, 32)
            chunks_recon: Reconstructed chunks (B, num_chunks, 25, 7, 5, 32)
            vq_loss: VQ commitment loss from model
            
        Returns:
            loss: Total loss (scalar)
        """
        chunks_recon, vq_loss, _= outputs
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
        
        # Total loss
        total_loss = self.recon_weight * recon_loss + vq_loss
        
        return total_loss
    

class VQVAELoss(nn.Module):
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
    
    def forward(self, x, x_recon, vq_loss):
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