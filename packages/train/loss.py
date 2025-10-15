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