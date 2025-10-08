import torch 

import torch.nn.functional as F

class VaeLoss:
    def __init__(self, beta=1.0):
        self.beta = beta

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
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kld = torch.mean(kld)

        return recon_loss + self.beta * kld
    