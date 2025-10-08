import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE3D(nn.Module):
    def __init__(self, in_channels=50, latent_dim=128):
        '''
        Convolutional Variational Autoencoder for 5D tensors.
        
        Args:
            in_channels: Number of input channels (default: 50)
            latent_dim: Dimensionality of latent space (default: 128)
        
        Input shape: (batch, in_channels, 7, 5, 250)
        '''
        super(ConvVAE3D, self).__init__()
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        # ENCODER
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(1, 1, 4), stride=(1, 1, 2), padding=(0, 0, 1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(64, 128, kernel_size=(1, 1, 4), stride=(1, 1, 2), padding=(0, 0, 1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(128, 256, kernel_size=(1, 1, 4), stride=(1, 1, 2), padding=(0, 0, 1)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(256, 512, kernel_size=(1, 1, 4), stride=(1, 1, 2), padding=(0, 0, 1)),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2),
        )
        
        # Calculate flattened dimension after encoder
        self.flatten_dim = 512 * 7 * 5 * 15  
        
        # Latent space mapping
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        
        # DECODER
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=(1, 1, 4), stride=(1, 1, 2), padding=(0, 0, 1), output_padding=(0, 0, 1)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose3d(256, 128, kernel_size=(1, 1, 4), stride=(1, 1, 2), padding=(0, 0, 1), output_padding=(0, 0, 0)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose3d(128, 64, kernel_size=(1, 1, 4), stride=(1, 1, 2), padding=(0, 0, 1), output_padding=(0, 0, 1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose3d(64, in_channels, kernel_size=(1, 1, 4), stride=(1, 1, 2), padding=(0, 0, 1), output_padding=(0, 0, 0)),
        )
    
    def encode(self, x):
        '''Encode input to latent space parameters'''
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        '''Reparameterization trick: z = mu + sigma * epsilon'''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        '''Decode latent vector to reconstruction'''
        h = self.decoder_input(z)
        h = h.view(-1, 512, 7, 5, 15)  # Reshape to spatial dimensions
        reconstruction = self.decoder(h)
        return reconstruction
    
    def forward(self, x):
        '''Full forward pass through VAE'''
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar



