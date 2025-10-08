import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE3D(nn.Module):
    def __init__(self, in_channels=50, latent_dim=128, hidden_dims=None):
        '''
        Convolutional Variational Autoencoder for 5D tensors.
        
        Args:
            in_channels: Number of input channels (default: 50)
            latent_dim: Dimensionality of latent space (default: 128)
            hidden_dims: List of channel dimensions for encoder layers (default: [32, 64, 128])
        
        Input shape: (batch, in_channels, 7, 5, 250)
        '''
        super(ConvVAE3D, self).__init__()
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]
        
        self.hidden_dims = hidden_dims
        
        # ENCODER - Build dynamically based on hidden_dims
        encoder_layers = []
        prev_channels = in_channels
        
        for i, h_dim in enumerate(hidden_dims):
            encoder_layers.extend([
                nn.Conv3d(prev_channels, h_dim, kernel_size=3, stride=1 if i == 0 else 2, padding=1),
                nn.BatchNorm3d(h_dim),
                nn.ReLU()
            ])
            prev_channels = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate flattened dimension dynamically
        # For input (7, 5, 250) with 2 stride-2 convolutions:
        # After layer 1: (7, 5, 250) - no downsampling
        # After layer 2: (4, 3, 125) - stride 2
        # After layer 3: (2, 2, 63) - stride 2
        spatial_dims = self._calculate_spatial_dims(hidden_dims)
        self.flatten_dim = hidden_dims[-1] * spatial_dims[0] * spatial_dims[1] * spatial_dims[2]
        
        # Latent space mapping
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        self.spatial_dims = spatial_dims
        
        # DECODER - Build dynamically (reverse of encoder)
        decoder_layers = []
        reversed_hidden_dims = list(reversed(hidden_dims))
        
        for i in range(len(reversed_hidden_dims) - 1):
            decoder_layers.extend([
                nn.ConvTranspose3d(
                    reversed_hidden_dims[i],
                    reversed_hidden_dims[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=self._get_output_padding(i)
                ),
                nn.BatchNorm3d(reversed_hidden_dims[i + 1]),
                nn.ReLU()
            ])
        
        # Final layer to reconstruct original input channels
        decoder_layers.append(
            nn.Conv3d(reversed_hidden_dims[-1], in_channels, kernel_size=3, stride=1, padding=1)
        )
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def _calculate_spatial_dims(self, hidden_dims):
        '''Calculate spatial dimensions after encoder'''
        h, w, d = 7, 5, 250
        
        # First layer has stride 1
        # Subsequent layers have stride 2
        for i, _ in enumerate(hidden_dims):
            if i > 0:  # Skip first layer
                h = (h + 2 * 1 - 3) // 2 + 1
                w = (w + 2 * 1 - 3) // 2 + 1
                d = (d + 2 * 1 - 3) // 2 + 1
        
        return (h, w, d)
    
    def _get_output_padding(self, layer_idx):
        '''Determine output padding for transposed convolutions'''
        # Adjust based on layer to match original dimensions
        if layer_idx == 0:
            return (1, 0, 0)
        elif layer_idx == 1:
            return (0, 0, 1)
        return (0, 0, 0)
    
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
        h = h.view(-1, self.hidden_dims[-1], *self.spatial_dims)  # Reshape to spatial dimensions
        reconstruction = self.decoder(h)
        return reconstruction
    
    def forward(self, x):
        '''Full forward pass through VAE'''
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


class old_ConvVAE3D(nn.Module):
    def __init__(self, in_channels=50, latent_dim=128):
        '''
        Convolutional Variational Autoencoder for 5D tensors.
        
        Args:
            in_channels: Number of input channels (default: 50)
            latent_dim: Dimensionality of latent space (default: 128)
        
        Input shape: (batch, in_channels, 7, 5, 250)
        '''
        super(old_ConvVAE3D, self).__init__()
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        # ENCODER
        self.encoder = nn.Sequential(

            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )
        
        # Calculate flattened dimension after encoder
        self.flatten_dim = 128 * 2 * 2 * 63
        
        # Latent space mapping
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        
        # DECODER
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=(1, 0, 0)
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=(0, 0, 1)
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, in_channels, kernel_size=3, stride=1, padding=1),
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
        h = h.view(-1, 128, 2, 2, 63)  # Reshape to spatial dimensions
        reconstruction = self.decoder(h)
        return reconstruction
    
    def forward(self, x):
        '''Full forward pass through VAE'''
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar



