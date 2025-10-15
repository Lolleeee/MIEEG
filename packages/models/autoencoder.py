import math

import torch
import torch.nn as nn


class basicConv3DAE(nn.Module):
    def __init__(self, in_channels=50, latent_dim=128):
        """
        3D Convolutional Autoencoder for input shape [batch, 50, 7, 5, 250]

        Args:
            in_channels: Number of input channels (50 in your case)
            embedding_dim: Size of the latent embedding space
        """
        super(basicConv3DAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(

            nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
        )

        # Calculate flattened size after encoder
        self.flat_size = 256 * 2 * 2 * 63  # 64512

        # Bottleneck (embedding space)
        self.fc_encoder = nn.Linear(self.flat_size, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, self.flat_size)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=(1, 0, 0)
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=(0, 0, 1)
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, in_channels, kernel_size=3, stride=1, padding=1),
        )

    def encode(self, x):
        """Encode input to latent embedding"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        embedding = self.fc_encoder(x)
        return embedding

    def decode(self, embedding):
        """Decode from latent embedding to output"""
        x = self.fc_decoder(embedding)
        x = x.view(-1, 256, 2, 2, 63)  # Reshape
        x = self.decoder(x)
        return x

    def forward(self, x):
        """Full forward pass through autoencoder"""
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)

        return reconstruction

import math
import torch
import torch.nn as nn


class ResidualBlock3D(nn.Module):
    """Residual block for 3D convolutions with skip connections"""
    def __init__(self, channels):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.activation = nn.GELU()  # Better gradient flow than ReLU
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual  # Skip connection preserves gradients
        out = self.activation(out)
        return out


class Conv3DAE(nn.Module):
    def __init__(self, in_channels=50, latent_dim=128):
        """
        Improved 3D Convolutional Autoencoder with enhanced gradient flow
        
        Key improvements:
        - Residual connections to combat vanishing gradients
        - GELU activation for smoother gradient flow
        - Proper weight initialization (He initialization)
        - Skip connections between encoder and decoder
        - Layer normalization in bottleneck
        """
        super(Conv3DAE, self).__init__()
        
        # Encoder with residual blocks
        self.encoder_conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
        )
        self.encoder_res1 = ResidualBlock3D(64)
        
        self.encoder_conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.GELU(),
        )
        self.encoder_res2 = ResidualBlock3D(128)
        
        self.encoder_conv3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.GELU(),
        )
        self.encoder_res3 = ResidualBlock3D(256)
        
        # Calculate flattened size
        self.flat_size = 256 * 2 * 2 * 63
        
        # Bottleneck with layer normalization for stable gradients
        self.fc_encoder = nn.Sequential(
            nn.Linear(self.flat_size, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, self.flat_size),
        )
        
        # Decoder with residual blocks and skip connections
        self.decoder_res3 = ResidualBlock3D(256)
        self.decoder_conv3 = nn.Sequential(
            nn.ConvTranspose3d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=(1, 0, 0)
            ),
            nn.BatchNorm3d(128),
            nn.GELU(),
        )
        
        self.decoder_res2 = ResidualBlock3D(128)
        self.decoder_conv2 = nn.Sequential(
            nn.ConvTranspose3d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=(0, 0, 1)
            ),
            nn.BatchNorm3d(64),
            nn.GELU(),
        )
        
        self.decoder_res1 = ResidualBlock3D(64)
        self.decoder_conv1 = nn.Conv3d(64, in_channels, kernel_size=3, stride=1, padding=1)
        
        # Initialize weights for optimal gradient flow
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """He initialization for layers with GELU/ReLU activations"""
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm3d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def encode(self, x):
        """Encode with residual connections"""
        x = self.encoder_conv1(x)
        x = self.encoder_res1(x)
        
        x = self.encoder_conv2(x)
        x = self.encoder_res2(x)
        
        x = self.encoder_conv3(x)
        x = self.encoder_res3(x)
        
        x = x.view(x.size(0), -1)
        embedding = self.fc_encoder(x)
        return embedding
    
    def decode(self, embedding):
        """Decode with residual connections"""
        x = self.fc_decoder(embedding)
        x = x.view(-1, 256, 2, 2, 63)
        
        x = self.decoder_res3(x)
        x = self.decoder_conv3(x)
        
        x = self.decoder_res2(x)
        x = self.decoder_conv2(x)
        
        x = self.decoder_res1(x)
        x = self.decoder_conv1(x)
        return x
    
    def forward(self, x):
        """Full forward pass"""
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return reconstruction