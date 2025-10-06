import math

import torch
import torch.nn as nn


class Conv3DAutoencoder(nn.Module):
    def __init__(self, in_channels=50, embedding_dim=128):
        """
        3D Convolutional Autoencoder for input shape [batch, 50, 7, 5, 250]

        Args:
            in_channels: Number of input channels (50 in your case)
            embedding_dim: Size of the latent embedding space
        """
        super(Conv3DAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # Input: [batch, 50, 7, 5, 250]
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            # [batch, 64, 7, 5, 250]
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            # [batch, 128, 4, 3, 125]
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            # [batch, 256, 2, 2, 63]
        )

        # Calculate flattened size after encoder
        self.flat_size = 256 * 2 * 2 * 63  # 64512

        # Bottleneck (embedding space)
        self.fc_encoder = nn.Linear(self.flat_size, embedding_dim)
        self.fc_decoder = nn.Linear(embedding_dim, self.flat_size)

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
