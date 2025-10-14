import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ECA3D(nn.Module):
    """
    Efficient Channel Attention for 3D features
    """
    def __init__(self, channels, b=1, gamma=2):
        super(ECA3D, self).__init__()
        # Adaptive kernel size calculation
        t = int(abs((math.log2(channels) / gamma) + b / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Global average pooling
        y = self.avg_pool(x)
        
        # 1D convolution along channel dimension
        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        
        # Apply attention
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class DilatedResBlock3D(nn.Module):
    """
    Residual block with dilated convolutions for multi-scale features
    """
    def __init__(self, channels, dilation=2, groups=8):
        super(DilatedResBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, 
                               padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, channels)
        
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, 
                               dilation=dilation, 
                               padding=dilation, bias=False)
        self.gn2 = nn.GroupNorm(groups, channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.eca = ECA3D(channels)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.gn2(out)
        
        # Add residual connection
        out += identity
        out = self.relu(out)
        
        # Apply ECA attention
        out = self.eca(out)
        
        return out


class Conv3DAEP(nn.Module):
    def __init__(self, in_channels=25, latent_dim=128):
        """
        Enhanced 3D Convolutional Autoencoder with:
        - GroupNorm instead of BatchNorm
        - Dilated residual blocks
        - ECA attention
        - Lightweight design (moderate depth)
        
        Input shape: [batch, 50, 7, 5, 250]
        
        Args:
            in_channels: Number of input channels (50)
            latent_dim: Size of the latent embedding space
        """
        super(Conv3DAEP, self).__init__()
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        # Encoder (5 conv layers + 2 residual blocks = 7 layers total)
        self.encoder_conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )
        self.eca1 = ECA3D(64)
        
        self.encoder_conv2 = nn.Sequential(
            nn.Conv3d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.ReLU(inplace=True),
        )
        
        # Dilated residual block 1
        self.res_block1 = DilatedResBlock3D(96, dilation=2, groups=8)
        
        self.encoder_conv3 = nn.Sequential(
            nn.Conv3d(96, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
        )
        self.eca2 = ECA3D(128)
        
        # Dilated residual block 2
        self.res_block2 = DilatedResBlock3D(128, dilation=3, groups=8)
        
        # Calculate flattened size
        self.flat_size = 128 * 2 * 2 * 63  # After 2x stride=2 layers
        
        # Bottleneck with dropout for regularization
        self.fc_encoder = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, latent_dim)
        )
        
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, self.flat_size)
        )
        
        # Decoder (mirror encoder structure)
        self.res_block3 = DilatedResBlock3D(128, dilation=3, groups=8)
        
        self.decoder_conv1 = nn.Sequential(
            nn.ConvTranspose3d(128, 96, kernel_size=3, stride=2, 
                              padding=1, output_padding=(1, 0, 0), bias=False),
            nn.GroupNorm(8, 96),
            nn.ReLU(inplace=True),
        )
        self.eca3 = ECA3D(96)
        
        self.res_block4 = DilatedResBlock3D(96, dilation=2, groups=8)
        
        self.decoder_conv2 = nn.Sequential(
            nn.ConvTranspose3d(96, 64, kernel_size=3, stride=2, 
                              padding=1, output_padding=(0, 0, 1), bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )
        
        self.decoder_conv3 = nn.Conv3d(64, in_channels, kernel_size=3, 
                                       stride=1, padding=1)
    
    def encode(self, x):
        """Encode input to latent embedding"""
        x = self.encoder_conv1(x)
        x = self.eca1(x)
        
        x = self.encoder_conv2(x)
        x = self.res_block1(x)
        
        x = self.encoder_conv3(x)
        x = self.eca2(x)
        x = self.res_block2(x)
        
        x = x.view(x.size(0), -1)
        embedding = self.fc_encoder(x)
        
        return embedding
    
    def decode(self, embedding):
        """Decode from latent embedding to output"""
        x = self.fc_decoder(embedding)
        x = x.view(-1, 128, 2, 2, 63)
        
        x = self.res_block3(x)
        
        x = self.decoder_conv1(x)
        x = self.eca3(x)
        x = self.res_block4(x)
        
        x = self.decoder_conv2(x)
        x = self.decoder_conv3(x)
        
        return x
    
    def forward(self, x):
        """Full forward pass through autoencoder"""
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return reconstruction, embedding

# Training example
if __name__ == "__main__":
    # Create model
    model = Conv3DAEP(in_channels=25, latent_dim=128)
    print(model)
    # Test with sample input
    batch_size = 4
    x = torch.randn(batch_size, 25, 7, 5, 250)
    
    # Forward pass
    reconstruction, embedding = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {reconstruction.shape}")
    print(f"Embedding shape: {embedding.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
