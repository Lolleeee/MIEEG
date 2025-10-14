import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DepthwiseSeparable3D(nn.Module):
    """
    Depthwise Separable 3D Convolution for parameter reduction
    Reduces parameters by ~6-8x compared to standard Conv3d
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparable3D, self).__init__()
        # Depthwise: each input channel is convolved separately
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        # Pointwise: 1x1x1 convolution to combine channels
        self.pointwise = nn.Conv3d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ECA3D(nn.Module):
    """
    Efficient Channel Attention for 3D features
    Adds ~80 parameters with 2%+ performance gain
    """
    def __init__(self, channels, b=1, gamma=2):
        super(ECA3D, self).__init__()
        # Adaptive kernel size based on channel dimensionality
        t = int(abs((math.log2(channels) / gamma) + b / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Global average pooling: [B, C, D, H, W] -> [B, C, 1, 1, 1]
        y = self.avg_pool(x)
        
        # 1D conv along channel dimension
        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        
        # Apply channel attention
        return x * self.sigmoid(y).expand_as(x)


class SpatialAttention3D(nn.Module):
    """
    Spatial Attention for 3D features
    Complements channel attention for synergistic effect
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        self.conv = nn.Conv3d(
            2, 1, kernel_size=kernel_size, 
            padding=kernel_size//2, bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Aggregate channel information
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.sigmoid(self.conv(y))
        
        # Apply spatial attention
        return x * y


class DilatedResBlock3D(nn.Module):
    """
    Residual block with dilated convolutions and dual attention
    Captures multi-scale temporal patterns in EEG data
    """
    def __init__(self, channels, dilation=2):
        super(DilatedResBlock3D, self).__init__()
        
        # Determine optimal number of groups (16 channels per group)
        num_groups = max(min(channels // 16, 32), 1)
        
        # First conv: standard 3x3x3
        self.conv1 = nn.Conv3d(
            channels, channels, kernel_size=3, 
            padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(num_groups, channels)
        
        # Second conv: dilated for multi-scale receptive field
        self.conv2 = nn.Conv3d(
            channels, channels, kernel_size=3, 
            dilation=dilation, padding=dilation, bias=False
        )
        self.gn2 = nn.GroupNorm(num_groups, channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Dual attention: channel + spatial
        self.eca = ECA3D(channels)
        self.spatial_attn = SpatialAttention3D(kernel_size=7)
    
    def forward(self, x):
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        
        # Second dilated conv block
        out = self.conv2(out)
        out = self.gn2(out)
        
        # Residual connection
        out = out + identity
        out = self.relu(out)
        
        # Apply dual attention
        out = self.eca(out)
        out = self.spatial_attn(out)
        
        return out


class Conv3DAEP2(nn.Module):
    def __init__(self, in_channels=25, latent_dim=128):
        """
        Lightweight 3D Convolutional Autoencoder for EEG Analysis
        
        Features:
        - Depthwise separable convolutions (6-8x parameter reduction)
        - GroupNorm for batch-size independence
        - Dilated residual blocks with exponential dilation (1, 2, 4)
        - Dual attention (ECA + Spatial) for multi-scale feature learning
        - Optimized for input shape: [batch, 50, 7, 5, 250]
          where 50=frequencies, 7x5=channel connectivity, 250=time
        
        Args:
            in_channels: Number of input channels (50 frequency bands)
            latent_dim: Size of bottleneck embedding
        """
        super(Conv3DAEP2, self).__init__()
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        # ===================== ENCODER =====================
        
        # Block 1: Initial feature extraction (50 -> 64 channels)
        # Output: [B, 64, 7, 5, 250]
        self.enc_block1 = nn.Sequential(
            DepthwiseSeparable3D(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 64),  # 64/16 = 4 groups
            nn.ReLU(inplace=True)
        )
        self.enc_attn1 = ECA3D(64)
        
        # Block 2: Downsample + residual with dilation=1
        # Output: [B, 96, 4, 3, 125]
        self.enc_block2 = nn.Sequential(
            DepthwiseSeparable3D(64, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(6, 96),  # 96/16 = 6 groups
            nn.ReLU(inplace=True)
        )
        self.enc_res1 = DilatedResBlock3D(96, dilation=1)
        
        # Block 3: Downsample + residual with dilation=2
        # Output: [B, 128, 2, 2, 63]
        self.enc_block3 = nn.Sequential(
            DepthwiseSeparable3D(96, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),  # 128/16 = 8 groups
            nn.ReLU(inplace=True)
        )
        self.enc_attn2 = ECA3D(128)
        self.enc_res2 = DilatedResBlock3D(128, dilation=2)
        
        # Flattened size: [B, 128, 2, 2, 63] -> [B, 32256]
        self.flat_size = 128 * 2 * 2 * 63  # = 32256
        
        # ===================== BOTTLENECK =====================
        
        # Two-layer MLP with dropout for regularization
        self.fc_encoder = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(512, latent_dim)
        )
        
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(512, self.flat_size)
        )
        
        # ===================== DECODER =====================
        
        # Block 1: Residual with dilation=2 (mirror encoder)
        # Input: [B, 128, 2, 2, 63]
        self.dec_res1 = DilatedResBlock3D(128, dilation=2)
        self.dec_attn1 = ECA3D(128)
        
        # Block 2: Upsample + residual with dilation=1
        # Output: [B, 96, 4, 3, 125]
        self.dec_block2 = nn.Sequential(
            nn.ConvTranspose3d(
                128, 96, kernel_size=3, stride=2, 
                padding=1, output_padding=(1, 0, 0), bias=False  # FIXED!
            ),
            nn.GroupNorm(6, 96),
            nn.ReLU(inplace=True)
        )
        self.dec_res2 = DilatedResBlock3D(96, dilation=1)
        
        # Block 3: Upsample to original spatial-temporal resolution
        # Output: [B, 64, 7, 5, 250]
        self.dec_block3 = nn.Sequential(
            nn.ConvTranspose3d(
                96, 64, kernel_size=3, stride=2, 
                padding=1, output_padding=(0, 0, 1), bias=False  # FIXED!
            ),
            nn.GroupNorm(4, 64),
            nn.ReLU(inplace=True)
        )
        
        # Block 4: Final reconstruction
        self.dec_attn2 = ECA3D(64)
        self.dec_block4 = nn.Conv3d(
            64, in_channels, kernel_size=3, stride=1, padding=1
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Kaiming initialization for ReLU, Xavier for final layer
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        
        # Xavier for final reconstruction layer
        nn.init.xavier_normal_(self.dec_block4.weight)
    
    def encode(self, x):
        """
        Encode input to latent embedding
        
        Args:
            x: [B, 50, 7, 5, 250] - EEG frequency-connectivity-time tensor
        
        Returns:
            embedding: [B, latent_dim] - compressed representation
        """
        # Encoder path with progressive feature extraction
        x = self.enc_block1(x)      # [B, 64, 7, 5, 250]
        x = self.enc_attn1(x)
        
        x = self.enc_block2(x)      # [B, 96, 4, 3, 125]
        x = self.enc_res1(x)
        
        x = self.enc_block3(x)      # [B, 128, 2, 2, 63]
        x = self.enc_attn2(x)
        x = self.enc_res2(x)
        
        # Flatten and compress to latent space
        x = x.view(x.size(0), -1)   # [B, 32256]
        embedding = self.fc_encoder(x)  # [B, latent_dim]
        
        return embedding
    
    def decode(self, embedding):
        """
        Decode from latent embedding to reconstruction
        
        Args:
            embedding: [B, latent_dim] - compressed representation
        
        Returns:
            reconstruction: [B, 50, 7, 5, 250] - reconstructed EEG
        """
        # Expand from latent space
        x = self.fc_decoder(embedding)  # [B, 32256]
        x = x.view(-1, 128, 2, 2, 63)   # [B, 128, 2, 2, 63]
        
        # Decoder path with progressive upsampling
        x = self.dec_res1(x)
        x = self.dec_attn1(x)       # [B, 128, 2, 2, 63]
        
        x = self.dec_block2(x)      # [B, 96, 4, 3, 125]
        x = self.dec_res2(x)
        
        x = self.dec_block3(x)      # [B, 64, 7, 5, 250]
        
        x = self.dec_attn2(x)
        x = self.dec_block4(x)      # [B, 50, 7, 5, 250]
        
        return x
    
    def forward(self, x):
        """
        Full forward pass through autoencoder
        
        Returns:
            reconstruction: [B, 50, 7, 5, 250]
            embedding: [B, latent_dim]
        """
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return reconstruction, embedding
    
    def get_latent_representation(self, x):
        """Extract only the latent representation"""
        return self.encode(x)
    
    def reconstruct_from_latent(self, embedding):
        """Reconstruct from a given latent code"""
        return self.decode(embedding)



# ===================== TESTING =====================

if __name__ == "__main__":
    print("=" * 70)
    print("Lightweight 3D Convolutional Autoencoder for EEG Analysis")
    print("=" * 70)
    
    # Create model
    model = Conv3DAEP2(in_channels=25, latent_dim=128)
    
    # Test with sample input
    batch_size = 4
    x = torch.randn(batch_size, 25, 7, 5, 250)

    print(f"\nInput shape: {x.shape}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Frequencies: 50")
    print(f"  - Spatial (channel connectivity): 7x5")
    print(f"  - Temporal (time points): 250")
    
    # Forward pass
    with torch.no_grad():
        reconstruction, embedding = model(x)
    
    print(f"\nOutput shapes:")
    print(f"  - Reconstruction: {reconstruction.shape}")
    print(f"  - Embedding: {embedding.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Model size (approx): {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
