import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class LayerNorm3d(nn.Module):
    """LayerNorm for 3D data (channels-first format)"""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
    
    def forward(self, x):
        # x: (B, C, D, H, W)
        # Permute to (B, D, H, W, C) for LayerNorm
        x = x.permute(0, 2, 3, 4, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # Permute back to (B, C, D, H, W)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class StochasticDepth(nn.Module):
    """Drop paths (Stochastic Depth) per sample for regularization"""
    def __init__(self, drop_prob: float = 0.0, mode: str = "row"):
        super().__init__()
        self.drop_prob = drop_prob
        self.mode = mode
    
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvNeXt3DBlock(nn.Module):
    """
    ConvNeXt block adapted for 3D data
    
    Architecture:
    - Depthwise 7x7x7 3D convolution
    - LayerNorm
    - 1x1x1 conv to expand channels (4x)
    - GELU activation
    - 1x1x1 conv to reduce channels back
    - Layer scale + Stochastic depth
    - Residual connection
    """
    def __init__(
        self, 
        dim: int,
        kernel_size: int = 7,
        layer_scale_init: float = 1e-6,
        drop_path: float = 0.0,
        expansion_ratio: int = 4
    ):
        super().__init__()
        
        # Depthwise convolution
        self.dwconv = nn.Conv3d(
            dim, dim, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            groups=dim,
            bias=True
        )
        
        # Normalization
        self.norm = LayerNorm3d(dim)
        
        # Pointwise/Inverted Bottleneck MLP
        self.pwconv1 = nn.Conv3d(dim, dim * expansion_ratio, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(dim * expansion_ratio, dim, kernel_size=1, bias=True)
        
        # Layer scale
        self.gamma = nn.Parameter(
            layer_scale_init * torch.ones(dim, 1, 1, 1)
        ) if layer_scale_init > 0 else None
        
        # Stochastic depth
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x):
        shortcut = x
        
        # Depthwise conv
        x = self.dwconv(x)
        
        # Norm
        x = self.norm(x)
        
        # Inverted bottleneck
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Layer scale
        if self.gamma is not None:
            x = self.gamma * x
        
        # Stochastic depth + residual
        x = self.drop_path(x)
        x = shortcut + x
        
        return x


class Conv3DAE(nn.Module):
    """
    Fully tweakable ConvNeXt-based 3D Autoencoder
    
    Args:
        in_channels: Input channels
        latent_dim: Latent dimension at bottleneck
        base_channels: Base channel width (default: 32)
        channel_multipliers: Channel multiplier for each stage (e.g., [1, 2, 4, 8])
        depths: Number of ConvNeXt blocks per stage (e.g., [2, 2, 4, 2])
        kernel_size: Kernel size for depthwise convolutions (default: 7)
        layer_scale_init: Initial value for layer scale (default: 1e-6)
        drop_path_rate: Stochastic depth rate (default: 0.1)
        expansion_ratio: Channel expansion in ConvNeXt blocks (default: 4)
    """
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        base_channels: int = 32,
        channel_multipliers: List[int] = [1, 2, 4, 8],
        depths: List[int] = [2, 2, 4, 2],
        kernel_size: int = 7,
        layer_scale_init: float = 1e-6,
        drop_path_rate: float = 0.1,
        expansion_ratio: int = 4
    ):
        super().__init__()
        
        assert len(channel_multipliers) == len(depths), \
            "channel_multipliers and depths must have same length"
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.num_stages = len(depths)
        
        # Calculate channels for each stage
        stage_channels = [base_channels * mult for mult in channel_multipliers]
        
        # Calculate drop path rates (linearly increase per block)
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        
        # ============ ENCODER ============
        
        # Stem - initial projection
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, stage_channels[0], kernel_size=3, padding=1, bias=True),
            LayerNorm3d(stage_channels[0])
        )
        
        # Build encoder stages
        self.encoder_stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        block_idx = 0
        for stage_idx in range(self.num_stages):
            # ConvNeXt blocks for this stage
            blocks = []
            in_ch = stage_channels[stage_idx]
            
            for _ in range(depths[stage_idx]):
                blocks.append(
                    ConvNeXt3DBlock(
                        dim=in_ch,
                        kernel_size=kernel_size,
                        layer_scale_init=layer_scale_init,
                        drop_path=dp_rates[block_idx],
                        expansion_ratio=expansion_ratio
                    )
                )
                block_idx += 1
            
            self.encoder_stages.append(nn.Sequential(*blocks))
            
            # Downsampling (except for last stage)
            if stage_idx < self.num_stages - 1:
                self.downsample_layers.append(
                    nn.Sequential(
                        LayerNorm3d(stage_channels[stage_idx]),
                        nn.Conv3d(
                            stage_channels[stage_idx],
                            stage_channels[stage_idx + 1],
                            kernel_size=2,
                            stride=2,
                            bias=True
                        )
                    )
                )
        
        # Bottleneck - compress to latent dimension
        final_channels = stage_channels[-1]
        self.bottleneck_encode = nn.Sequential(
            LayerNorm3d(final_channels),
            nn.Conv3d(final_channels, latent_dim, kernel_size=1, bias=True),
            nn.GELU()
        )
        
        # ============ DECODER ============
        
        # Bottleneck - expand from latent dimension
        self.bottleneck_decode = nn.Sequential(
            nn.Conv3d(latent_dim, final_channels, kernel_size=1, bias=True),
            LayerNorm3d(final_channels),
            nn.GELU()
        )
        
        # Build decoder stages (reverse order)
        self.decoder_stages = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        
        for stage_idx in range(self.num_stages - 1, -1, -1):
            # Upsampling (except for last decoder stage)
            if stage_idx < self.num_stages - 1:
                self.upsample_layers.append(
                    nn.Sequential(
                        nn.ConvTranspose3d(
                            stage_channels[stage_idx + 1],
                            stage_channels[stage_idx],
                            kernel_size=2,
                            stride=2,
                            bias=True
                        ),
                        LayerNorm3d(stage_channels[stage_idx])
                    )
                )
            
            # ConvNeXt blocks for this stage
            blocks = []
            out_ch = stage_channels[stage_idx]
            
            for _ in range(depths[stage_idx]):
                blocks.append(
                    ConvNeXt3DBlock(
                        dim=out_ch,
                        kernel_size=kernel_size,
                        layer_scale_init=layer_scale_init,
                        drop_path=0.0,  # No drop path in decoder
                        expansion_ratio=expansion_ratio
                    )
                )
            
            self.decoder_stages.append(nn.Sequential(*blocks))
        
        # Final projection back to input channels
        self.head = nn.Conv3d(stage_channels[0], in_channels, kernel_size=3, padding=1, bias=True)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def encode(self, x):
        """Encode input to latent representation"""
        # Stem
        x = self.stem(x)
        
        # Encoder stages with downsampling
        for stage_idx in range(self.num_stages):
            x = self.encoder_stages[stage_idx](x)
            if stage_idx < self.num_stages - 1:
                x = self.downsample_layers[stage_idx](x)
        
        # Bottleneck
        z = self.bottleneck_encode(x)
        return z
    
    def decode(self, z):
        """Decode latent representation to output"""
        # Bottleneck
        x = self.bottleneck_decode(z)
        
        # Decoder stages with upsampling
        for stage_idx in range(self.num_stages):
            if stage_idx > 0:
                x = self.upsample_layers[stage_idx - 1](x)
            x = self.decoder_stages[stage_idx](x)
        
        # Head
        x = self.head(x)
        return x
    
    def forward(self, x):
        """Full forward pass"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def get_latent_shape(self, input_shape):
        """Calculate latent tensor shape given input shape"""
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape[1:])
            z = self.encode(dummy)
            return z.shape


# ============ EXAMPLE USAGE ============

if __name__ == "__main__":
    # Configuration examples
    
    # Tiny model
    model = Conv3DAE(
        in_channels=16,
        latent_dim=32,
        base_channels=32,
        channel_multipliers=[1, 2, 4, 4],
        depths=[1, 2, 4, 4],
        kernel_size=3,
        drop_path_rate=0.1
    )
    
    
    # Test with input shape (batch, channels, 7, 5, 64)
    batch_size = 4
    x = torch.randn(batch_size, 16, 7, 5, 128)

    
    
    print("=" * 70)
    print("Testing ConvNeXt 3D AutoEncoder Variants")
    print("=" * 70)
            
    with torch.no_grad():
        x_recon, z = model(x)

    total_params = sum(p.numel() for p in model.parameters())

    print(f"Input shape:        {tuple(x.shape)}")
    print(f"Latent shape:       {tuple(z.shape)}")
    print(f"Output shape:       {tuple(x_recon.shape)}")
    print(f"Total parameters:   {total_params:,}")
    print(f"Compression ratio:  {x.numel() / z.numel():.2f}x")

    print("\n" + "=" * 70)
    print("Customization Options:")
    print("=" * 70)
    print("- base_channels:       Control overall model width")
    print("- channel_multipliers: Control per-stage channel scaling")
    print("- depths:              Control number of blocks per stage")
    print("- kernel_size:         Depthwise conv kernel (3, 5, 7, etc.)")
    print("- drop_path_rate:      Regularization strength")
    print("- expansion_ratio:     MLP expansion factor (default: 4)")
    print("- layer_scale_init:    Layer scale initialization")
