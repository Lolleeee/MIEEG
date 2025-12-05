import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from dataclasses import dataclass
import numpy as np

# --- CONFIG ---

@dataclass
class VQAELightConfig:
    """Configuration for the VQ-VAE model."""
    use_quantizer: bool = True
    use_cwt: bool = True
    use_inverse_cwt: bool = False # NEW: Use Physics-Informed Decoder Head
    output_coeffs: bool = False  # If True, output CWT coeffs instead of time signal
    # CWT parameters
    cwt_frequencies: tuple = None
    chunk_samples: int = 160  # If None, no chunking

    # Data shape parameters
    num_input_channels: int = 2   # Power + Phase
    num_freq_bands: int = 25
    spatial_rows: int = 7
    spatial_cols: int = 5        
    time_samples: int = 160
    orig_channels: int = 32      
    
    # Encoder parameters
    encoder_2d_channels: list = None   # [16, 32]
    encoder_3d_channels: list = None   # [32, 64]
    embedding_dim: int = 64
    
    # VQ parameters
    codebook_size: int = 256
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    epsilon: float = 1e-5
    
    # Decoder parameters
    decoder_channels: list = None      # [64, 32]
    
    # Dropout
    dropout_2d: float = 0.05
    dropout_3d: float = 0.05
    dropout_bottleneck: float = 0.1
    dropout_decoder: float = 0.05
    
    # Architecture
    use_separable_conv: bool = True
    use_group_norm: bool = True
    num_groups: int = 8
    use_residual: bool = True
    use_squeeze_excitation: bool = True
    
    def __post_init__(self):
        if self.encoder_2d_channels is None:
            self.encoder_2d_channels = [16, 32]
        if self.encoder_3d_channels is None:
            self.encoder_3d_channels = [32, 64]
        if self.decoder_channels is None:
            self.decoder_channels = [64, 32]
        if self.cwt_frequencies is None:
            frequencies = np.logspace(np.log10(0.5), np.log10(79.9), 25)
            self.cwt_frequencies = tuple(frequencies)

# --- UTILS ---

class SqueezeExcitation2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return x * self.se(x)

class SqueezeExcitation3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, reduced, 1),
            nn.SiLU(inplace=True),
            nn.Conv3d(reduced, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return x * self.se(x)

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_residual=False, use_se=False):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.se = SqueezeExcitation2D(out_channels) if use_se else nn.Identity()
    def forward(self, x):
        res = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.se(x)
        return x + res if self.use_residual else x

class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_residual=False, use_se=False):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.se = SqueezeExcitation3D(out_channels) if use_se else nn.Identity()
    def forward(self, x):
        res = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.se(x)
        return x + res if self.use_residual else x

class VectorQuantizerLight(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.register_buffer('embeddings', torch.randn(num_embeddings, embedding_dim))
        nn.init.xavier_uniform_(self.embeddings)
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_embed_avg', self.embeddings.clone())
        
    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        flat_input_norm = F.normalize(flat_input, p=2, dim=1)
        embeddings_norm = F.normalize(self.embeddings, p=2, dim=1)
        distances = (torch.sum(flat_input_norm ** 2, dim=1, keepdim=True) + 
                     torch.sum(embeddings_norm ** 2, dim=1) - 
                     2 * torch.matmul(flat_input_norm, embeddings_norm.t()))
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(encoding_indices, self.embeddings)
        
        if self.training:
            self._ema_update(flat_input, encoding_indices)
        
        e_latent_loss = F.mse_loss(quantized.detach(), flat_input)
        quantized = flat_input + (quantized - flat_input).detach()
        q_latent_loss = F.mse_loss(quantized, flat_input.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = quantized.view(input_shape)
        avg_probs = torch.bincount(encoding_indices, minlength=self.num_embeddings).float() / len(encoding_indices)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return quantized, encoding_indices, {'vq_loss': vq_loss, 'perplexity': perplexity, 'codebook_usage': (avg_probs > 0).sum().float() / self.num_embeddings}

    def _ema_update(self, flat_input, encoding_indices):
        encodings_onehot = F.one_hot(encoding_indices, num_classes=self.num_embeddings).float()
        updated_cluster_size = torch.sum(encodings_onehot, dim=0)
        self.ema_cluster_size.data.mul_(self.decay).add_(updated_cluster_size, alpha=1 - self.decay)
        n = torch.sum(self.ema_cluster_size)
        self.ema_cluster_size.data.add_(self.epsilon).div_(n + self.num_embeddings * self.epsilon).mul_(n)
        embed_sum = torch.matmul(encodings_onehot.t(), flat_input)
        self.ema_embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
        self.embeddings.data.copy_(self.ema_embed_avg / self.ema_cluster_size.unsqueeze(1))

# --- ENCODERS ---

class Encoder2DStageLight(nn.Module):
    def __init__(self, config: VQAELightConfig):
        super().__init__()
        self.config = config
        layers = []
        in_channels = config.num_input_channels
        for i, out_channels in enumerate(config.encoder_2d_channels):
            if config.use_separable_conv and i > 0:
                conv = DepthwiseSeparableConv2d(in_channels, out_channels, 3, 2, 1, use_se=config.use_squeeze_excitation)
            else:
                conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False)
            norm = nn.GroupNorm(min(config.num_groups, out_channels), out_channels) if config.use_group_norm else nn.BatchNorm2d(out_channels)
            se = SqueezeExcitation2D(out_channels) if config.use_squeeze_excitation and not (config.use_separable_conv and i > 0) else nn.Identity()
            layers.extend([conv, norm, nn.SiLU(inplace=True), se])
            if i < len(config.encoder_2d_channels) - 1: layers.append(nn.Dropout2d(p=config.dropout_2d))
            in_channels = out_channels
        self.conv_net = nn.Sequential(*layers)
        self.freq_out = config.num_freq_bands
        self.time_out = config.time_samples
        for _ in config.encoder_2d_channels:
            self.freq_out = (self.freq_out + 1) // 2
            self.time_out = (self.time_out + 1) // 2
        self.out_channels = config.encoder_2d_channels[-1]
    def forward(self, x): return self.conv_net(x)

class Encoder3DStageLight(nn.Module):
    def __init__(self, config: VQAELightConfig, channels_in: int, time_in: int):
        super().__init__()
        self.config = config
        layers = []
        in_channels = channels_in
        for i, out_channels in enumerate(config.encoder_3d_channels):
            if config.use_separable_conv:
                conv = DepthwiseSeparableConv3d(in_channels, out_channels, 3, 2, 1, use_se=config.use_squeeze_excitation)
            else:
                conv = nn.Conv3d(in_channels, out_channels, 3, 2, 1, bias=False)
            norm = nn.GroupNorm(min(config.num_groups, out_channels), out_channels) if config.use_group_norm else nn.BatchNorm3d(out_channels)
            se = SqueezeExcitation3D(out_channels) if config.use_squeeze_excitation and not config.use_separable_conv else nn.Identity()
            layers.extend([conv, norm, nn.SiLU(inplace=True), se])
            if i < len(config.encoder_3d_channels) - 1: layers.append(nn.Dropout3d(p=config.dropout_3d))
            in_channels = out_channels
        self.conv_net = nn.Sequential(*layers)
        row_out, col_out, time_out = config.spatial_rows, config.spatial_cols, time_in
        for _ in config.encoder_3d_channels:
            row_out, col_out, time_out = (row_out + 1) // 2, (col_out + 1) // 2, (time_out + 1) // 2
        flatten_dim = config.encoder_3d_channels[-1] * row_out * col_out * time_out
        self.projection = nn.Sequential(
            nn.Flatten(), nn.Dropout(p=config.dropout_bottleneck),
            nn.Linear(flatten_dim, config.embedding_dim, bias=False), nn.LayerNorm(config.embedding_dim)
        )
    def forward(self, x): return self.projection(self.conv_net(x))

# --- DECODER ---

class ResidualBlock1d(nn.Module):
    def __init__(self, channels: int, kernel_size=3, use_se=True, norm_groups=8):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False)
        self.pointwise1 = nn.Conv1d(channels, channels, 1, bias=False)
        self.norm1 = nn.GroupNorm(norm_groups, channels)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False)
        self.pointwise2 = nn.Conv1d(channels, channels, 1, bias=False)
        self.norm2 = nn.GroupNorm(norm_groups, channels)
        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(channels, max(8, channels // 4), 1), nn.SiLU(inplace=True), nn.Conv1d(max(8, channels // 4), channels, 1), nn.Sigmoid()) if use_se else nn.Identity()
    def forward(self, x):
        res = x
        x = self.act(self.norm1(self.pointwise1(self.conv1(x))))
        x = self.norm2(self.pointwise2(self.conv2(x)))
        x = self.se(x)
        return x + res

class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable convolution (10x fewer params than standard conv).
    """
    def __init__(self, channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        
        # Depthwise: each channel processed independently
        self.depthwise = nn.Conv1d(
            channels, channels, 
            kernel_size, 
            padding=padding, 
            groups=channels,  # Key: groups = channels
            bias=False
        )
        
        # Pointwise: mix channels
        self.pointwise = nn.Conv1d(channels, out_channels, 1, bias=False)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EfficientResidualBlock1d(nn.Module):
    """
    Efficient residual block using depthwise separable convs.
    ~5x fewer params than standard residual block.
    """
    def __init__(self, channels):
        super().__init__()
        
        self.block = nn.Sequential(
            DepthwiseSeparableConv1d(channels, channels, kernel_size=3),
            nn.GroupNorm(16, channels),
            nn.GELU(),
            DepthwiseSeparableConv1d(channels, channels, kernel_size=3),
            nn.GroupNorm(16, channels)
        )
        
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.act(x + self.block(x))


class EfficientChannelAttention(nn.Module):
    """
    Efficient Squeeze-and-Excitation style attention.
    Only ~1K parameters but huge quality boost.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, reduced_channels, 1),
            nn.GELU(),
            nn.Conv1d(reduced_channels, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Global context
        attn_weights = self.attention(x)  # (B, C, 1)
        
        # Apply attention
        return x * attn_weights

class EfficientPowerfulDecoder(nn.Module):
    """
    Powerful decoder with <600K parameters.
    
    Key efficient tricks:
    1. Depthwise separable convolutions (10x fewer params)
    2. Gradual upsampling (4 stages instead of 2)
    3. Shared residual blocks
    4. Channel-wise attention (cheaper than full attention)
    """
    def __init__(self, config: VQAELightConfig):
        super().__init__()
        self.config = config
        
        # Stage 1: Project to very low resolution
        init_time = 10  # Start even smaller for gradual upsampling
        self.init_channels = 256
        
        self.projection = nn.Sequential(
            nn.Linear(config.embedding_dim, self.init_channels * init_time),
            nn.LayerNorm(self.init_channels * init_time),
            nn.GELU()
        )
        
        # Stage 2: Gradual upsampling with depthwise separable convs
        # 10 → 20 → 40 → 80 → 160 (4 stages)
        
        # 10 → 20
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            DepthwiseSeparableConv1d(256, 256, kernel_size=5),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            EfficientResidualBlock1d(256)
        )
        
        # 20 → 40
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            DepthwiseSeparableConv1d(256, 256, kernel_size=5),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            EfficientResidualBlock1d(256)
        )
        
        # 40 → 80
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            DepthwiseSeparableConv1d(256, 256, kernel_size=5),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            EfficientResidualBlock1d(256)
        )
        
        # 80 → 160
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            DepthwiseSeparableConv1d(256, 256, kernel_size=5),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            EfficientResidualBlock1d(256)
        )
        
        # Stage 3: Efficient channel-wise attention
        self.channel_attention = EfficientChannelAttention(256, reduction=16)
        
        # Stage 4: Final projection (grouped conv for efficiency)
        self.final = nn.Sequential(
            nn.Conv1d(256, 1600, 1, groups=8),  # Grouped 1x1 conv
        )
    
    def forward(self, z_q):
        B = z_q.shape[0]
        
        # Project
        x = self.projection(z_q)
        x = x.view(B, self.init_channels, 10)
        
        # Gradual upsampling
        x = self.up1(x)  # 10 → 20
        x = self.up2(x)  # 20 → 40
        x = self.up3(x)  # 40 → 80
        x = self.up4(x)  # 80 → 160
        
        # Apply attention
        x = self.channel_attention(x)
        
        # Final projection
        x = self.final(x)
        
        return x


class VQAELight(nn.Module):
    def __init__(self, config: VQAELightConfig | dict):
        super().__init__()
        if isinstance(config, dict): config = VQAELightConfig(**config)
        self.config = config
        self.use_cwt = config.use_cwt
        self.chunk_samples = config.chunk_samples

        if self.use_cwt:
            from packages.models.wavelet_head import CWTHead
            self.cwt_head = CWTHead(
                frequencies=config.cwt_frequencies,
                fs=160,
                num_channels=config.orig_channels,
                n_cycles=5.0,
                trainable=False,
                chunk_samples=config.chunk_samples
            )
            
            if config.use_inverse_cwt:
                from packages.models.wavelet_head import InverseCWTHead
                self.inv_cwt_head = InverseCWTHead(self.cwt_head)

        self.encoder_2d = Encoder2DStageLight(config)
        enc3d_in = self.encoder_2d.out_channels * self.encoder_2d.freq_out
        self.encoder_3d = Encoder3DStageLight(config, channels_in=enc3d_in, time_in=self.encoder_2d.time_out)
        
        self.vq = VectorQuantizerLight(
            config.codebook_size, config.embedding_dim,
            config.commitment_cost, config.ema_decay, config.epsilon
        )
        
        self.decoder = EfficientPowerfulDecoder(config)
        self.apply(self._init_weights)

        self.register_buffer(
            'good_channel_indices',
            torch.tensor([i for i in range(35) if i not in [0, 2, 4]])
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None: nn.init.constant_(m.weight, 1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def flatten_reconstruction(self, x):
        B, Ch, F, R, C, T = x.shape
        
        # Merge spatial dimensions
        x = x.reshape(B, Ch, F, R * C, T)
        
        # Use pre-computed indices
        x = torch.index_select(x, dim=3, index=self.good_channel_indices)
        
        # Flatten
        x = x.permute(0, 3, 1, 2, 4).reshape(B, 32 * Ch * F, T)
        
        return x
    
    def encode(self, x):
        B, Ch, F, R, C, T = x.shape
        x = x.permute(0, 3, 4, 1, 2, 5).reshape(B * R * C, Ch, F, T)
        
        x = self.encoder_2d(x)
        
        C_out, F_out, T_out = x.shape[1], x.shape[2], x.shape[3]
        x = x.view(B, R, C, C_out, F_out, T_out).permute(0, 3, 4, 1, 2, 5).reshape(B, C_out * F_out, R, C, T_out)
        
        return self.encoder_3d(x)
    
    def decode(self, z_q):
        """
        Decode from latent representation.
        
        Returns:
            If output_coeffs=True: (B, 1600, T) - CWT coefficients
            If use_inverse_cwt=True: (B, 32, T) - Raw EEG signal via iCWT
            Otherwise: (B, 32, T) - Raw EEG signal
        """
        x = self.decoder(z_q)  # (B, Out_Channels, T)
        
        # If output_coeffs is True, return coefficients without iCWT
        if self.config.output_coeffs:
            # x is already (B, 1600, T) from decoder
            return x
        
        # If Inverse CWT is ON (and not outputting coeffs), pass through head
        if self.use_cwt and self.config.use_inverse_cwt:
            x = self.inv_cwt_head(x)  # (B, 32, T)
            
        return x
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (B, 32, T_full) - Raw EEG signal
            
        Returns:
            dict with:
                'reconstruction': 
                    - (B, 1600, T) if output_coeffs=True
                    - (B, 32, T_full) otherwise (after unchunking if needed)
                'embeddings': (B, embedding_dim)
                'quantized': (B, embedding_dim) 
                'indices': (B,)
                'vq_loss': scalar
                'perplexity': scalar
                'codebook_usage': scalar
        """
        # Apply CWT transform
        if self.use_cwt:
            x_cwt = self.cwt_head(x)  # (B, 2, F, 7, 5, T)
        else:
            x_cwt = x 
            
        # Encode
        z_e = self.encode(x_cwt)

        # Quantize (optional)
        if self.config.use_quantizer:
            z_q, indices, vq_losses = self.vq(z_e)
        else:
            z_q = z_e
            indices = torch.zeros(z_e.shape[0], device=z_e.device).long()
            vq_losses = {
                'vq_loss': torch.tensor(0., device=z_e.device), 
                'perplexity': torch.tensor(0., device=z_e.device), 
                'codebook_usage': torch.tensor(1., device=z_e.device)
            }
        
        # Decode
        recon = self.decode(z_q)

        # Target for loss computation
        if self.config.output_coeffs:
            target = self.flatten_reconstruction(x_cwt)  
        else:
            target = x  # (B, 32, T_full)

        # Handle unchunking for raw EEG output (not for coeffs)
        if self.use_cwt and self.chunk_samples is not None:
            # Only unchunk if we're outputting raw EEG (not coeffs)
            recon = self.cwt_head._unchunk(recon)
            
            if self.config.output_coeffs:
                target = self.cwt_head._unchunk(target)
            
        return {
            'reconstruction': recon,
            'embeddings': z_e,
            'quantized': z_q,
            'indices': indices,
            **vq_losses,
            'target': target
        }

if __name__ == "__main__":
    print("="*60)
    print("TEST 1: Normal mode (raw EEG output)")
    print("="*60)
    config1 = VQAELightConfig(
        use_cwt=True, 
        use_inverse_cwt=True,
        embedding_dim=64,
        output_coeffs=True
    )
    model1 = VQAELight(config1)
    x = torch.randn(2, 32, 640)
    with torch.no_grad(): 
        out1 = model1(x)
    print(f"Input: {x.shape}")
    print(f"Reconstruction: {out1['reconstruction'].shape}")  # Should be (2, 32, 640)
    print(f"Target: {out1['target'].shape}")  # Should be (2, 32, 640)
    assert out1['reconstruction'].shape == (2, 32, 640), "Expected raw EEG output"
    
    print("\n" + "="*60)
    print("TEST 2: Coefficient output mode")
    print("="*60)
    config2 = VQAELightConfig(
        use_cwt=True, 
        use_inverse_cwt=False,  # Not used when output_coeffs=True
        output_coeffs=True,
        embedding_dim=64
    )
    model2 = VQAELight(config2)
    with torch.no_grad(): 
        out2 = model2(x)
    print(f"Input: {x.shape}")
    print(f"Reconstruction: {out2['reconstruction'].shape}")  # Should be (2, 1600, 160)
    assert out2['reconstruction'].shape == (2, 1600, 640), "Expected coefficient output"
    
    print("\n" + "="*60)
    print("TEST 3: No iCWT, no coeff output (direct decoder output)")
    print("="*60)
    config3 = VQAELightConfig(
        use_cwt=True, 
        use_inverse_cwt=False,
        output_coeffs=False,
        embedding_dim=64
    )
    model3 = VQAELight(config3)
    with torch.no_grad(): 
        out3 = model3(x)
    print(f"Input: {x.shape}")
    print(f"Reconstruction: {out3['reconstruction'].shape}")  # Should be (2, 32, 640)
    
    print("\n✅ All tests passed!")
