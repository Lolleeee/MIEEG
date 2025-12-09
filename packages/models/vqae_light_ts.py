import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class VQAELightConfig:
    use_quantizer: bool = False
    use_cwt: bool = True

    # CWT parameters
    cwt_frequencies: tuple = None
    chunk_samples: int = 160  # if None, no chunking
    use_log_compression = True
    normalize_outputs = True
    learnable_norm = True
    # Data shape parameters
    num_input_channels: int = 2   # power + phase
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
    dropout_bottleneck: float = 0.
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

        distances = (
            torch.sum(flat_input_norm ** 2, dim=1, keepdim=True) +
            torch.sum(embeddings_norm ** 2, dim=1) -
            2 * torch.matmul(flat_input_norm, embeddings_norm.t())
        )

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
        codebook_usage = (avg_probs > 0).sum().float() / self.num_embeddings

        return quantized, encoding_indices, {
            'vq_loss': vq_loss,
            'perplexity': perplexity,
            'codebook_usage': codebook_usage
        }

    def _ema_update(self, flat_input, encoding_indices):
        encodings_onehot = F.one_hot(encoding_indices, num_classes=self.num_embeddings).float()
        updated_cluster_size = torch.sum(encodings_onehot, dim=0)

        self.ema_cluster_size.data.mul_(self.decay).add_(updated_cluster_size, alpha=1 - self.decay)

        n = torch.sum(self.ema_cluster_size)
        self.ema_cluster_size.data.add_(self.epsilon).div_(
            n + self.num_embeddings * self.epsilon
        ).mul_(n)

        embed_sum = torch.matmul(encodings_onehot.t(), flat_input)
        self.ema_embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        self.embeddings.data.copy_(self.ema_embed_avg / self.ema_cluster_size.unsqueeze(1))


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
    def forward(self, x):
        return x * self.se(x)


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
    def forward(self, x):
        return x * self.se(x)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 use_residual=False, use_se=False):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.se = SqueezeExcitation2D(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        res = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.se(x)
        return x + res if self.use_residual else x


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 use_residual=False, use_se=False):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.se = SqueezeExcitation3D(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        res = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.se(x)
        return x + res if self.use_residual else x


class Encoder2DStageLight(nn.Module):
    def __init__(self, config: VQAELightConfig):
        super().__init__()
        self.config = config
        layers = []
        in_channels = config.num_input_channels

        for i, out_channels in enumerate(config.encoder_2d_channels):
            if config.use_separable_conv and i > 0:
                conv = DepthwiseSeparableConv2d(
                    in_channels, out_channels, 3, 2, 1,
                    use_se=config.use_squeeze_excitation
                )
            else:
                conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False)

            if config.use_group_norm:
                norm = nn.GroupNorm(min(config.num_groups, out_channels), out_channels)
            else:
                norm = nn.BatchNorm2d(out_channels)

            se = (
                SqueezeExcitation2D(out_channels)
                if config.use_squeeze_excitation and not (config.use_separable_conv and i > 0)
                else nn.Identity()
            )

            layers.extend([conv, norm, nn.SiLU(inplace=True), se])
            if i < len(config.encoder_2d_channels) - 1:
                layers.append(nn.Dropout2d(p=config.dropout_2d))

            in_channels = out_channels

        self.conv_net = nn.Sequential(*layers)

        self.freq_out = config.num_freq_bands
        self.time_out = config.time_samples
        for _ in config.encoder_2d_channels:
            self.freq_out = (self.freq_out + 1) // 2
            self.time_out = (self.time_out + 1) // 2

        self.out_channels = config.encoder_2d_channels[-1]

    def forward(self, x):
        return self.conv_net(x)


class Encoder3DStageLight(nn.Module):
    def __init__(self, config: VQAELightConfig, channels_in: int, time_in: int):
        super().__init__()
        self.config = config
        layers = []
        in_channels = channels_in

        for i, out_channels in enumerate(config.encoder_3d_channels):
            if config.use_separable_conv:
                conv = DepthwiseSeparableConv3d(
                    in_channels, out_channels, 3, 2, 1,
                    use_se=config.use_squeeze_excitation
                )
            else:
                conv = nn.Conv3d(in_channels, out_channels, 3, 2, 1, bias=False)

            if config.use_group_norm:
                norm = nn.GroupNorm(min(config.num_groups, out_channels), out_channels)
            else:
                norm = nn.BatchNorm3d(out_channels)

            se = (
                SqueezeExcitation3D(out_channels)
                if config.use_squeeze_excitation and not config.use_separable_conv
                else nn.Identity()
            )

            layers.extend([conv, norm, nn.SiLU(inplace=True), se])
            if i < len(config.encoder_3d_channels) - 1:
                layers.append(nn.Dropout3d(p=config.dropout_3d))

            in_channels = out_channels

        self.conv_net = nn.Sequential(*layers)

        row_out, col_out, time_out = config.spatial_rows, config.spatial_cols, time_in
        for _ in config.encoder_3d_channels:
            row_out = (row_out + 1) // 2
            col_out = (col_out + 1) // 2
            time_out = (time_out + 1) // 2

        flatten_dim = config.encoder_3d_channels[-1] * row_out * col_out * time_out
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=config.dropout_bottleneck),
            nn.Linear(flatten_dim, config.embedding_dim, bias=False),
            nn.LayerNorm(config.embedding_dim)
        )

    def forward(self, x):
        x = self.conv_net(x)
        return self.projection(x)

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            channels, channels,
            kernel_size,
            padding=padding,
            groups=channels,
            bias=False
        )
        self.pointwise = nn.Conv1d(channels, out_channels, 1, bias=False)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EfficientResidualBlock1d(nn.Module):
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
        attn_weights = self.attention(x)  # (B, C, 1)
        return x * attn_weights

class MultiScaleDecoder(nn.Module):
    """
    Multi-scale decoder that reconstructs low and high frequencies separately.
    Addresses time-frequency trade-off in EEG reconstruction.
    """
    def __init__(self, config: VQAELightConfig):
        super().__init__()
        self.config = config
        
        # Shared projection from bottleneck
        self.projection = nn.Sequential(
            nn.Linear(config.embedding_dim, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )
        
        # Low-frequency path (0-20 Hz) - smooth, long-range dependencies
        self.low_freq_path = nn.Sequential(
            nn.Linear(512, 128 * 40),
            nn.Unflatten(1, (128, 40)),
            nn.Upsample(scale_factor=4, mode='linear', align_corners=False),  # → 160
            DepthwiseSeparableConv1d(128, 64, kernel_size=9),  # Large kernel for smooth features
            nn.GroupNorm(8, 64),
            nn.GELU(),
            EfficientResidualBlock1d(64)
        )
        
        # High-frequency path (20-80 Hz) - preserve sharp details
        self.high_freq_path = nn.Sequential(
            nn.Linear(512, 128 * 40),
            nn.Unflatten(1, (128, 40)),
            nn.Upsample(scale_factor=4, mode='linear', align_corners=False),  # → 160
            DepthwiseSeparableConv1d(128, 64, kernel_size=3),  # Small kernel for details
            nn.GroupNorm(8, 64),
            nn.GELU(),
            EfficientResidualBlock1d(64)
        )
        
        # Combine both frequency bands
        self.combine = nn.Sequential(
            nn.Conv1d(128, 128, 1),  # Mix low + high (64 + 64 = 128)
            nn.GELU(),
            EfficientChannelAttention(128, reduction=8),
            nn.Conv1d(128, config.orig_channels, 1, groups=8)  # → 32 channels
        )
    
    def forward(self, z_q):
        """
        Args:
            z_q: (B, embedding_dim)
        Returns:
            (B, 32, 160)
        """
        x = self.projection(z_q)  # (B, 512)
        
        # Parallel frequency-specific processing
        low = self.low_freq_path(x)   # (B, 64, 160)
        high = self.high_freq_path(x)  # (B, 64, 160)
        
        # Concatenate and combine
        combined = torch.cat([low, high], dim=1)  # (B, 128, 160)
        return self.combine(combined)  # (B, 32, 160)


class VQAELight(nn.Module):
    def __init__(self, config: VQAELightConfig | Dict):
        super().__init__()
        if isinstance(config, dict):
            config = VQAELightConfig(**config)
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
                chunk_samples=config.chunk_samples,
                use_log_compression=config.use_log_compression,
                normalize_outputs=config.normalize_outputs,
                learnable_norm=config.learnable_norm
            )
        
        self.encoder_2d = Encoder2DStageLight(config)
        enc3d_in = self.encoder_2d.out_channels * self.encoder_2d.freq_out
        self.encoder_3d = Encoder3DStageLight(
            config, channels_in=enc3d_in, time_in=self.encoder_2d.time_out
        )
        
        self.vq = VectorQuantizerLight(
            config.codebook_size,
            config.embedding_dim,
            config.commitment_cost,
            config.ema_decay,
            config.epsilon
        )
        
        self.decoder = MultiScaleDecoder(config)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def encode(self, x):
        """
        x: (B_chunk, 2, F, 7, 5, T_chunk)
        returns: (B_chunk, embedding_dim)
        """
        Bc, Ch, F, R, C, T = x.shape
        x = x.permute(0, 3, 4, 1, 2, 5).reshape(Bc * R * C, Ch, F, T)
        x = self.encoder_2d(x)

        C_out, F_out, T_out = x.shape[1], x.shape[2], x.shape[3]
        x = x.view(Bc, R, C, C_out, F_out, T_out)
        x = x.permute(0, 3, 4, 1, 2, 5).reshape(Bc, C_out * F_out, R, C, T_out)

        return self.encoder_3d(x)

    def decode(self, z_q):
        """
        z_q: (B_chunk, embedding_dim)
        returns: (B_chunk, 32, 160)
        """
        return self.decoder(z_q)

    def forward(self, x):
        """
        x: (B, 32, 640) raw EEG

        returns:
            'reconstruction': (B, 32, 640)
            'target': (B, 32, 640)
        """
        if self.use_cwt:
            x_cwt = self.cwt_head(x)   # expects chunked CWT output
        else:
            raise ValueError("This version assumes use_cwt=True")

        # Encode
        z_e = self.encode(x_cwt)

        # Quantize
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

        # Decode chunk-level EEG
        recon_chunk = self.decode(z_q)  # (B_chunk, 32, 160)

        # Unchunk back to full (B, 32, 640)
        if self.use_cwt and self.chunk_samples is not None:
            recon = self.cwt_head._unchunk(recon_chunk)
        else:
            recon = recon_chunk

        return {
            'reconstruction': recon,
            'embeddings': z_e,
            'quantized': z_q,
            'indices': indices,
            **vq_losses
        }

if __name__ == "__main__":
    cfg = VQAELightConfig(
        use_cwt=True,
        embedding_dim=64
    )
    model = VQAELight(cfg)
    x = torch.randn(2, 32, 640)
    with torch.no_grad():
        out = model(x)
    print("Input:", x.shape)
    print("Reconstruction:", out['reconstruction'].shape)
    assert out['reconstruction'].shape == (2, 32, 640)
    print("✅ OK")
