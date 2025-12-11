import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class VQAELightConfig:
    use_quantizer: bool = False

    # CWT parameters
    cwt_frequencies: tuple = None
    chunk_samples: int = 160
    use_log_compression: bool = True
    normalize_outputs: bool = True
    learnable_norm: bool = True

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
        attn_weights = self.attention(x)
        return x * attn_weights


class SOTATimeFrequencyDecoder(nn.Module):
    """
    State-of-the-art decoder combining insights from RAVE, D-FaST, and TF-Fusion.
    
    Key principles:
    1. Multi-scale processing (like RAVE's multi-band)
    2. Separate time/freq encodings (like TF-Fusion)
    3. Cross-attention fusion (like D-FaST)
    """
    def __init__(self, config: VQAELightConfig):
        super().__init__()
        self.config = config
        hidden_dim = max(512, config.embedding_dim // 2)
        
        # Shared projection
        self.projection = nn.Sequential(
            nn.Linear(config.embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # ===== FREQUENCY DOMAIN BRANCH =====
        self.freq_branch = nn.Sequential(
            nn.Linear(hidden_dim, 128 * 40),
            nn.Unflatten(1, (128, 40)),
            
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            
            DepthwiseSeparableConv1d(64, 64, kernel_size=7),
            nn.GroupNorm(8, 64),
            nn.GELU()
        )
        
        # Separate heads for magnitude and phase
        self.freq_magnitude_head = nn.Sequential(
            nn.Conv1d(64, config.orig_channels, 1),
            nn.Softplus()
        )
        
        self.freq_phase_head = nn.Sequential(
            nn.Conv1d(64, config.orig_channels, 1),
            nn.Tanh()
        )
        
        # ===== TIME DOMAIN BRANCH =====
        self.time_branch = nn.Sequential(
            nn.Linear(hidden_dim, 256 * 40),
            nn.Unflatten(1, (256, 40)),
            
            nn.ConvTranspose1d(256, 128, kernel_size=8, stride=2, padding=3),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            DepthwiseSeparableConv1d(128, 128, kernel_size=5),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=2, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU()
        )
        
        # ===== CROSS-ATTENTION FUSION =====
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=8,
            batch_first=True
        )
        
        # Final projection after fusion
        self.fusion_head = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.GELU(),
            EfficientResidualBlock1d(64),
            nn.Conv1d(64, config.orig_channels, 1)
        )
        
        # Learnable weights
        self.alpha_freq = nn.Parameter(torch.tensor(0.7))
    
    def forward(self, z_q):
        """
        Args:
            z_q: (B, embedding_dim)
        Returns:
            (B, 32, 160) - final signal
        """
        shared = self.projection(z_q)
        
        # Branch 1: Frequency domain
        freq_features = self.freq_branch(shared)
        freq_magnitude = self.freq_magnitude_head(freq_features)
        freq_phase = self.freq_phase_head(freq_features) * torch.pi
        
        # Branch 2: Time domain  
        time_features = self.time_branch(shared)
        
        # Cross-attention fusion
        freq_features_upsampled = F.interpolate(
            freq_features, 
            size=time_features.shape[-1], 
            mode='linear', 
            align_corners=False
        )
        
        # Reshape for attention
        time_attn = time_features.transpose(1, 2)
        freq_attn = freq_features_upsampled.transpose(1, 2)
        
        # Cross-attention
        fused_features, _ = self.cross_attention(
            query=time_attn,
            key=freq_attn,
            value=freq_attn
        )
        
        fused_features = fused_features.transpose(1, 2)
        
        # Time-domain reconstruction
        time_recon = self.fusion_head(fused_features)
        
        # Frequency-domain reconstruction
        freq_recon = self._freq_to_time(freq_magnitude, freq_phase, target_length=160)
        
        # Combine both paths
        final = self.alpha_freq * freq_recon + (1 - self.alpha_freq) * time_recon
        
        return final
    
    def _freq_to_time(self, magnitude, phase, target_length):
        """Convert frequency domain to time domain."""
        n_fft = (target_length // 2) + 1
        
        if magnitude.shape[-1] != n_fft:
            magnitude = F.interpolate(magnitude, size=n_fft, mode='linear', align_corners=False)
            phase = F.interpolate(phase, size=n_fft, mode='linear', align_corners=False)
        
        complex_spectrum = magnitude * torch.exp(1j * phase)
        time_signal = torch.fft.irfft(complex_spectrum, n=target_length, dim=-1)
        
        return time_signal


class VQAELight(nn.Module):
    """VQ-Autoencoder with CWT preprocessing for EEG signals."""
    
    def __init__(self, config: VQAELightConfig | Dict):
        super().__init__()
        if isinstance(config, dict):
            config = VQAELightConfig(**config)
        
        self.config = config
        
        # CWT head
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
        
        # Encoder
        self.encoder_2d = Encoder2DStageLight(config)
        enc3d_in = self.encoder_2d.out_channels * self.encoder_2d.freq_out
        self.encoder_3d = Encoder3DStageLight(
            config, channels_in=enc3d_in, time_in=self.encoder_2d.time_out
        )
        
        # Vector quantizer
        self.vq = VectorQuantizerLight(
            config.codebook_size,
            config.embedding_dim,
            config.commitment_cost,
            config.ema_decay,
            config.epsilon
        )
        
        # Decoder
        self.decoder = SOTATimeFrequencyDecoder(config)
        
        # Initialize weights
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
        Args:
            x: (B_chunk, 2, F, 7, 5, T_chunk) - CWT output
        Returns:
            (B_chunk, embedding_dim)
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
        Args:
            z_q: (B_chunk, embedding_dim)
        Returns:
            (B_chunk, 32, 160)
        """
        return self.decoder(z_q)

    def forward(self, x):
        """
        Args:
            x: (B, 32, 640) - raw EEG

        Returns:
            dict with keys:
                'reconstruction': (B, 32, 640)
                'embeddings': (B_chunk, embedding_dim)
                'quantized': (B_chunk, embedding_dim)
                'indices': (B_chunk,)
                'vq_loss': scalar
                'perplexity': scalar
                'codebook_usage': scalar
        """
        # CWT transform
        x_cwt = self.cwt_head(x)  # (B_chunk, 2, F, 7, 5, T_chunk)

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

        # Unchunk back to full sequence
        recon = self.cwt_head._unchunk(recon_chunk)  # (B, 32, 640)

        cwt_rec = self.cwt_head(recon)

        return {
            'reconstruction': cwt_rec,
            'embeddings': z_e,
            'quantized': z_q,
            'indices': indices,
            **vq_losses,
            'target': x_cwt,
            'time_rec': recon
        }


if __name__ == "__main__":
    cfg = VQAELightConfig(
        use_quantizer=False,
        embedding_dim=64
    )
    model = VQAELight(cfg)
    x = torch.randn(2, 32, 640)
    
    with torch.no_grad():
        out = model(x)
    
    print("Input:", x.shape)
    print("Reconstruction:", out['reconstruction'].shape)
    print("Target:", out['target'].shape)
    print("Model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))