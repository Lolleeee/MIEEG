import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from dataclasses import dataclass
import numpy as np
@dataclass
class VQAELightConfig:
    """Configuration for the VQ-VAE model."""
    use_quantizer: bool = True
    use_cwt: bool = True

    # CWT parameters
    cwt_frequencies: tuple = None
    chunk_samples: int = None  # If None, no chunking

    # Data shape parameters
    num_input_channels: int = 2   # Power + Phase
    num_freq_bands: int = 30
    spatial_rows: int = 7
    spatial_cols: int = 5
    time_samples: int = 80        # Fixed time window size
    orig_channels: int = 32       # Target output channels
    
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
            frequencies = np.concatenate([
                np.linspace(1, 4, 3), np.linspace(4, 8, 5)[1:], 
                np.linspace(8, 13, 8)[1:], np.linspace(13, 30, 10)[1:], 
                np.linspace(30, 80, 8)[1:]
            ])
            self.cwt_frequencies = tuple(frequencies)


# --- Utility Blocks ---

class SqueezeExcitation2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return x * self.se(x)

class SqueezeExcitation3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, reduced_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv3d(reduced_channels, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return x * self.se(x)

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_residual=False, use_se=False):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.se = SqueezeExcitation2D(out_channels) if use_se else nn.Identity()
    def forward(self, x):
        identity = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.se(x)
        return x + identity if self.use_residual else x

class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_residual=False, use_se=False):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.se = SqueezeExcitation3D(out_channels) if use_se else nn.Identity()
    def forward(self, x):
        identity = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.se(x)
        return x + identity if self.use_residual else x

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
        
        return quantized, encoding_indices, {
            'vq_loss': vq_loss, 'perplexity': perplexity, 
            'codebook_usage': (avg_probs > 0).sum().float() / self.num_embeddings
        }
    
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
                conv = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, use_se=config.use_squeeze_excitation)
            else:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            
            norm = nn.GroupNorm(min(config.num_groups, out_channels), out_channels) if config.use_group_norm else nn.BatchNorm2d(out_channels)
            se = SqueezeExcitation2D(out_channels) if config.use_squeeze_excitation and not (config.use_separable_conv and i > 0) else nn.Identity()
            
            layers.extend([conv, norm, nn.SiLU(inplace=True), se])
            if i < len(config.encoder_2d_channels) - 1:
                layers.append(nn.Dropout2d(p=config.dropout_2d))
                
            in_channels = out_channels
        
        self.conv_net = nn.Sequential(*layers)
        
        # Calculate output dimensions
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
                conv = DepthwiseSeparableConv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, use_se=config.use_squeeze_excitation)
            else:
                conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            
            norm = nn.GroupNorm(min(config.num_groups, out_channels), out_channels) if config.use_group_norm else nn.BatchNorm3d(out_channels)
            se = SqueezeExcitation3D(out_channels) if config.use_squeeze_excitation and not config.use_separable_conv else nn.Identity()
            
            layers.extend([conv, norm, nn.SiLU(inplace=True), se])
            if i < len(config.encoder_3d_channels) - 1:
                layers.append(nn.Dropout3d(p=config.dropout_3d))
                
            in_channels = out_channels
        
        self.conv_net = nn.Sequential(*layers)
        
        # Calculate final dimensions
        row_out = config.spatial_rows
        col_out = config.spatial_cols
        time_out = time_in
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
        x = self.projection(x)
        return x


class DecoderLight(nn.Module):
    def __init__(self, config: VQAELightConfig):
        super().__init__()
        self.config = config
        
        # Initial projection based on fixed time_samples
        init_time = config.time_samples // (2 ** len(config.decoder_channels))
        init_channels = config.decoder_channels[0]
        self.init_dim = init_channels * init_time
        
        self.projection = nn.Sequential(
            nn.Linear(config.embedding_dim, self.init_dim, bias=False),
            nn.LayerNorm(self.init_dim),
            nn.Dropout(p=0.1),
            nn.SiLU(inplace=True)
        )
        
        layers = []
        in_channels = init_channels
        
        for i, out_channels in enumerate(config.decoder_channels[1:]):
            layers.extend([
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(min(config.num_groups, out_channels), out_channels) if config.use_group_norm else nn.BatchNorm1d(out_channels),
                nn.SiLU(inplace=True)
            ])
            if i < len(config.decoder_channels[1:]) - 1:
                layers.append(nn.Dropout(p=config.dropout_decoder))
            in_channels = out_channels
        
        # Final layer
        layers.append(nn.ConvTranspose1d(in_channels, config.orig_channels, kernel_size=4, stride=2, padding=1))
        
        self.decoder_net = nn.Sequential(*layers)
    
    def forward(self, z_q):
        B = z_q.shape[0]
        x = self.projection(z_q)
        x = x.view(B, self.config.decoder_channels[0], -1)
        x = self.decoder_net(x)
        
        # Ensure exact output size match
        if x.shape[-1] != self.config.time_samples:
            x = F.interpolate(x, size=self.config.time_samples, mode='linear', align_corners=False)
        
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

        self.encoder_2d = Encoder2DStageLight(config)
        
        enc3d_in_channels = self.encoder_2d.out_channels * self.encoder_2d.freq_out
        self.encoder_3d = Encoder3DStageLight(config, channels_in=enc3d_in_channels, time_in=self.encoder_2d.time_out)
        
        self.vq = VectorQuantizerLight(
            num_embeddings=config.codebook_size, embedding_dim=config.embedding_dim,
            commitment_cost=config.commitment_cost, decay=config.ema_decay, epsilon=config.epsilon
        )
        
        self.decoder = DecoderLight(config)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None: nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
            if module.weight is not None: nn.init.constant_(module.weight, 1)
            if module.bias is not None: nn.init.constant_(module.bias, 0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, 2, F, R, C, T)
        B, Ch, F, R, C, T = x.shape

        # 1. Reshape for 2D Encoder: Process (Freq, Time) maps independently
        x_2d = x.permute(0, 3, 4, 1, 2, 5).reshape(B * R * C, Ch, F, T)
        x_2d = self.encoder_2d(x_2d)

        # Output: (B*R*C, C_out, F_out, T_out)
        C_out, F_out, T_out = x_2d.shape[1], x_2d.shape[2], x_2d.shape[3]
        
        # 2. Reshape for 3D Encoder: Combine spatial and frequency features
        x_3d = x_2d.view(B, R, C, C_out, F_out, T_out)
        x_3d = x_3d.permute(0, 3, 4, 1, 2, 5).reshape(B, C_out * F_out, R, C, T_out)

        return self.encoder_3d(x_3d)
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.use_cwt:
            x = self.cwt_head(x)  # (B, 2, F, 7, 5, T)
        z_e = self.encode(x)

        if self.config.use_quantizer:
            z_q, indices, vq_losses = self.vq(z_e)
        else:
            z_q = z_e
            indices = torch.zeros(z_e.shape[0], dtype=torch.long, device=z_e.device)
            vq_losses = {'vq_loss': torch.tensor(0.), 'perplexity': torch.tensor(0.), 'codebook_usage': torch.tensor(1.)}
        
        recon = self.decode(z_q)

        return {
            'reconstruction': recon,
            'embeddings': z_e,
            'quantized': z_q,
            'indices': indices,
            **vq_losses
        }


if __name__ == "__main__":
    print("="*60)
    print("DUAL-CHANNEL VQ-AE LIGHT MODEL TEST")
    print("="*60)
    
    config = VQAELightConfig(
        num_input_channels=2,
        num_freq_bands=30,
        spatial_rows=7,
        spatial_cols=5,
        time_samples=80,
        use_group_norm=True,
        use_squeeze_excitation=True,
        use_residual=True
    )
    
    model = VQAELight(config)
    x = torch.randn(2, 2, 30, 7, 5, 80)
    
    with torch.no_grad():
        out = model(x)
    
    print(f"\nInput shape:        {x.shape}")
    print(f"Output shape:       {out['reconstruction'].shape}")
    print(f"Embedding shape:    {out['embeddings'].shape}")
    print(f"Model Parameters:    {sum(p.numel() for p in model.parameters() if p.requires_grad)}    ")
    print(model)
