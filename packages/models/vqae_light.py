import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class VQVAELightConfig:
    """Lightweight configuration for the VQ-VAE model."""
    use_quantizer: bool = True
    
    # Data shape parameters
    num_freq_bands: int = 25
    spatial_rows: int = 7
    spatial_cols: int = 5
    time_samples: int = 250
    chunk_dim: int = 50
    orig_channels: int = 32
    
    # Encoder parameters - REDUCED
    encoder_2d_channels: list = None   # [16, 32] instead of [32, 64]
    encoder_3d_channels: list = None   # [32, 64] instead of [64, 128, 256]
    embedding_dim: int = 64            # 64 instead of 256
    
    # VQ parameters
    codebook_size: int = 256           # 256 instead of 512
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    epsilon: float = 1e-5
    
    # Decoder parameters - REDUCED
    decoder_channels: list = None      # [64, 32] instead of [256, 128, 64]
    
    # Dropout - REDUCED for lightweight model
    dropout_2d: float = 0.05
    dropout_3d: float = 0.05
    dropout_bottleneck: float = 0.1
    dropout_decoder: float = 0.05
    
    # Architecture improvements
    use_separable_conv: bool = True
    use_group_norm: bool = True       # GroupNorm instead of BatchNorm
    num_groups: int = 8               # Number of groups for GroupNorm
    use_residual: bool = True         # Add residual connections
    use_squeeze_excitation: bool = True  # Add SE blocks for channel attention
    
    def __post_init__(self):
        if self.encoder_2d_channels is None:
            self.encoder_2d_channels = [16, 32]
        if self.encoder_3d_channels is None:
            self.encoder_3d_channels = [32, 64]
        if self.decoder_channels is None:
            self.decoder_channels = [64, 32]
        
        self.num_chunks = self.time_samples // self.chunk_dim
        
        assert self.time_samples % self.chunk_dim == 0, \
            f"time_samples ({self.time_samples}) must be divisible by chunk_dim ({self.chunk_dim})"


class SqueezeExcitation2D(nn.Module):
    """
    Squeeze-and-Excitation block for 2D features.
    Learns channel-wise attention weights to emphasize important features.
    """
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
    
    def forward(self, x):
        return x * self.se(x)


class SqueezeExcitation3D(nn.Module):
    """Squeeze-and-Excitation block for 3D features."""
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
    
    def forward(self, x):
        return x * self.se(x)


class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise separable convolution with optional residual and SE block.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        use_residual=False,
        use_se=False
    ):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        
        self.se = SqueezeExcitation2D(out_channels) if use_se else nn.Identity()
        
    def forward(self, x):
        identity = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.se(x)
        
        if self.use_residual:
            x = x + identity
        
        return x


class DepthwiseSeparableConv3d(nn.Module):
    """3D depthwise separable convolution with optional residual and SE."""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        use_residual=False,
        use_se=False
    ):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)
        
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        
        self.se = SqueezeExcitation3D(out_channels) if use_se else nn.Identity()
        
    def forward(self, x):
        identity = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.se(x)
        
        if self.use_residual:
            x = x + identity
        
        return x


class VectorQuantizerLight(nn.Module):
    """Lightweight vector quantizer with EMA updates."""
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # Initialize codebook with better initialization
        self.register_buffer('embeddings', torch.randn(num_embeddings, embedding_dim))
        nn.init.xavier_uniform_(self.embeddings)
        
        # EMA buffers
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_embed_avg', self.embeddings.clone())
        
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Normalize for better codebook utilization
        flat_input_norm = F.normalize(flat_input, p=2, dim=1)
        embeddings_norm = F.normalize(self.embeddings, p=2, dim=1)
        
        # Efficient distance computation
        distances = (
            torch.sum(flat_input_norm ** 2, dim=1, keepdim=True)
            + torch.sum(embeddings_norm ** 2, dim=1)
            - 2 * torch.matmul(flat_input_norm, embeddings_norm.t())
        )
        
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(encoding_indices, self.embeddings)
        
        if self.training:
            self._ema_update(flat_input, encoding_indices)
        
        # Losses
        e_latent_loss = F.mse_loss(quantized.detach(), flat_input)
        quantized = flat_input + (quantized - flat_input).detach()  # Straight-through estimator
        q_latent_loss = F.mse_loss(quantized, flat_input.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = quantized.view(input_shape)
        
        # Codebook usage metrics
        avg_probs = torch.bincount(
            encoding_indices, minlength=self.num_embeddings
        ).float() / len(encoding_indices)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        loss_dict = {
            'vq_loss': vq_loss,
            'commitment_loss': e_latent_loss,
            'codebook_loss': q_latent_loss,
            'perplexity': perplexity,
            'codebook_usage': (avg_probs > 0).sum().float() / self.num_embeddings
        }
        
        return quantized, encoding_indices, loss_dict
    
    def _ema_update(self, flat_input: torch.Tensor, encoding_indices: torch.Tensor):
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


class Encoder2DStageLight(nn.Module):
    """Supercharged 2D encoder with GroupNorm, residuals, and SE blocks."""
    
    def __init__(self, config: VQVAELightConfig):
        super().__init__()
        self.config = config
        
        layers = []
        in_channels = 1
        
        for i, out_channels in enumerate(config.encoder_2d_channels):
            # Convolution
            if config.use_separable_conv and i > 0:
                conv = DepthwiseSeparableConv2d(
                    in_channels, out_channels,
                    kernel_size=3, stride=2, padding=1,
                    use_residual=False,  # Can't use residual when stride=2
                    use_se=config.use_squeeze_excitation
                )
            else:
                conv = nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=3, stride=2, padding=1, bias=False
                )
            
            # Normalization
            if config.use_group_norm:
                # Ensure num_groups divides out_channels
                num_groups = min(config.num_groups, out_channels)
                while out_channels % num_groups != 0:
                    num_groups -= 1
                norm = nn.GroupNorm(num_groups, out_channels)
            else:
                norm = nn.BatchNorm2d(out_channels)
            
            # SE block (if not already in separable conv)
            se = nn.Identity()
            if config.use_squeeze_excitation and not (config.use_separable_conv and i > 0):
                se = SqueezeExcitation2D(out_channels)
            
            layers.extend([
                conv,
                norm,
                nn.SiLU(inplace=True),
                se,
                nn.Dropout2d(p=config.dropout_2d) if i < len(config.encoder_2d_channels) - 1 else nn.Identity()
            ])
            in_channels = out_channels
        
        self.conv_net = nn.Sequential(*layers)
        
        # Calculate output dimensions
        self.freq_out = config.num_freq_bands
        self.time_out = config.chunk_dim
        for _ in config.encoder_2d_channels:
            self.freq_out = (self.freq_out + 1) // 2
            self.time_out = (self.time_out + 1) // 2
        
        self.out_channels = config.encoder_2d_channels[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_net(x)


class Encoder3DStageLight(nn.Module):
    """Supercharged 3D encoder with GroupNorm, residuals, and SE blocks."""
    
    def __init__(self, config: VQVAELightConfig, channels_in: int, time_in: int, **kwargs):
        super().__init__()
        self.config = config
        
        layers = []
        in_channels = channels_in
        
        for i, out_channels in enumerate(config.encoder_3d_channels):
            # Convolution
            if config.use_separable_conv:
                conv = DepthwiseSeparableConv3d(
                    in_channels, out_channels,
                    kernel_size=3, stride=2, padding=1,
                    use_residual=False,
                    use_se=config.use_squeeze_excitation
                )
            else:
                conv = nn.Conv3d(
                    in_channels, out_channels,
                    kernel_size=3, stride=2, padding=1, bias=False
                )
            
            # Normalization
            if config.use_group_norm:
                num_groups = min(config.num_groups, out_channels)
                while out_channels % num_groups != 0:
                    num_groups -= 1
                norm = nn.GroupNorm(num_groups, out_channels)
            else:
                norm = nn.BatchNorm3d(out_channels)
            
            # SE block
            se = nn.Identity()
            if config.use_squeeze_excitation and not config.use_separable_conv:
                se = SqueezeExcitation3D(out_channels)
            
            layers.extend([
                conv,
                norm,
                nn.SiLU(inplace=True),
                se,
                nn.Dropout3d(p=config.dropout_3d) if i < len(config.encoder_3d_channels) - 1 else nn.Identity()
            ])
            in_channels = out_channels
        
        self.conv_net = nn.Sequential(*layers)
        
        # Calculate output dimensions
        row_out = config.spatial_rows
        col_out = config.spatial_cols
        time_out = time_in
        
        for _ in config.encoder_3d_channels:
            row_out = (row_out + 1) // 2
            col_out = (col_out + 1) // 2
            time_out = (time_out + 1) // 2
        
        flatten_dim = config.encoder_3d_channels[-1] * row_out * col_out * time_out
        
        # Projection with LayerNorm
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=config.dropout_bottleneck),
            nn.Linear(flatten_dim, config.embedding_dim, bias=False),
            nn.LayerNorm(config.embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_net(x)
        x = self.projection(x)
        return x


class DecoderLight(nn.Module):
    """Supercharged decoder with GroupNorm."""
    
    def __init__(self, config: VQVAELightConfig):
        super().__init__()
        self.config = config
        
        # Initial projection
        init_time = config.chunk_dim // (2 ** len(config.decoder_channels))
        init_channels = config.decoder_channels[0]
        self.init_dim = init_channels * init_time
        
        self.projection = nn.Sequential(
            nn.Linear(config.embedding_dim, self.init_dim, bias=False),
            nn.LayerNorm(self.init_dim),
            nn.Dropout(p=0.1),
            nn.SiLU(inplace=True)
        )
        
        # Transposed convolutions
        layers = []
        in_channels = init_channels
        
        for i, out_channels in enumerate(config.decoder_channels[1:]):
            layers.extend([
                nn.ConvTranspose1d(
                    in_channels, out_channels,
                    kernel_size=3, stride=2, padding=1, bias=False
                ),
                nn.GroupNorm(min(config.num_groups, out_channels), out_channels) if config.use_group_norm else nn.BatchNorm1d(out_channels),
                nn.SiLU(inplace=True),
                nn.Dropout(p=config.dropout_decoder) if i < len(config.decoder_channels[1:]) - 1 else nn.Identity()
            ])
            in_channels = out_channels
        
        # Final layer
        layers.append(
            nn.ConvTranspose1d(
                in_channels, config.orig_channels,
                kernel_size=4, stride=2, padding=1
            )
        )
        
        self.decoder_net = nn.Sequential(*layers)
    
    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        batch_size = z_q.shape[0]
        
        x = self.projection(z_q)
        x = x.view(batch_size, self.config.decoder_channels[0], -1)
        x = self.decoder_net(x)
        
        if x.shape[-1] != self.config.chunk_dim:
            x = F.interpolate(x, size=self.config.chunk_dim, mode='linear', align_corners=False)
        
        return x


class VQAELight(nn.Module):
    """
    Supercharged lightweight VQ-VAE with:
    - GroupNorm for better generalization
    - Squeeze-Excitation blocks for channel attention
    - Residual connections where applicable
    - Better weight initialization
    - Normalized embeddings in VQ
    """
    
    def __init__(self, config: VQVAELightConfig | dict):
        super().__init__()
        
        if isinstance(config, dict):
            config = VQVAELightConfig(**config)
        elif not isinstance(config, VQVAELightConfig):
            raise TypeError(f"config must be VQVAELightConfig or dict, got {type(config)}")
        
        self.config = config
        
        # Encoder stages
        self.encoder_2d = Encoder2DStageLight(config)
        self.encoder_3d = Encoder3DStageLight(
            config,
            time_in=self.encoder_2d.time_out,
            channels_in=self.encoder_2d.out_channels * self.encoder_2d.freq_out
        )
        
        # Vector quantization
        self.vq = VectorQuantizerLight(
            num_embeddings=config.codebook_size,
            embedding_dim=config.embedding_dim,
            commitment_cost=config.commitment_cost,
            decay=config.ema_decay,
            epsilon=config.epsilon
        )
        
        # Decoder
        self.decoder = DecoderLight(config)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Better weight initialization."""
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
            if module.weight is not None:
                nn.init.constant_(module.weight, 1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, F, R, C, T = x.shape
        
        # Chunk time
        x_chunked = self._chunk_time(x)
        
        # 2D encoder
        BnC = x_chunked.shape[0]
        x_2d = x_chunked.permute(0, 2, 3, 1, 4)
        x_2d = x_2d.reshape(BnC * R * C, 1, F, self.config.chunk_dim)
        x_2d = self.encoder_2d(x_2d)
        
        # Reshape for 3D encoder
        C_out, F_out, T_out = (
            self.encoder_2d.out_channels,
            self.encoder_2d.freq_out,
            self.encoder_2d.time_out
        )
        x_3d = x_2d.view(BnC, R, C, C_out*F_out, T_out)
        x_3d = x_3d.permute(0, 3, 1, 2, 4)
        
        # 3D encoder
        z_e = self.encoder_3d(x_3d)
        
        return z_e
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode
        z_e = self.encode(x)
        
        # Quantize
        if self.config.use_quantizer:
            z_q, indices, vq_losses = self.vq(z_e)
        else:
            z_q = z_e
            indices = torch.zeros(z_e.shape[0], dtype=torch.long, device=z_e.device)
            vq_losses = {
                'vq_loss': torch.tensor(0.0, device=z_e.device),
                'perplexity': torch.tensor(0.0, device=z_e.device),
                'codebook_usage': torch.tensor(1.0, device=z_e.device)
            }
        
        # Decode
        recon = self.decode(z_q)
        recon = recon.reshape(-1, self.config.orig_channels, self.config.time_samples)
        
        return {
            'reconstruction': recon,
            'embeddings': z_e,
            'quantized': z_q,
            'indices': indices,
            **vq_losses
        }
    
    def _chunk_time(self, x: torch.Tensor) -> torch.Tensor:
        B, F, R, C, T = x.shape
        nChunk = self.config.num_chunks
        ChunkDim = self.config.chunk_dim
        
        x = x.reshape(B, F, R, C, nChunk, ChunkDim)
        x = x.permute(0, 4, 1, 2, 3, 5)
        x = x.reshape(B * nChunk, F, R, C, ChunkDim)
        
        return x


if __name__ == "__main__":
    print("="*60)
    print("SUPERCHARGED VQ-VAE LIGHT")
    print("="*60)
    
    # Configuration
    config = VQVAELightConfig(
        use_group_norm=True,
        use_squeeze_excitation=True,
        use_residual=True,
        use_separable_conv=True
    )
    
    model = VQAELight(config)
    params = sum(p.numel() for p in model.parameters())
    
    print(f"\nTotal parameters: {params:,}")
    print(f"Embedding dim: {config.embedding_dim}")
    print(f"Codebook size: {config.codebook_size}")
    print(f"GroupNorm: {config.use_group_norm}")
    print(f"SE blocks: {config.use_squeeze_excitation}")
    print(f"Separable conv: {config.use_separable_conv}")
    
    # Test forward pass
    print("\n" + "="*60)
    print("FORWARD PASS TEST")
    print("="*60)
    
    x = torch.randn(2, 25, 7, 5, 160)
    
    with torch.no_grad():
        out = model(x)
    
    print(f"\nInput shape:        {x.shape}")
    print(f"Output shape:       {out['reconstruction'].shape}")
    print(f"Embedding shape:    {out['embeddings'].shape}")
    print(f"VQ loss:            {out['vq_loss'].item():.4f}")
    print(f"Perplexity:         {out['perplexity'].item():.2f}")
    print(f"Codebook usage:     {out['codebook_usage'].item():.2%}")
    
    # Model summary
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    print(model)