import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class VQAE23Config:
    use_quantizer: bool = False
    use_cwt: bool = True

    # CWT parameters
    cwt_frequencies: tuple = None
    chunk_samples: int = 160
    use_log_compression = True
    normalize_outputs = True
    learnable_norm = True
    
    # Data shape parameters
    num_input_channels: int = 2
    num_freq_bands: int = 25
    spatial_rows: int = 7
    spatial_cols: int = 5        
    time_samples: int = 160
    orig_channels: int = 32      

    # Encoder parameters
    encoder_2d_channels: list = None   # [16, 32]
    encoder_3d_channels: list = None   # [32, 64]
    embedding_dim: int = 128  # FIXED: Increased from 64 to reduce compression

    # VQ parameters
    codebook_size: int = 256
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    epsilon: float = 1e-5
    
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
    use_refinement_heads: bool = False  # FIXED: Made optional (default off)
    
    def __post_init__(self):
        if self.encoder_2d_channels is None:
            self.encoder_2d_channels = [16, 32]
        if self.encoder_3d_channels is None:
            self.encoder_3d_channels = [32, 64]
        if self.cwt_frequencies is None:
            frequencies = np.logspace(np.log10(0.5), np.log10(79.9), 25)
            self.cwt_frequencies = tuple(frequencies)


class VectorQuantizer(nn.Module):
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

        # FIXED: Use direct cosine distance (faster, cleaner)
        flat_input_norm = F.normalize(flat_input, p=2, dim=1)
        embeddings_norm = F.normalize(self.embeddings, p=2, dim=1)
        
        # Cosine distance: 1 - cosine_similarity
        distances = 1.0 - torch.matmul(flat_input_norm, embeddings_norm.t())

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


class Encoder2DStage(nn.Module):
    def __init__(self, config: VQAE23Config):
        super().__init__()
        self.config = config
        layers = []
        in_channels = config.num_input_channels

        for i, out_channels in enumerate(config.encoder_2d_channels):
            if config.use_separable_conv and i > 0:
                # FIXED: Disable SE when using separable conv (already has SE inside)
                conv = DepthwiseSeparableConv2d(
                    in_channels, out_channels, 3, 2, 1,
                    use_se=False  # Consistent with 3D encoder
                )
            else:
                conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False)

            if config.use_group_norm:
                norm = nn.GroupNorm(min(config.num_groups, out_channels), out_channels)
            else:
                norm = nn.BatchNorm2d(out_channels)

            # FIXED: Only add SE for regular conv, not separable
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


class Encoder3DStage(nn.Module):
    def __init__(self, config: VQAE23Config, channels_in: int, time_in: int):
        super().__init__()
        self.config = config
        layers = []
        in_channels = channels_in

        for i, out_channels in enumerate(config.encoder_3d_channels):
            if config.use_separable_conv:
                conv = DepthwiseSeparableConv3d(
                    in_channels, out_channels, 3, 2, 1,
                    use_se=False  # SE inside separable conv
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


class SymmetricalDecoder3D(nn.Module):
    """Symmetrical 3D→2D decoder with computed target shapes."""
    
    def __init__(self, config: VQAE23Config):
        super().__init__()
        self.config = config
        
        # Calculate encoder output dimensions
        enc_3d_out_channels = config.encoder_3d_channels[-1]
        enc_2d_out_channels = config.encoder_2d_channels[-1]
        
        # FIXED: Compute encoder output dimensions using SAME logic as encoder
        freq_sizes_2d = [config.num_freq_bands]
        time_sizes_2d = [config.time_samples]
        
        for _ in config.encoder_2d_channels:
            freq_sizes_2d.append((freq_sizes_2d[-1] + 1) // 2)
            time_sizes_2d.append((time_sizes_2d[-1] + 1) // 2)
        
        enc_2d_freq_out = freq_sizes_2d[-1]  # 7
        enc_2d_time_out = time_sizes_2d[-1]  # 40
        
        # Compute 3D encoder output spatial dims
        spatial_sizes_3d = [(config.spatial_rows, config.spatial_cols, enc_2d_time_out)]
        
        for _ in config.encoder_3d_channels:
            h, w, t = spatial_sizes_3d[-1]
            spatial_sizes_3d.append(((h + 1) // 2, (w + 1) // 2, (t + 1) // 2))
        
        # Store computed dimensions
        self.enc_3d_spatial = spatial_sizes_3d[-1]  # (2, 2, 10)
        self.enc_2d_dims = (enc_2d_freq_out, enc_2d_time_out)  # (7, 40)
        
        # FIXED: Decoder target shapes = REVERSE of encoder progression
        # 3D decoder: reverse the spatial_sizes_3d list (excluding the last one which is the bottleneck)
        self.target_shapes_3d = list(reversed(spatial_sizes_3d[:-1]))  # [(4, 3, 20), (7, 5, 40)]
        
        # 2D decoder: reverse the freq/time sizes (excluding the last one which is bottleneck)
        freq_targets = list(reversed(freq_sizes_2d[:-1]))  # [13, 25]
        time_targets = list(reversed(time_sizes_2d[:-1]))  # [80, 160]
        self.target_shapes_2d = list(zip(freq_targets, time_targets))  # [(13, 80), (25, 160)]
        
        # ===== BOTTLENECK EXPANSION =====
        bottleneck_spatial = self.enc_3d_spatial
        bottleneck_channels = enc_3d_out_channels
        bottleneck_size = bottleneck_channels * np.prod(bottleneck_spatial)
        
        self.bottleneck_expand = nn.Sequential(
            nn.Linear(config.embedding_dim, bottleneck_size),
            nn.LayerNorm(bottleneck_size),
            nn.GELU()
        )
        self.bottleneck_shape = (bottleneck_channels, *bottleneck_spatial)
        
        # ===== 3D DECODER =====
        self.decoder_3d_layers = nn.ModuleList()
        self.decoder_3d_norms = nn.ModuleList()
        self.decoder_3d_ses = nn.ModuleList()
        self.decoder_3d_residuals = nn.ModuleList()
        
        decoder_3d_channels = list(reversed(config.encoder_3d_channels))
        decoder_3d_channels[-1] = enc_2d_out_channels * enc_2d_freq_out
        in_ch = bottleneck_channels
        
        for i, out_ch in enumerate(decoder_3d_channels):
            conv = nn.ConvTranspose3d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False
            )
            norm = nn.GroupNorm(min(config.num_groups, out_ch), out_ch)
            se = SqueezeExcitation3D(out_ch, reduction=4)
            
            residual = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                nn.Conv3d(in_ch, out_ch, 1, bias=False)
            )
            
            self.decoder_3d_layers.append(conv)
            self.decoder_3d_norms.append(norm)
            self.decoder_3d_ses.append(se)
            self.decoder_3d_residuals.append(residual)
            
            in_ch = out_ch
        
        # ===== 2D DECODER =====
        self.decoder_2d_layers = nn.ModuleList()
        self.decoder_2d_norms = nn.ModuleList()
        self.decoder_2d_ses = nn.ModuleList()
        self.decoder_2d_residuals = nn.ModuleList()
        
        decoder_2d_channels = list(reversed(config.encoder_2d_channels))
        in_ch_2d = enc_2d_out_channels
        
        for i, out_ch in enumerate(decoder_2d_channels):
            if i == len(decoder_2d_channels) - 1:
                out_ch = config.num_input_channels
            
            conv = nn.ConvTranspose2d(
                in_ch_2d, out_ch, kernel_size=4, stride=2, padding=1, bias=False
            )
            norm = nn.GroupNorm(min(config.num_groups, out_ch), out_ch) if out_ch > 2 else nn.Identity()
            se = SqueezeExcitation2D(out_ch, reduction=4) if out_ch >= 16 else nn.Identity()
            
            residual = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch_2d, out_ch, 1, bias=False)
            )
            
            self.decoder_2d_layers.append(conv)
            self.decoder_2d_norms.append(norm)
            self.decoder_2d_ses.append(se)
            self.decoder_2d_residuals.append(residual)
            
            in_ch_2d = out_ch
        
        # FIXED: Optional refinement heads
        if config.use_refinement_heads:
            self.magnitude_refine = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(8, 1, 1),
                nn.Softplus()
            )
            self.phase_refine = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(8, 1, 1),
                nn.Tanh()
            )
        else:
            self.magnitude_refine = None
            self.phase_refine = None
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z_q, encoder_features=None):
        Bc = z_q.shape[0]
        
        # Bottleneck expansion
        x = self.bottleneck_expand(z_q)
        x = x.view(Bc, *self.bottleneck_shape)
        
        # 3D Decoder
        for i, (conv, norm, se, residual) in enumerate(zip(
            self.decoder_3d_layers, self.decoder_3d_norms, 
            self.decoder_3d_ses, self.decoder_3d_residuals
        )):
            x_res = residual(x)
            x = conv(x)
            
            target = self.target_shapes_3d[i]
            if x.shape[2:] != target:
                x = F.interpolate(x, size=target, mode='trilinear', align_corners=False)
            if x_res.shape[2:] != target:
                x_res = F.interpolate(x_res, size=target, mode='trilinear', align_corners=False)
            
            x = norm(x)
            x = F.gelu(x)
            x = se(x)
            x = x + x_res
        
        # Reshape for 2D decoder
        Bc, C, H, W, T = x.shape
        x = x.permute(0, 2, 3, 1, 4).reshape(Bc * H * W, C // self.enc_2d_dims[0], self.enc_2d_dims[0], T)
        
        # 2D Decoder
        for i, (conv, norm, se, residual) in enumerate(zip(
            self.decoder_2d_layers, self.decoder_2d_norms,
            self.decoder_2d_ses, self.decoder_2d_residuals
        )):
            x_res = residual(x)
            x = conv(x)
            
            target = self.target_shapes_2d[i]
            if x.shape[2:] != target:
                x = F.interpolate(x, size=target, mode='bilinear', align_corners=False)
            if x_res.shape[2:] != target:
                x_res = F.interpolate(x_res, size=target, mode='bilinear', align_corners=False)
            
            x = norm(x)
            x = F.gelu(x)
            x = se(x)
            x = x + x_res
        
        # Optional refinement
        if self.magnitude_refine is not None:
            magnitude = self.magnitude_refine(x[:, 0:1, :, :])
            phase = self.phase_refine(x[:, 1:2, :, :])
            x = torch.cat([magnitude, phase], dim=1)
        
        # Final reshape
        x = x.view(Bc, self.config.spatial_rows, self.config.spatial_cols, 
                   self.config.num_input_channels, self.config.num_freq_bands, self.config.time_samples)
        x = x.permute(0, 3, 4, 1, 2, 5)
        
        return x


class VQAE23(nn.Module):
    """Optimized VQAE with fixes applied."""
    
    def __init__(self, config: VQAE23Config | Dict):
        super().__init__()
        if isinstance(config, dict):
            config = VQAE23Config(**config)
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
        
        self.encoder_2d = Encoder2DStage(config)
        enc3d_in = self.encoder_2d.out_channels * self.encoder_2d.freq_out
        self.encoder_3d = Encoder3DStage(
            config, channels_in=enc3d_in, time_in=self.encoder_2d.time_out
        )
        
        self.vq = VectorQuantizer(
            config.codebook_size,
            config.embedding_dim,
            config.commitment_cost,
            config.ema_decay,
            config.epsilon
        )
        
        self.decoder = SymmetricalDecoder3D(config)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
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
        Bc, Ch, F, R, C, T = x.shape
        x = x.permute(0, 3, 4, 1, 2, 5).reshape(Bc * R * C, Ch, F, T)
        x = self.encoder_2d(x)

        C_out, F_out, T_out = x.shape[1], x.shape[2], x.shape[3]
        x = x.view(Bc, R, C, C_out, F_out, T_out)
        x = x.permute(0, 3, 4, 1, 2, 5).reshape(Bc, C_out * F_out, R, C, T_out)

        return self.encoder_3d(x)

    def decode(self, z_q):
        return self.decoder(z_q)

    def forward(self, x):
        if self.use_cwt:
            x_cwt = self.cwt_head(x)
        else:
            raise ValueError("This version assumes use_cwt=True")

        z_e = self.encode(x_cwt)

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

        recon_cwt = self.decode(z_q)

        Bc, C, F, H, W, T = recon_cwt.shape
        recon_flat = recon_cwt.reshape(Bc, C * F * H * W, T)
        target_flat = x_cwt.reshape(Bc, C * F * H * W, T)
        
        if self.use_cwt and self.chunk_samples is not None:
            recon = self.cwt_head._unchunk(recon_flat)
            target = self.cwt_head._unchunk(target_flat)
        else:
            recon = recon_flat
            target = target_flat

        return {
            'reconstruction': recon,
            'target': target,
            'embeddings': z_e,
            'quantized': z_q,
            'indices': indices,
            **vq_losses
        }
    
if __name__ == "__main__":
    config = VQAE23Config()
    model = VQAE23(config)
    dummy_input = torch.randn(2, config.orig_channels, config.chunk_samples * 4)
    print(dummy_input.shape)
    outputs = model(dummy_input)
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")
        else:
            print(f"{k}: {v}")
    
    print("model patameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))