import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class VectorQuantizer(nn.Module):
    """EMA-based Vector Quantizer"""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", torch.randn(num_embeddings, embedding_dim))

    def forward(self, z):
        flat_z = z.view(-1, self.embedding_dim)
        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding ** 2, dim=1)
            - 2 * torch.matmul(flat_z, self.embedding.t())
        )
        indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(indices, self.num_embeddings).type(flat_z.dtype)
        z_q = torch.matmul(encodings, self.embedding)

        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                    (1 - self.decay) * torch.sum(encodings, dim=0)
            dw = torch.matmul(encodings.t(), flat_z)
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw
            n = torch.sum(self.ema_cluster_size)
            cluster_size = (
                (self.ema_cluster_size + self.eps)
                / (n + self.num_embeddings * self.eps)
                * n
            )
            self.embedding.data = self.ema_w / cluster_size.unsqueeze(1)

        loss = self.commitment_cost * torch.mean((z_q.detach() - flat_z) ** 2)
        z_q = flat_z + (z_q - flat_z).detach()
        z_q = z_q.view_as(z)
        return z_q, loss, indices
    
class DilatedTemporalSmoother(nn.Module):
    """
    Dilated 1D temporal convolutions for smooth temporal transitions.
    Uses large receptive field with dilations for continuity.
    """
    def __init__(self, channels, kernel_size=15):
        super().__init__()
        
        # Multiple dilated temporal convs for large receptive field
        self.temporal_conv1 = nn.Conv3d(
            channels, channels,
            kernel_size=(1, 1, kernel_size),  # Only temporal
            padding=(0, 0, kernel_size // 2 * 1),  # dilation=1
            dilation=(1, 1, 1),
            groups=channels,  # Depthwise
            bias=False
        )
        
        self.temporal_conv2 = nn.Conv3d(
            channels, channels,
            kernel_size=(1, 1, kernel_size),
            padding=(0, 0, kernel_size // 2 * 2),  # dilation=2
            dilation=(1, 1, 2),
            groups=channels,
            bias=False
        )
        
        self.temporal_conv3 = nn.Conv3d(
            channels, channels,
            kernel_size=(1, 1, kernel_size),
            padding=(0, 0, kernel_size // 2 * 4),  # dilation=4
            dilation=(1, 1, 4),
            groups=channels,
            bias=False
        )
        
        # Pointwise to mix
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        if channels % 8 == 0:
            self.norm = nn.GroupNorm(8, channels)
        else:
            for i in reversed(range(1, 8)):
                if channels % i == 0:
                    self.norm = nn.GroupNorm(i, channels)
                    break
        self.act = nn.GELU()
        
    def forward(self, x):
        # Sum dilated temporal convolutions
        t1 = self.temporal_conv1(x)
        t2 = self.temporal_conv2(x)
        t3 = self.temporal_conv3(x)
        
        x = t1 + t2 + t3  # Multi-scale temporal features
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ProgressiveTemporalUpsample(nn.Module):
    """
    Progressive temporal upsampling: upsample 2x, smooth, repeat.
    Much gentler than aggressive single-stage upsampling.
    """
    def __init__(self, channels, num_stages=2):
        super().__init__()
        self.num_stages = num_stages
        
        # Each stage: upsample 2x + smooth
        self.stages = nn.ModuleList([
            nn.Sequential(
                # Upsample temporal only (2x)
                nn.Upsample(scale_factor=(1, 1, 2), mode='trilinear', align_corners=False),
                # Smooth with dilated temporal convs
                DilatedTemporalSmoother(channels, kernel_size=15),
            )
            for _ in range(num_stages)
        ])
    
    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x
    
class DeepRefinementBlock(nn.Module):
    """
    Deep refinement block with residual connections.
    Multiple convolution layers for better feature refinement.
    """
    def __init__(self, channels, num_layers=3):
        super().__init__()
        ngroups = self._get_num_groups(channels)
        
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(ngroups, channels),
                nn.GELU()
            ])
        
        self.refinement = nn.Sequential(*layers)
        
        # Residual connection
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def _get_num_groups(self, channels):
        ngroups = min(8, channels)
        if channels % ngroups != 0:
            for g in reversed(range(1, ngroups + 1)):
                if channels % g == 0:
                    return g
        return ngroups
    
    def forward(self, x):
        return x + self.gamma * self.refinement(x)

class PowerfulProgressiveDecoderBlock(nn.Module):
    """
    Ultra-powerful decoder block with:
    - Deeper refinement networks
    - Larger channel capacity
    - Residual connections
    - Multi-scale processing
    """
    def __init__(self, in_channels, out_channels, current_shape, target_shape, 
                 spatial_upscale, temporal_upscale, use_skip=False, 
                 is_final=False, capacity_multiplier=2):
        super().__init__()
        self.use_skip = use_skip
        self.target_shape = target_shape
        self.spatial_upscale = spatial_upscale
        self.temporal_upscale = temporal_upscale
        self.is_final = is_final
        
        skip_channels = in_channels if use_skip else 0
        
        # Increase capacity - decoder uses MORE channels than encoder
        expanded_channels = out_channels * capacity_multiplier
        ngroups = self._get_num_groups(expanded_channels)
        ngroups_out = self._get_num_groups(out_channels)
        
        # Calculate groups for multi-scale refinement
        multi_scale_channels = expanded_channels // 2
        ngroups_multi = self._get_num_groups(multi_scale_channels)
        combined_channels = expanded_channels * 3 // 2
        ngroups_combined = self._get_num_groups(combined_channels)
        
        # STAGE 1: Initial expansion with deep refinement
        self.pre_conv = nn.Sequential(
            nn.Conv3d(in_channels + skip_channels, expanded_channels,
                     kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(ngroups, expanded_channels),
            nn.GELU(),
            DeepRefinementBlock(expanded_channels, num_layers=3)
        )
        
        # STAGE 2: Spatial upsampling with more capacity
        if spatial_upscale > 1:
            self.spatial_upsample = nn.Sequential(
                nn.Conv3d(expanded_channels, expanded_channels * (spatial_upscale ** 2),
                         kernel_size=3, padding=1, bias=False),
                nn.Upsample(scale_factor=(spatial_upscale, spatial_upscale, 1), 
                           mode='nearest'),
                nn.Conv3d(expanded_channels * (spatial_upscale ** 2), expanded_channels,
                         kernel_size=1, bias=False),
                nn.GroupNorm(ngroups, expanded_channels),
                nn.GELU(),
                DeepRefinementBlock(expanded_channels, num_layers=2)
            )
        else:
            self.spatial_upsample = nn.Identity()
        
        # STAGE 3: Progressive temporal upsampling with deep smoothing
        if temporal_upscale > 1:
            num_temporal_stages = int(math.log2(temporal_upscale))
            self.temporal_upsample = ProgressiveTemporalUpsample(
                expanded_channels, num_stages=num_temporal_stages
            )
        else:
            self.temporal_upsample = nn.Identity()
        
        # STAGE 4: Multi-scale refinement before final output
        self.multi_scale_refine = nn.ModuleList([
            # Large receptive field
            nn.Sequential(
                nn.Conv3d(expanded_channels, multi_scale_channels,
                         kernel_size=7, padding=3, bias=False),
                nn.GroupNorm(ngroups_multi, multi_scale_channels),  
                nn.GELU()
            ),
            # Medium receptive field
            nn.Sequential(
                nn.Conv3d(expanded_channels, multi_scale_channels,
                         kernel_size=5, padding=2, bias=False),
                nn.GroupNorm(ngroups_multi, multi_scale_channels),  
                nn.GELU()
            ),
            # Small receptive field
            nn.Sequential(
                nn.Conv3d(expanded_channels, multi_scale_channels,
                         kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(ngroups_multi, multi_scale_channels), 
                nn.GELU()
            ),
        ])
        
        # Combine multi-scale features
        self.combine_scales = nn.Sequential(
            nn.Conv3d(combined_channels, expanded_channels,
                     kernel_size=1, bias=False),
            nn.GroupNorm(ngroups, expanded_channels),
            nn.GELU()
        )
        
        # STAGE 5: Final deep refinement and channel reduction
        if is_final:
            self.final_refine = nn.Sequential(
                DeepRefinementBlock(expanded_channels, num_layers=4),
                nn.Conv3d(expanded_channels, out_channels, kernel_size=3, 
                         padding=1, bias=False),
                nn.GroupNorm(ngroups_out, out_channels),
                DeepRefinementBlock(out_channels, num_layers=3),
                # No activation on final output
                nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=True)
            )
        else:
            self.final_refine = nn.Sequential(
                DeepRefinementBlock(expanded_channels, num_layers=3),
                nn.Conv3d(expanded_channels, out_channels, kernel_size=3, 
                         padding=1, bias=False),
                nn.GroupNorm(ngroups_out, out_channels),
                nn.GELU(),
                DeepRefinementBlock(out_channels, num_layers=2)
            )
        
    def _get_num_groups(self, channels):
        """Calculate valid number of groups for GroupNorm."""
        ngroups = min(8, channels)
        if channels % ngroups != 0:
            for g in reversed(range(1, ngroups + 1)):
                if channels % g == 0:
                    return g
        return ngroups
    
    def forward(self, x, skip=None):
        # Skip connection
        if self.use_skip and skip is not None:
            x = torch.cat([x, skip], dim=1)
        
        # Stage 1: Initial expansion
        x = self.pre_conv(x)
        
        # Stage 2: Spatial upsampling
        x = self.spatial_upsample(x)
        
        # Stage 3: Temporal upsampling
        x = self.temporal_upsample(x)
        
        # Adjust to target size if needed
        current_shape = x.shape[2:]
        if current_shape != self.target_shape:
            x = F.interpolate(x, size=self.target_shape, mode='trilinear', align_corners=False)
        
        # Stage 4: Multi-scale refinement
        scales = [refine(x) for refine in self.multi_scale_refine]
        x_multi = torch.cat(scales, dim=1)
        x = x + self.combine_scales(x_multi)  # Residual
        
        # Stage 5: Final refinement
        x = self.final_refine(x)
        
        return x

class VQVAE(nn.Module):
    """
    VQ-VAE with asymmetric architecture:
    - Lightweight encoder (just compress)
    - POWERFUL decoder (reconstruct details)
    - Single embedding vector per chunk (for transformers)
    """
    def __init__(
        self,
        in_channels=25,
        input_spatial=(7, 5, 32),
        embedding_dim=128,
        codebook_size=512,
        num_downsample_stages=3,
        use_quantizer=True,
        use_skip_connections=False,
        decoder_capacity_multiplier=2
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.input_spatial = input_spatial
        self.embedding_dim = embedding_dim
        self.num_stages = num_downsample_stages
        self.use_quantizer = use_quantizer
        self.use_skip_connections = use_skip_connections
        self.decoder_capacity_multiplier = decoder_capacity_multiplier

        # Encoder configuration (lightweight)
        self.encoder_config = self._plan_encoder_stages(
            input_spatial, num_downsample_stages
        )
        
        # Build encoder (same as before - lightweight)
        self.encoder = self._build_encoder(in_channels, self.encoder_config)
        
        # Bottleneck
        final_shape = self.encoder_config[-1]['output_shape']
        final_channels = self.encoder_config[-1]['out_channels']
        self.flat_size = final_channels * math.prod(final_shape)
        self.final_shape = final_shape
        self.final_channels = final_channels
        
        # ✅ NEW: Flatten to single embedding vector
        self.to_embedding = nn.Sequential(
            nn.Flatten(),  # (B, C, H, W, T) -> (B, C*H*W*T)
            nn.Linear(self.flat_size, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        self.vq = VectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=embedding_dim
        )
        
        # ✅ NEW: Expand from single vector back to spatial
        self.from_embedding = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, self.flat_size)
        )
        
        # Build POWERFUL progressive decoder
        self.decoder_blocks = self._build_decoder(in_channels, self.encoder_config)
        
    def _plan_encoder_stages(self, input_spatial, num_stages):
        """Plan encoder stages (same as before)."""
        config = []
        current_shape = list(input_spatial)
        base_channels = 64
        
        for i in range(num_stages):
            out_channels = base_channels * (2 ** i)
            stride = []
            output_shape = []
            for dim_size in current_shape:
                s = 2 if dim_size >= 4 else 1
                stride.append(s)
                output_shape.append((dim_size + s - 1) // s)
            
            config.append({
                'in_channels': base_channels * (2 ** (i-1)) if i > 0 else None,
                'out_channels': out_channels,
                'input_shape': tuple(current_shape),
                'output_shape': tuple(output_shape),
                'stride': tuple(stride),
                'kernel_size': 3,
                'padding': 1
            })
            current_shape = output_shape
        
        return config
    
    def _build_encoder(self, in_channels, config):
        """Build encoder (lightweight)."""
        layers = []
        for i, stage in enumerate(config):
            in_ch = in_channels if i == 0 else config[i-1]['out_channels']
            out_ch = stage['out_channels']
            layers.extend([
                nn.Conv3d(in_ch, out_ch, kernel_size=stage['kernel_size'],
                         stride=stage['stride'], padding=stage['padding'], bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.GELU()
            ])
        return nn.Sequential(*layers)
    
    def _build_decoder(self, out_channels, encoder_config):
        """Build POWERFUL decoder."""
        blocks = nn.ModuleList()
        decoder_config = list(reversed(encoder_config))
        
        for i, stage in enumerate(decoder_config):
            if i == 0:
                in_ch = self.final_channels
                current_shape = self.final_shape
            else:
                in_ch = decoder_config[i-1]['in_channels']
                current_shape = decoder_config[i-1]['input_shape']
            
            out_ch = stage['in_channels'] if stage['in_channels'] is not None else out_channels
            if i == len(decoder_config) - 1:
                out_ch = out_channels
            
            target_shape = stage['input_shape']
            stride = stage['stride']
            
            spatial_upscale = stride[0]
            temporal_upscale = stride[2]
            is_final = (i == len(decoder_config) - 1)
            
            blocks.append(PowerfulProgressiveDecoderBlock(
                in_ch, out_ch, current_shape, target_shape,
                spatial_upscale, temporal_upscale,
                use_skip=self.use_skip_connections,
                is_final=is_final,
                capacity_multiplier=self.decoder_capacity_multiplier
            ))
        
        return blocks
    
    def _get_encoder_features(self, x):
        """Extract encoder features."""
        features = []
        z = x
        layer_idx = 0
        for i, stage_config in enumerate(self.encoder_config):
            stage_layers = self.encoder[layer_idx:layer_idx+3]
            for layer in stage_layers:
                z = layer(z)
            features.append(z)
            layer_idx += 3
        return features
    
    def encode(self, x):
        """
        Encode input to single embedding vector.
        
        Input: (B, in_channels, H, W, T)
        Output: (B, embedding_dim) - single vector per chunk
        """
        if self.use_skip_connections:
            encoder_features = self._get_encoder_features(x)
            z = encoder_features[-1]  # (B, C, H, W, T)
            self._encoder_features = encoder_features
        else:
            z = self.encoder(x)  # (B, C, H, W, T)
            self._encoder_features = None
        
        # ✅ Flatten to single vector
        z = self.to_embedding(z)  # (B, embedding_dim)

        if self.use_quantizer:
            z_q, vq_loss, indices = self.vq(z)  # Input: (B, embedding_dim)
        else:
            z_q = z
            vq_loss = torch.tensor(0., device=z.device)
            indices = None
        
        return z_q, vq_loss, indices
    
    def decode(self, z_q):
        """
        Decode from single embedding vector.
        
        Input: (B, embedding_dim) - single vector
        Output: (B, in_channels, H, W, T)
        """
        # ✅ Expand from single vector to spatial
        z = self.from_embedding(z_q)  # (B, flat_size)
        
        # Reshape to spatial feature map
        z = z.view(-1, self.final_channels, *self.final_shape)  # (B, C, H, W, T)
        
        encoder_features = self._encoder_features if hasattr(self, '_encoder_features') else None
        
        for i, block in enumerate(self.decoder_blocks):
            if self.use_skip_connections and encoder_features is not None:
                skip_idx = len(encoder_features) - 1 - i
                skip = encoder_features[skip_idx] if 0 <= skip_idx < len(encoder_features) else None
                z = block(z, skip)
            else:
                z = block(z, None)
        
        return z
    
    def forward(self, x):
        """Full forward pass."""
        z_q, vq_loss, indices = self.encode(x)
        x_recon = self.decode(z_q)
        
        if hasattr(self, '_encoder_features'):
            del self._encoder_features
        
        return x_recon, vq_loss, indices