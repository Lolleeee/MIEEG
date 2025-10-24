import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


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
    """Dilated 1D temporal convolutions for smooth temporal transitions."""
    def __init__(self, channels, kernel_size=15):
        super().__init__()
        
        self.temporal_conv1 = nn.Conv3d(
            channels, channels, kernel_size=(1, 1, kernel_size),
            padding=(0, 0, (kernel_size - 1) * 1 // 2),  # = 7
            dilation=(1, 1, 1),
            groups=channels, bias=False
        )
        self.temporal_conv2 = nn.Conv3d(
            channels, channels, kernel_size=(1, 1, kernel_size),
            padding=(0, 0, (kernel_size - 1) * 2 // 2),  # = 14
            dilation=(1, 1, 2),
            groups=channels, bias=False
        )
        self.temporal_conv3 = nn.Conv3d(
            channels, channels, kernel_size=(1, 1, kernel_size),
            padding=(0, 0, (kernel_size - 1) * 4 // 2),  # = 28
            dilation=(1, 1, 4),
            groups=channels, bias=False
        )
        
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        ngroups = min(8, channels)
        if channels % ngroups != 0:
            for g in reversed(range(1, ngroups + 1)):
                if channels % g == 0:
                    ngroups = g
                    break
        self.norm = nn.GroupNorm(ngroups, channels)
        self.act = nn.GELU()
        
    def forward(self, x):
        t1 = self.temporal_conv1(x)
        t2 = self.temporal_conv2(x)
        t3 = self.temporal_conv3(x)
        x = t1 + t2 + t3
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ProgressiveTemporalUpsample(nn.Module):
    """Progressive temporal upsampling with smoothing."""
    def __init__(self, channels, num_stages=2):
        super().__init__()
        self.num_stages = num_stages
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=(1, 1, 2), mode='trilinear', align_corners=False),
                DilatedTemporalSmoother(channels, kernel_size=15),
            )
            for _ in range(num_stages)
        ])
    
    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


class ProgressiveDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, current_shape, target_shape, 
                 spatial_upscale, temporal_upscale, use_skip=False, skip_mode='concat',
                 skip_channels=None): 
        super().__init__()
        self.use_skip = use_skip
        self.target_shape = target_shape
        self.spatial_upscale = spatial_upscale
        self.temporal_upscale = temporal_upscale
        self.skip_mode = skip_mode
        
        # Determine skip channel handling
        if skip_mode == 'concat' and use_skip:
            # Use provided skip_channels or default to in_channels
            concat_skip_channels = skip_channels if skip_channels is not None else in_channels
        else:
            concat_skip_channels = 0
            if use_skip and skip_mode in ['add', 'weighted_add']:
                skip_ch = skip_channels if skip_channels is not None else in_channels
                self.skip_proj = nn.Conv3d(skip_ch, out_channels, kernel_size=1, bias=False)
        
        ngroups = self._get_num_groups(out_channels)
        
        # Initial conv
        self.pre_conv = nn.Sequential(
            nn.Conv3d(in_channels + concat_skip_channels, out_channels,
                     kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(ngroups, out_channels),
            nn.GELU()
        )
        
        # Spatial upsampling
        if spatial_upscale > 1:
            self.spatial_upsample = nn.Sequential(
                nn.Conv3d(out_channels, out_channels * (spatial_upscale ** 2),
                         kernel_size=3, padding=1, bias=False),
                nn.Upsample(scale_factor=(spatial_upscale, spatial_upscale, 1), mode='nearest'),
                nn.Conv3d(out_channels * (spatial_upscale ** 2), out_channels,
                         kernel_size=1, bias=False),
                nn.GroupNorm(ngroups, out_channels),
                nn.GELU()
            )
        else:
            self.spatial_upsample = nn.Identity()
        
        # Progressive temporal upsampling
        if temporal_upscale > 1:
            num_temporal_stages = int(math.log2(temporal_upscale))
            self.temporal_upsample = ProgressiveTemporalUpsample(
                out_channels, num_stages=num_temporal_stages
            )
        else:
            self.temporal_upsample = nn.Identity()
        
        # Final refinement
        self.refine = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(ngroups, out_channels),
            nn.GELU()
        )
        
    def _get_num_groups(self, channels):
        ngroups = min(8, channels)
        if channels % ngroups != 0:
            for g in reversed(range(1, ngroups + 1)):
                if channels % g == 0:
                    return g
        return ngroups
    
    def forward(self, x, skip=None, skip_weight=1.0):
        """Forward pass with optional skip connection."""
        
        # Process skip connection based on mode
        processed_skip = None
        
        if self.use_skip and skip is not None and skip_weight > 0:
            # Scale by weight
            processed_skip = skip * skip_weight
            
            if self.skip_mode == 'concat':
                # Concatenate before pre_conv
                x = torch.cat([x, processed_skip], dim=1)
            
            elif self.skip_mode in ['add', 'weighted_add']:
                # Project skip to match out_channels (only once!)
                if hasattr(self, 'skip_proj'):
                    processed_skip = self.skip_proj(processed_skip)
        
        # Initial processing
        x = self.pre_conv(x)
        
        # Add skip for additive modes (after pre_conv)
        if self.skip_mode in ['add', 'weighted_add'] and processed_skip is not None:
            x = x + processed_skip
        
        # Upsampling
        x = self.spatial_upsample(x)
        x = self.temporal_upsample(x)
        
        # Adjust to target size
        current_shape = x.shape[2:]
        if current_shape != self.target_shape:
            x = F.interpolate(x, size=self.target_shape, mode='trilinear', align_corners=False)
        
        # Final refinement
        x = self.refine(x)
        
        return x



class VQVAE(nn.Module):
    """
    VQ-VAE with controllable skip connections.
    
    Features:
    - Global skip strength control
    - Per-layer skip strength control
    - Different skip modes (concat, add, weighted_add)
    - Progressive temporal upsampling
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
        skip_strength=1.0,  # Global skip strength (0.0 to 1.0)
        skip_strengths=None,  # Per-layer skip strengths (list of floats)
        skip_mode='concat'  # 'concat', 'add', or 'weighted_add'
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.input_spatial = input_spatial
        self.embedding_dim = embedding_dim
        self.num_stages = num_downsample_stages
        self.use_quantizer = use_quantizer
        self.use_skip_connections = use_skip_connections
        self.skip_mode = skip_mode
        
        # Global skip strength
        self.register_buffer('skip_strength', torch.tensor(skip_strength))
        
        # Per-layer skip strengths
        if skip_strengths is None:
            self.skip_strengths = [1.0] * num_downsample_stages
        else:
            assert len(skip_strengths) == num_downsample_stages, \
                f"skip_strengths must have length {num_downsample_stages}"
            self.skip_strengths = skip_strengths
            
        self._embedding_initialized = False
        # Encoder configuration
        self.encoder_config = self._plan_encoder_stages(input_spatial, num_downsample_stages)
        
        # Build encoder
        self.encoder = self._build_encoder(in_channels, self.encoder_config)
        
        # Bottleneck
        final_shape = self.encoder_config[-1]['output_shape']
        final_channels = self.encoder_config[-1]['out_channels']
        self.flat_size = final_channels * math.prod(final_shape)
        self.final_shape = final_shape
        self.final_channels = final_channels
        
        # Embedding layers
        self.to_embedding = nn.Sequential(
            nn.Flatten(),
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
        
        self.from_embedding = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, self.flat_size)
        )
        
        # Build progressive decoder
        self.decoder_blocks = self._build_decoder(in_channels, self.encoder_config)
        
        # Output layer (no activation)
        self.output_layer = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=3, padding=1, bias=True
        )

    def _initialize_embedding_layers(self, encoder_output_shape):
        """Initialize embedding layers based on actual encoder output."""
        flat_size = math.prod(encoder_output_shape)
        
        self.to_embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, self.embedding_dim * 2),
            nn.LayerNorm(self.embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        ).to(next(self.encoder.parameters()).device)
        
        self.from_embedding = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.LayerNorm(self.embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim * 2, flat_size)
        ).to(next(self.encoder.parameters()).device)
        
        self._embedding_initialized = True
        self._actual_flat_size = flat_size

    def _plan_encoder_stages(self, input_spatial, num_stages):
        """Plan encoder stages."""
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
        """Build encoder."""
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
        """Build progressive decoder."""
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
            
            # ✅ NEW: Get skip's expected channels (from corresponding encoder stage)
            skip_idx = len(decoder_config) - 1 - i
            if 0 <= skip_idx < len(encoder_config):
                skip_ch = encoder_config[skip_idx]['out_channels']
            else:
                skip_ch = None
            
            target_shape = stage['input_shape']
            stride = stage['stride']
            
            spatial_upscale = stride[0]
            temporal_upscale = stride[2]
            
            blocks.append(ProgressiveDecoderBlock(
                in_ch, out_ch, current_shape, target_shape,
                spatial_upscale, temporal_upscale,
                use_skip=self.use_skip_connections,
                skip_mode=self.skip_mode,
                skip_channels=skip_ch  # ✅ Pass skip's actual channels
            ))
        
        return blocks
    
    def _get_encoder_features(self, x):
        """Extract encoder features for skip connections."""
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
    
    def set_skip_strength(self, strength):
        """
        Set global skip connection strength.
        
        Args:
            strength: Float between 0.0 (no skips) and 1.0 (full skips)
        """
        self.skip_strength = torch.clamp(torch.tensor(strength), 0.0, 1.0)
    
    def set_skip_strengths(self, strengths):
        """
        Set per-layer skip connection strengths.
        
        Args:
            strengths: List of floats (0.0 to 1.0) for each decoder layer
        """
        assert len(strengths) == len(self.skip_strengths), \
            f"Expected {len(self.skip_strengths)} strengths, got {len(strengths)}"
        self.skip_strengths = [float(np.clip(s, 0.0, 1.0)) for s in strengths]
    
    def get_skip_info(self):
        """Get current skip connection configuration."""
        return {
            'enabled': self.use_skip_connections,
            'mode': self.skip_mode,
            'global_strength': self.skip_strength.item(),
            'layer_strengths': self.skip_strengths
        }
    
    def encode(self, x):
        """Encode input."""
        if self.use_skip_connections:
            encoder_features = self._get_encoder_features(x)
            z = encoder_features[-1]
            self._encoder_features = encoder_features
        else:
            z = self.encoder(x)
            self._encoder_features = None
        
        # Initialize embedding layers on first forward pass
        if not self._embedding_initialized:
            
            self._initialize_embedding_layers(z.shape[1:])
            # Store actual shape for decoder
            self.final_shape = z.shape[2:]
            self.final_channels = z.shape[1]
        
        z = self.to_embedding(z)

        if self.use_quantizer:
            z_q, vq_loss, indices = self.vq(z)
        else:
            z_q, vq_loss, indices = z, torch.tensor(0., device=z.device), None
        
        return z_q, vq_loss, indices
    
    def decode(self, z_q):
        """Decode from embedding."""
        z = self.from_embedding(z_q)
        z = z.view(-1, self.final_channels, *self.final_shape)
        
        encoder_features = self._encoder_features if hasattr(self, '_encoder_features') else None
        
        for i, block in enumerate(self.decoder_blocks):
            skip = None
            skip_weight = 0.0
            
            if self.use_skip_connections and encoder_features is not None:
                skip_idx = len(encoder_features) - 1 - i
                if 0 <= skip_idx < len(encoder_features):
                    skip = encoder_features[skip_idx]
                    
                    # Compute total skip weight (global * per-layer)
                    skip_weight = self.skip_strength.item() * self.skip_strengths[i]
            
            z = block(z, skip, skip_weight)
        
        z = self.output_layer(z)

        return z
    
    def forward(self, x):
        """Full forward pass."""
        z_q, vq_loss, indices = self.encode(x)
        x_recon = self.decode(z_q)
        
        if hasattr(self, '_encoder_features'):
            del self._encoder_features
        
        return x_recon, vq_loss, indices


class SkipConnectionScheduler:
    """
    Scheduler for gradually adjusting skip connection strength during training.
    
    Strategies:
    - 'linear': Linear ramp from 0 to max_strength
    - 'cosine': Smooth cosine ramp
    - 'exponential': Fast early growth, slow later
    - 'step': Step increases at intervals
    """
    def __init__(self, model, schedule='linear', warmup_epochs=10, 
                 max_strength=1.0, min_strength=0.0):
        self.model = model
        self.schedule = schedule
        self.warmup_epochs = warmup_epochs
        self.max_strength = max_strength
        self.min_strength = min_strength
        self.current_epoch = 0
    
    def step(self, epoch=None):
        """
        Update skip strength for current epoch.
        
        Args:
            epoch: Current epoch number (if None, uses internal counter)
        
        Returns:
            Current skip strength
        """
        if epoch is None:
            epoch = self.current_epoch
            self.current_epoch += 1
        
        if epoch >= self.warmup_epochs:
            strength = self.max_strength
        else:
            progress = epoch / self.warmup_epochs
            
            if self.schedule == 'linear':
                strength = self.min_strength + progress * (self.max_strength - self.min_strength)
            
            elif self.schedule == 'cosine':
                strength = self.min_strength + 0.5 * (1 - math.cos(math.pi * progress)) * \
                          (self.max_strength - self.min_strength)
            
            elif self.schedule == 'exponential':
                strength = self.min_strength + (1 - math.exp(-5 * progress)) * \
                          (self.max_strength - self.min_strength)
            
            elif self.schedule == 'step':
                # Step increases every 1/3 of warmup
                if progress < 0.33:
                    strength = self.min_strength
                elif progress < 0.67:
                    strength = (self.min_strength + self.max_strength) / 2
                else:
                    strength = self.max_strength
            
            else:
                raise ValueError(f"Unknown schedule: {self.schedule}")
        
        self.model.set_skip_strength(strength)
        return strength
    
    def reset(self):
        """Reset epoch counter."""
        self.current_epoch = 0