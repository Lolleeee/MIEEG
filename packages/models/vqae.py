import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PixelShuffle3d(nn.Module):
    """3D Pixel Shuffle for upsampling."""
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        
    def forward(self, x):
        batch_size, channels, d, h, w = x.size()
        r = self.upscale_factor
        out_channels = channels // (r ** 3)
        x = x.view(batch_size, out_channels, r, r, r, d, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(batch_size, out_channels, d * r, h * r, w * r)
        return x


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


class ProgressiveDecoderBlock(nn.Module):
    """
    Decoder block with progressive temporal upsampling and spatial upsampling.
    Temporal: Multiple 2x stages with smoothing (gentle, continuous)
    Spatial: Single stage with nearest neighbor (can be blocky)
    """
    def __init__(self, in_channels, out_channels, current_shape, target_shape, 
                 spatial_upscale, temporal_upscale, use_skip=False):
        super().__init__()
        self.use_skip = use_skip
        self.target_shape = target_shape
        self.spatial_upscale = spatial_upscale
        self.temporal_upscale = temporal_upscale
        
        skip_channels = in_channels if use_skip else 0
        ngroups = self._get_num_groups(out_channels)
        
        # Initial conv
        self.pre_conv = nn.Sequential(
            nn.Conv3d(in_channels + skip_channels, out_channels,
                     kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(ngroups, out_channels),
            nn.GELU()
        )
        
        # Spatial upsampling (can be simple/blocky)
        if spatial_upscale > 1:
            self.spatial_upsample = nn.Sequential(
                # First, expand channels for spatial upsampling
                nn.Conv3d(out_channels, out_channels * (spatial_upscale ** 2),
                         kernel_size=3, padding=1, bias=False),
                # Spatial upsample only (not temporal)
                nn.Upsample(scale_factor=(spatial_upscale, spatial_upscale, 1), 
                           mode='nearest'),
                # Pointwise to reduce channels back to out_channels
                nn.Conv3d(out_channels * (spatial_upscale ** 2), out_channels,
                         kernel_size=1, bias=False),
                nn.GroupNorm(ngroups, out_channels),  # Now matches out_channels
                nn.GELU()
            )
        else:
            self.spatial_upsample = nn.Identity()
        
        # Progressive temporal upsampling (smooth, gradual)
        if temporal_upscale > 1:
            # Calculate number of 2x stages needed
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
    
    def forward(self, x, skip=None):
        # Skip connection
        if self.use_skip and skip is not None:
            x = torch.cat([x, skip], dim=1)
        
        # Initial processing
        x = self.pre_conv(x)
        
        # Spatial upsampling (fast, can be blocky)
        x = self.spatial_upsample(x)
        
        # Progressive temporal upsampling (slow, smooth)
        x = self.temporal_upsample(x)
        
        # Adjust to exact target size if needed
        current_shape = x.shape[2:]
        if current_shape != self.target_shape:
            x = F.interpolate(x, size=self.target_shape, mode='trilinear', align_corners=False)
        
        # Final refinement
        x = self.refine(x)
        
        return x


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


class VQVAE(nn.Module):
    """
    VQ-VAE with progressive temporal upsampling for smooth reconstructions.
    - Spatial: Single-stage upsampling (can be blocky)
    - Temporal: Multi-stage progressive upsampling (smooth, continuous)
    """
    def __init__(
        self,
        in_channels=25,
        input_spatial=(7, 5, 32),
        embedding_dim=128,
        codebook_size=512,
        num_downsample_stages=3,
        use_quantizer=True,
        use_skip_connections=False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.input_spatial = input_spatial
        self.embedding_dim = embedding_dim
        self.num_stages = num_downsample_stages
        self.use_quantizer = use_quantizer
        self.use_skip_connections = use_skip_connections

        # Encoder configuration
        self.encoder_config = self._plan_encoder_stages(
            input_spatial, num_downsample_stages
        )
        
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
            nn.Linear(embedding_dim * 2, self.flat_size)
        )
        
        # Build progressive decoder
        self.decoder_blocks = self._build_decoder(in_channels, self.encoder_config)
        
        self.output_layer = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=3, padding=1, bias=True  # bias=True for output layer
        )
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
        """Build progressive decoder with separate spatial/temporal upsampling."""
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
            
            # Separate spatial and temporal upscale factors
            spatial_upscale = stride[0]  # H, W (same stride)
            temporal_upscale = stride[2]  # T
            
            blocks.append(ProgressiveDecoderBlock(
                in_ch, out_ch, current_shape, target_shape,
                spatial_upscale, temporal_upscale,
                use_skip=self.use_skip_connections
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
        """Encode input."""
        if self.use_skip_connections:
            encoder_features = self._get_encoder_features(x)
            z = encoder_features[-1]
            self._encoder_features = encoder_features
        else:
            z = self.encoder(x)
            self._encoder_features = None
        
        z = self.to_embedding(z)

        if self.use_quantizer:
            z_q, vq_loss, indices = self.vq(z)
        else:
            z_q, vq_loss, indices = z, torch.tensor(0., device=z.device), None
        
        return z_q, vq_loss, indices
    
    def decode(self, z_q):
        """Decode with progressive upsampling."""
        z = self.from_embedding(z_q)
        z = z.view(-1, self.final_channels, *self.final_shape)
        
        encoder_features = self._encoder_features if hasattr(self, '_encoder_features') else None
        
        for i, block in enumerate(self.decoder_blocks):
            if self.use_skip_connections and encoder_features is not None:
                skip_idx = len(encoder_features) - 1 - i
                skip = encoder_features[skip_idx] if 0 <= skip_idx < len(encoder_features) else None
                z = block(z, skip)
            else:
                z = block(z, None)
        
        z = self.output_layer(z)

        return z
    
    def forward(self, x):
        """Full forward pass."""
        z_q, vq_loss, indices = self.encode(x)
        x_recon = self.decode(z_q)
        
        if hasattr(self, '_encoder_features'):
            del self._encoder_features
        print(f"reconstruction: {x_recon.shape}")
        return x_recon, vq_loss, indices
    
class SequenceProcessor(nn.Module):
    """
    Process full EEG window as sequence of chunks.
    Handles arbitrary chunk configurations.
    
    Example usage for (25, 7, 5, 250) -> (10, 25, 7, 5, 25):
        - Original: 25 channels, 7x5 spatial, 250 time samples
        - Chunked: 10 chunks, 25 channels each, 7x5 spatial, 25 samples per chunk
    """
    def __init__(
        self,
        chunk_shape=(25, 7, 5, 32),  # (C, H, W, T) per chunk
        embedding_dim=128,
        codebook_size=512,
        num_downsample_stages=3,
        use_quantizer=True
    ):
        super().__init__()
        
        self.chunk_shape = chunk_shape
        self.embedding_dim = embedding_dim
        self.use_quantizer = use_quantizer
        
        # Create chunk autoencoder
        self.chunk_ae = VQVAE(
            in_channels=chunk_shape[0],
            input_spatial=chunk_shape[1:],  # (H, W, T)
            embedding_dim=embedding_dim,
            codebook_size=codebook_size,
            num_downsample_stages=num_downsample_stages,
            use_quantizer=use_quantizer
        )
        
        # Positional embeddings (will be adjusted dynamically)
        self.max_chunks = 100  # Support up to 100 chunks
    
    def encode_sequence(self, chunks):
        """
        Encode sequence of chunks.
        Input: (B, num_chunks, C, H, W, T)
        Output: (B, num_chunks, embedding_dim), vq_loss, indices
        """
        batch_size, num_chunks = chunks.shape[:2]
        
        # Reshape to process all chunks
        chunks_flat = chunks.view(-1, *self.chunk_shape)
        
        # Encode each chunk
        embeddings, vq_loss, indices = self.chunk_ae.encode(chunks_flat)

        # Reshape back to sequence
        embeddings = embeddings.view(batch_size, num_chunks, self.embedding_dim)
        
        return embeddings, vq_loss, indices
    
    def decode_sequence(self, embeddings):
        """
        Decode sequence of embeddings back to chunks.
        Input: (B, num_chunks, embedding_dim)
        Output: (B, num_chunks, C, H, W, T)
        """
        batch_size, num_chunks = embeddings.shape[:2]
        
        # Reshape to decode all chunks
        embeddings_flat = embeddings.view(-1, self.embedding_dim)
        
        # Decode each chunk
        chunks_recon = self.chunk_ae.decode(embeddings_flat)

        # Reshape back to sequence
        chunks_recon = chunks_recon.view(batch_size, num_chunks, *self.chunk_shape)
        
        return chunks_recon
    
    def forward(self, chunks):
        """
        Full forward pass.
        Input: (B, num_chunks, C, H, W, T)
        Output: (reconstruction, vq_loss, indices)
        """
        embeddings, vq_loss, indices = self.encode_sequence(chunks)
        chunks_recon = self.decode_sequence(embeddings)
        return chunks_recon, vq_loss, indices, embeddings

