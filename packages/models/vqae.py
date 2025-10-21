import torch
import torch.nn as nn
import math
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PixelShuffle3d(nn.Module):
    """
    3D Pixel Shuffle for upsampling.
    Rearranges elements in a tensor from (B, C*r^3, D, H, W) to (B, C, D*r, H*r, W*r).
    """
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        
    def forward(self, x):
        batch_size, channels, d, h, w = x.size()
        r = self.upscale_factor
        
        # Calculate output channels
        out_channels = channels // (r ** 3)
        
        # Reshape: (B, C_out, r, r, r, D, H, W)
        x = x.view(batch_size, out_channels, r, r, r, d, h, w)
        
        # Permute to interleave: (B, C_out, D, r, H, r, W, r)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        
        # Collapse to final shape: (B, C_out, D*r, H*r, W*r)
        x = x.view(batch_size, out_channels, d * r, h * r, w * r)
        
        return x


class DecoderBlock(nn.Module):
    """
    Hybrid decoder block: PixelShuffle + adaptive interpolation.
    Handles non-uniform spatial dimensions gracefully.
    """
    def __init__(self, in_channels, out_channels, target_shape, upscale_factor, use_skip=False):
        super().__init__()
        self.use_skip = use_skip
        self.upscale_factor = upscale_factor
        self.target_shape = target_shape  # Exact target (H, W, T)
        
        # Calculate number of groups for GroupNorm
        skip_channels = in_channels if use_skip else 0
        ngroups = self._get_num_groups(out_channels)
        
        if upscale_factor > 1:
            # PixelShuffle upsampling for learnable features
            self.upsample = nn.Sequential(
                nn.Conv3d(in_channels + skip_channels, 
                         out_channels * (upscale_factor ** 3),
                         kernel_size=3, padding=1, bias=False),
                PixelShuffle3d(upscale_factor),
                nn.GroupNorm(ngroups, out_channels),
                nn.GELU()
            )
        else:
            # No upsampling, just refine
            self.upsample = nn.Sequential(
                nn.Conv3d(in_channels + skip_channels, out_channels,
                         kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(ngroups, out_channels),
                nn.GELU()
            )
        
        # Refinement after size adjustment
        self.refine = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                     padding=1, bias=False),
            nn.GroupNorm(ngroups, out_channels),
            nn.GELU()
        )
    
    def _get_num_groups(self, channels):
        """Calculate valid number of groups for GroupNorm."""
        ngroups = min(8, channels)
        if channels % ngroups != 0:
            # Find largest divisor <= 8
            for g in reversed(range(1, ngroups + 1)):
                if channels % g == 0:
                    return g
        return ngroups
    
    def forward(self, x, skip=None):
        # Concatenate skip connection
        if self.use_skip and skip is not None:
            x = torch.cat([x, skip], dim=1)
        
        # PixelShuffle upsampling
        x = self.upsample(x)
        
        # Adjust to exact target shape if needed
        if self.target_shape is not None:
            current_shape = x.shape[2:]  # (H, W, T)
            if current_shape != self.target_shape:
                # Use trilinear interpolation to reach exact size
                x = F.interpolate(
                    x, 
                    size=self.target_shape, 
                    mode='trilinear', 
                    align_corners=False
                )
        
        # Refinement convolutions
        x = self.refine(x)
        return x


class VQVAE(nn.Module):
    """
    Compress single EEG chunk with flexible input dimensions.
    Automatically adapts to input shape: (channels, H, W, T)
    
    NOW WITH OPTIMIZED DECODER:
    - Pixel Shuffle upsampling (no checkerboard artifacts)
    - Adaptive interpolation for exact dimension matching
    - Optional U-Net style skip connections
    - Progressive refinement at each stage
    """
    def __init__(
        self,
        in_channels=25,
        input_spatial=(7, 5, 32),  # (H, W, T)
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

        # Calculate compression strategy
        self.encoder_config = self._plan_encoder_stages(
            input_spatial, num_downsample_stages
        )
        
        # Build encoder
        self.encoder = self._build_encoder(in_channels, self.encoder_config)
        
        # Calculate flattened size after encoding
        final_shape = self.encoder_config[-1]['output_shape']
        final_channels = self.encoder_config[-1]['out_channels']
        self.flat_size = final_channels * math.prod(final_shape)
        
        # Store for decoder
        self.final_shape = final_shape
        self.final_channels = final_channels
        
        # Bottleneck to target embedding
        self.to_embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_size, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # Vector quantizer
        self.vq = VectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=embedding_dim
        )
        
        # Decoder bottleneck
        self.from_embedding = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, self.flat_size)
        )
        
        # Build optimized decoder with hybrid upsampling
        self.decoder_blocks = self._build_decoder(in_channels, self.encoder_config)
        
    def _plan_encoder_stages(self, input_spatial, num_stages):
        """Plan encoder stages to progressively downsample spatial dimensions."""
        config = []
        current_shape = list(input_spatial)
        base_channels = 64
        
        for i in range(num_stages):
            out_channels = base_channels * (2 ** i)
            
            # Calculate stride for each dimension
            stride = []
            output_shape = []
            for dim_size in current_shape:
                # Use stride=2 if dimension is large enough, else stride=1
                s = 2 if dim_size >= 4 else 1
                stride.append(s)
                output_shape.append((dim_size + s - 1) // s)  # Ceiling division
            
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
        """Build encoder based on configuration."""
        layers = []
        
        for i, stage in enumerate(config):
            in_ch = in_channels if i == 0 else config[i-1]['out_channels']
            out_ch = stage['out_channels']
            
            layers.extend([
                nn.Conv3d(
                    in_ch, out_ch,
                    kernel_size=stage['kernel_size'],
                    stride=stage['stride'],
                    padding=stage['padding'],
                    bias=False
                ),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.GELU()
            ])
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self, out_channels, encoder_config):
        """
        Build decoder with hybrid upsampling strategy.
        PixelShuffle for learnable features + interpolation for exact sizes.
        """
        blocks = nn.ModuleList()
        
        # Reverse encoder config
        decoder_config = list(reversed(encoder_config))
        
        for i, stage in enumerate(decoder_config):
            # Determine input channels
            if i == 0:
                in_ch = self.final_channels
            else:
                in_ch = decoder_config[i-1]['in_channels']
            
            # Determine output channels
            out_ch = stage['in_channels'] if stage['in_channels'] is not None else out_channels
            
            # Last layer outputs original channels
            if i == len(decoder_config) - 1:
                out_ch = out_channels
            
            # Get target shape and upscale factor
            target_shape = stage['input_shape']  # Target (H, W, T)
            stride = stage['stride']
            upscale_factor = stride[0] if all(s == stride[0] for s in stride) else 1
            
            # Create decoder block with target shape for adaptive upsampling
            blocks.append(DecoderBlock(
                in_ch, out_ch, target_shape, upscale_factor,
                use_skip=self.use_skip_connections
            ))
        
        return blocks
    
    def _get_encoder_features(self, x):
        """Extract intermediate encoder features for skip connections."""
        features = []
        z = x
        
        # Process encoder in stages
        layer_idx = 0
        for i, stage_config in enumerate(self.encoder_config):
            # Each stage has 3 layers: Conv3d, GroupNorm, GELU
            stage_layers = self.encoder[layer_idx:layer_idx+3]
            for layer in stage_layers:
                z = layer(z)
            features.append(z)
            layer_idx += 3
        
        return features
    
    def encode(self, x):
        """
        Encode single chunk.
        Input: (B, in_channels, H, W, T)
        Output: (B, embedding_dim), vq_loss, indices
        """
        # Get encoder features for skip connections if needed
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
        """
        Decode from embedding using hybrid decoder.
        Input: (B, embedding_dim)
        Output: (B, in_channels, H, W, T)
        """
        # Embedding to spatial features
        z = self.from_embedding(z_q)
        z = z.view(-1, self.final_channels, *self.final_shape)
        
        # Progressive decoding with optional skip connections
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
        
        # Clean up stored features
        if hasattr(self, '_encoder_features'):
            del self._encoder_features
        
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
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.max_chunks, embedding_dim) * 0.02
        )
    
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
        
        # Add positional encoding
        embeddings = embeddings + self.pos_embedding[:, :num_chunks, :]
        
        return embeddings, vq_loss, indices
    
    def decode_sequence(self, embeddings):
        """
        Decode sequence of embeddings back to chunks.
        Input: (B, num_chunks, embedding_dim)
        Output: (B, num_chunks, C, H, W, T)
        """
        batch_size, num_chunks = embeddings.shape[:2]
        
        # Remove positional encoding
        embeddings = embeddings - self.pos_embedding[:, :num_chunks, :]
        
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
        return chunks_recon, vq_loss, indices


class VectorQuantizer(nn.Module):
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
        # Flatten input
        flat_z = z.view(-1, self.embedding_dim)

        # Compute distances to embeddings
        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding ** 2, dim=1)
            - 2 * torch.matmul(flat_z, self.embedding.t())
        )

        # Get closest embedding index for each input
        indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(indices, self.num_embeddings).type(flat_z.dtype)

        # Quantized latent vectors
        z_q = torch.matmul(encodings, self.embedding)

        # EMA updates (no gradients)
        if self.training:
            # Update cluster size counts
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                    (1 - self.decay) * torch.sum(encodings, dim=0)

            # Update embedding averages
            dw = torch.matmul(encodings.t(), flat_z)
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw

            # Normalize so small clusters don’t die
            n = torch.sum(self.ema_cluster_size)
            cluster_size = (
                (self.ema_cluster_size + self.eps)
                / (n + self.num_embeddings * self.eps)
                * n
            )
            self.embedding.data = self.ema_w / cluster_size.unsqueeze(1)

        # Compute commitment loss
        loss = self.commitment_cost * torch.mean((z_q.detach() - flat_z) ** 2)

        # Straight-through estimator
        z_q = flat_z + (z_q - flat_z).detach()
        z_q = z_q.view_as(z)

        return z_q, loss, indices



# Example usage
if __name__ == "__main__":
    # Example 1: Original setup (25, 7, 5, 32) chunks
    print("Example 1: (25, 7, 5, 32) chunks")
    model1 = SequenceProcessor(
        chunk_shape=(25, 7, 5, 32),
        embedding_dim=128,
        codebook_size=512
    )
    
    # Input: 8 chunks of (25, 7, 5, 32)
    x1 = torch.randn(2, 8, 25, 7, 5, 32)
    recon1, loss1, indices1 = model1(x1)
    print(f"Input: {x1.shape}, Output: {recon1.shape}")
    
    # Example 2: (25, 7, 5, 250) split into 10 chunks of 25 samples
    print("\nExample 2: (25, 7, 5, 25) chunks from 250-sample signal")
    model2 = SequenceProcessor(
        chunk_shape=(25, 7, 5, 25),
        embedding_dim=128,
        codebook_size=512
    )
    
    # Input: 10 chunks of (25, 7, 5, 25)
    x2 = torch.randn(2, 10, 25, 7, 5, 25)
    recon2, loss2, indices2 = model2(x2)
    print(f"Input: {x2.shape}, Output: {recon2.shape}")
    
    # Example 3: Different channel count
    print("\nExample 3: (64, 8, 8, 16) chunks")
    model3 = SequenceProcessor(
        chunk_shape=(64, 8, 8, 16),
        embedding_dim=256,
        codebook_size=1024,
        num_downsample_stages=3
    )
    
    x3 = torch.randn(2, 5, 64, 8, 8, 16)
    recon3, loss3, indices3 = model3(x3)
    print(f"Input: {x3.shape}, Output: {recon3.shape}")








# Example usage
if __name__ == "__main__":
    print("="*70)
    print("VQ-VAE FOR EEG CHUNKS WITH SEPARATE LOSS")
    print("="*70)
    
    # ========== Single Chunk Model ==========
    print("\n1. Single Chunk Model:")
    print("-" * 70)
    
    chunk_ae = VQVAE(
        in_channels=25,
        input_spatial=(7, 5, 32),
        embedding_dim=128,
        codebook_size=512
    )
    from packages.train.loss import VQVAELoss, SequenceVQVAELoss
    # Loss function
    criterion_chunk = VQVAELoss(
        recon_loss_type='mse',
        recon_weight=1.0
    )
    
    # Test single chunk
    x_chunk = torch.randn(4, 25, 7, 5, 32)
    outputs = chunk_ae(x_chunk)
    x_recon, vq_loss, indices = outputs
    # Compute loss
    loss = criterion_chunk(outputs, x_chunk )
    
    print(f"Input shape: {x_chunk.shape}")
    print(f"Output shape: {x_recon.shape}")
    print(f"VQ loss: {vq_loss.item():.4f}")
    print(f"Total loss: {loss.item():.4f}")
    print(f"Compression: {25*7*5*32} → {128} = {(25*7*5*32)/128:.1f}x")
    
    # ========== Sequence Model ==========
    print("\n2. Sequence Model:")
    print("-" * 70)
    
    seq_processor = SequenceProcessor(chunk_shape=(25, 7, 5, 25), embedding_dim=128, codebook_size=512)
    
    # Loss function for sequences
    criterion_seq = SequenceVQVAELoss(
        recon_loss_type='mse',
        recon_weight=1.0
    )
    
    # Test sequence of chunks
    chunks = torch.randn(4, 8, 25, 7, 5, 25)
    outputs = seq_processor(chunks)
    
    # Compute loss
    loss_seq = criterion_seq(outputs, chunks)
    chunks_recon, vq_loss_seq, indices_seq = outputs
    print(f"Input chunks: {chunks.shape}")
    print(f"Output chunks: {chunks_recon.shape}")
    print(f"VQ loss: {vq_loss_seq.item():.4f}")
    print(f"Total loss: {loss_seq.item():.4f}")
    
    # Get embeddings for transformer
    embeddings, _, _ = seq_processor.encode_sequence(chunks)
    print(f"\nEmbeddings for transformer: {embeddings.shape}")
    print(f"  → (batch={embeddings.shape[0]}, seq_len={embeddings.shape[1]}, d_model={embeddings.shape[2]})")
    
    # ========== Training Example ==========
    print("\n3. Training Example:")
    print("-" * 70)
    
    # Setup
    model = seq_processor
    criterion = criterion_seq
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Simulated training step
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(chunks)
    
    # Compute loss
    loss = criterion(outputs, chunks)
    chunks_recon, vq_loss, indices = outputs
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    print(f"Training step completed!")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  VQ loss: {vq_loss.item():.4f}")
    
    print("\n" + "="*70)
    print("USAGE IN YOUR TRAINING LOOP:")
    print("="*70)
    print("""
# Initialize
model = SequenceProcessor(chunk_ae, num_chunks=8, embedding_dim=128)
criterion = SequenceVQVAELoss(recon_loss_type='mse')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
for batch in dataloader:
    chunks = batch  # (B, 8, 25, 7, 5, 32)
    
    optimizer.zero_grad()
    
    # Forward
    chunks_recon, vq_loss, _ = model(chunks)
    
    # Loss
    loss = criterion(chunks, chunks_recon, vq_loss)
    
    # Backward
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
# After training, get embeddings for transformer:
with torch.no_grad():
    embeddings, _, _ = model.encode_sequence(chunks)
    # embeddings: (B, 8, 128) ready for transformer!
    """)


    print(model)
