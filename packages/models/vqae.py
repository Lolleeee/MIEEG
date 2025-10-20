import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Discrete codebook for VQ-VAE"""
    def __init__(self, num_embeddings=512, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z):
        # z: (B, embedding_dim)
        # Compute distances to codebook
        distances = (
            torch.sum(z**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z, self.embedding.weight.t())
        )
        
        # Find nearest codebook entry
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices)
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        
        # VQ loss
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + 0.25 * e_latent_loss
        
        return quantized, loss, encoding_indices


import torch
import torch.nn as nn
import math


class VQVAE(nn.Module):
    """
    Compress single EEG chunk with flexible input dimensions.
    Automatically adapts to input shape: (channels, H, W, T)
    """
    def __init__(
        self,
        in_channels=25,
        input_spatial=(7, 5, 32),  # (H, W, T)
        embedding_dim=128,
        codebook_size=512,
        num_downsample_stages=3  # How aggressively to compress
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.input_spatial = input_spatial
        self.embedding_dim = embedding_dim
        self.num_stages = num_downsample_stages
        
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
        
        # Decoder (mirror of encoder)
        self.from_embedding = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, self.flat_size)
        )
        
        self.decoder = self._build_decoder(
            in_channels, self.encoder_config, final_channels, final_shape
        )
        
    def _plan_encoder_stages(self, input_spatial, num_stages):
        """
        Plan encoder stages to progressively downsample spatial dimensions.
        Returns list of dicts with stage configurations.
        """
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
                nn.GroupNorm(1, out_ch),
                nn.GELU()
            ])
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self, out_channels, encoder_config, final_channels, final_shape):
        """Build decoder by reversing encoder configuration with exact output sizes."""
        layers = []
        
        # Reverse the encoder config
        decoder_config = list(reversed(encoder_config))
        
        for i, stage in enumerate(decoder_config):
            in_ch = final_channels if i == 0 else decoder_config[i-1]['in_channels']
            out_ch = stage['in_channels'] if stage['in_channels'] is not None else out_channels
            
            # If last stage, output original channels
            if i == len(decoder_config) - 1:
                out_ch = out_channels
            
            target_shape = stage['input_shape']
            current_shape = final_shape if i == 0 else decoder_config[i-1]['input_shape']
            stride = stage['stride']
            kernel_size = stage['kernel_size']
            padding = stage['padding']
            
            # Calculate output_padding for exact reconstruction
            # Formula: output = (input - 1) * stride - 2*padding + kernel_size + output_padding
            # We want: output = target, so: output_padding = target - ((input - 1) * stride - 2*padding + kernel_size)
            output_padding = []
            use_transposed = all(s == 2 for s in stride)
            
            if use_transposed:
                for curr_dim, target_dim, s, p, k in zip(current_shape, target_shape, stride, [padding]*3, [kernel_size]*3):
                    expected_output = (curr_dim - 1) * s - 2*p + k
                    out_pad = target_dim - expected_output
                    # output_padding must be less than stride or dilation
                    out_pad = max(0, min(out_pad, s - 1))
                    output_padding.append(out_pad)
                
                layers.extend([
                    nn.ConvTranspose3d(
                        in_ch, out_ch,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=tuple(output_padding),
                        bias=False
                    ),
                    nn.GroupNorm(1, out_ch) if i < len(decoder_config) - 1 else nn.Identity(),
                    nn.GELU() if i < len(decoder_config) - 1 else nn.Identity()
                ])
            else:
                # For non-uniform strides or stride=1, use upsample + conv
                if target_shape != current_shape:
                    layers.extend([
                        nn.Upsample(size=target_shape, mode='trilinear', align_corners=False),
                        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                        nn.GroupNorm(1, out_ch) if i < len(decoder_config) - 1 else nn.Identity(),
                        nn.GELU() if i < len(decoder_config) - 1 else nn.Identity()
                    ])
                else:
                    # Same size, just change channels
                    layers.extend([
                        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                        nn.GroupNorm(1, out_ch) if i < len(decoder_config) - 1 else nn.Identity(),
                        nn.GELU() if i < len(decoder_config) - 1 else nn.Identity()
                    ])
        
        return nn.Sequential(*layers)
    
    def encode(self, x):
        """
        Encode single chunk.
        Input: (B, in_channels, H, W, T)
        Output: (B, embedding_dim), vq_loss, indices
        """
        z = self.encoder(x)
        z = self.to_embedding(z)
        
        # Vector quantization
        z_q, vq_loss, indices = self.vq(z)
        
        return z_q, vq_loss, indices
    
    def decode(self, z_q):
        """
        Decode from embedding.
        Input: (B, embedding_dim)
        Output: (B, in_channels, H, W, T)
        """
        z = self.from_embedding(z_q)
        final_shape = self.encoder_config[-1]['output_shape']
        final_channels = self.encoder_config[-1]['out_channels']
        z = z.view(-1, final_channels, *final_shape)
        x_recon = self.decoder(z)
        return x_recon
    
    def forward(self, x):
        """
        Full forward pass.
        Returns: (reconstruction, vq_loss, indices)
        """
        z_q, vq_loss, indices = self.encode(x)
        x_recon = self.decode(z_q)
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
        num_downsample_stages=3
    ):
        super().__init__()
        
        self.chunk_shape = chunk_shape
        self.embedding_dim = embedding_dim
        
        # Create chunk autoencoder
        self.chunk_ae = VQVAE(
            in_channels=chunk_shape[0],
            input_spatial=chunk_shape[1:],  # (H, W, T)
            embedding_dim=embedding_dim,
            codebook_size=codebook_size,
            num_downsample_stages=num_downsample_stages
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


# Helper class (you'll need to define this based on your VectorQuantizer implementation)
class VectorQuantizer(nn.Module):
    """Simple vector quantizer."""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z):
        """
        Input: (B, D)
        Output: (z_q, loss, indices)
        """
        # Flatten input if needed
        flat_z = z.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (
            torch.sum(flat_z**2, dim=1, keepdim=True) +
            torch.sum(self.embedding.weight**2, dim=1) -
            2 * torch.matmul(flat_z, self.embedding.weight.t())
        )
        
        # Get closest codebook entry
        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices)
        
        # Calculate loss
        e_latent_loss = torch.mean((z_q.detach() - flat_z)**2)
        q_latent_loss = torch.mean((z_q - flat_z.detach())**2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        z_q = flat_z + (z_q - flat_z).detach()
        
        # Reshape to match input
        z_q = z_q.view(z.shape)
        
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
