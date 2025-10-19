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


class CompactVQVAE_EEGChunk(nn.Module):
    """
    Compress single EEG chunk: (25, 7, 5, 32) → 128-256 dim
    Designed for sequence modeling with transformers.
    """
    def __init__(
        self,
        in_channels=25,
        input_spatial=(7, 5, 32),
        embedding_dim=128,
        codebook_size=512
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.input_spatial = input_spatial
        self.embedding_dim = embedding_dim
        
        # Efficient encoder (aggressive spatial compression)
        self.encoder = nn.Sequential(
            # Stage 1: (7,5,32) → (4,3,16)
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, 64),
            nn.GELU(),
            
            # Stage 2: (4,3,16) → (2,2,8)
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, 128),
            nn.GELU(),
            
            # Stage 3: (2,2,8) → (1,1,4)
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, 256),
            nn.GELU()
        )
        
        # After encoder: (256, 1, 1, 4) = 1024 features
        self.flat_size = 256 * 1 * 1 * 4
        
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
        
        # Decoder
        self.from_embedding = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, self.flat_size)
        )
        
        self.decoder = nn.Sequential(
            # (256, 1, 1, 4) → (128, 2, 2, 8)
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(1, 128),
            nn.GELU(),
            
            # (128, 2, 2, 8) → (64, 4, 3, 16)
            nn.Upsample(size=(4, 3, 16), mode='trilinear', align_corners=False),
            nn.Conv3d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, 64),
            nn.GELU(),
            
            # (64, 4, 3, 16) → (25, 7, 5, 32)
            nn.Upsample(size=(7, 5, 32), mode='trilinear', align_corners=False),
            nn.Conv3d(64, in_channels, kernel_size=3, padding=1)
        )
        
    def encode(self, x):
        """
        Encode single chunk.
        Input: (B, 25, 7, 5, 32)
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
        Output: (B, 25, 7, 5, 32)
        """
        z = self.from_embedding(z_q)
        z = z.view(-1, 256, 1, 1, 4)
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
    Process full 1-second EEG window as sequence of chunks.
    This is what feeds into your transformer!
    """
    def __init__(
        self,
        chunk_ae,
        num_chunks=8,
        embedding_dim=128
    ):
        super().__init__()
        self.chunk_ae = chunk_ae
        self.num_chunks = num_chunks
        self.embedding_dim = embedding_dim
        
        # Learned positional encoding for chunk position
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_chunks, embedding_dim) * 0.02
        )
    
    def encode_sequence(self, chunks):
        """
        Encode sequence of chunks.
        Input: (B, num_chunks, 25, 7, 5, 32)
        Output: (B, num_chunks, embedding_dim), vq_loss, indices
        """
        batch_size, num_chunks = chunks.shape[:2]
        
        # Reshape to process all chunks
        chunks_flat = chunks.view(-1, 25, 7, 5, 32)
        
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
        Output: (B, num_chunks, 25, 7, 5, 32)
        """
        batch_size, num_chunks = embeddings.shape[:2]
        
        # Remove positional encoding
        embeddings = embeddings - self.pos_embedding[:, :num_chunks, :]
        
        # Reshape to decode all chunks
        embeddings_flat = embeddings.view(-1, self.embedding_dim)
        
        # Decode each chunk
        chunks_recon = self.chunk_ae.decode(embeddings_flat)
        
        # Reshape back to sequence
        chunks_recon = chunks_recon.view(batch_size, num_chunks, 25, 7, 5, 32)
        
        return chunks_recon
    
    def forward(self, chunks):
        """
        Full forward pass.
        Input: (B, num_chunks, 25, 7, 5, 32)
        Output: (reconstruction, vq_loss, indices)
        """
        embeddings, vq_loss, indices = self.encode_sequence(chunks)
        chunks_recon = self.decode_sequence(embeddings)
        return chunks_recon, vq_loss, indices








# Example usage
if __name__ == "__main__":
    print("="*70)
    print("VQ-VAE FOR EEG CHUNKS WITH SEPARATE LOSS")
    print("="*70)
    
    # ========== Single Chunk Model ==========
    print("\n1. Single Chunk Model:")
    print("-" * 70)
    
    chunk_ae = CompactVQVAE_EEGChunk(
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
    x_recon, vq_loss, indices = chunk_ae(x_chunk)
    
    # Compute loss
    loss = criterion_chunk(x_chunk, x_recon, vq_loss)
    
    print(f"Input shape: {x_chunk.shape}")
    print(f"Output shape: {x_recon.shape}")
    print(f"VQ loss: {vq_loss.item():.4f}")
    print(f"Total loss: {loss.item():.4f}")
    print(f"Compression: {25*7*5*32} → {128} = {(25*7*5*32)/128:.1f}x")
    
    # ========== Sequence Model ==========
    print("\n2. Sequence Model:")
    print("-" * 70)
    
    seq_processor = SequenceProcessor(
        chunk_ae=chunk_ae,
        num_chunks=8,
        embedding_dim=128
    )
    
    # Loss function for sequences
    criterion_seq = SequenceVQVAELoss(
        recon_loss_type='mse',
        recon_weight=1.0
    )
    
    # Test sequence of chunks
    chunks = torch.randn(4, 8, 25, 7, 5, 32)
    chunks_recon, vq_loss_seq, indices_seq = seq_processor(chunks)
    
    # Compute loss
    loss_seq = criterion_seq(chunks, chunks_recon, vq_loss_seq)
    
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
    chunks_recon, vq_loss, _ = model(chunks)
    
    # Compute loss
    loss = criterion(chunks, chunks_recon, vq_loss)
    
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
