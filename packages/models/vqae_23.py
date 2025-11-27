import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class VQAEConfig:
    """Configuration for the VQ-VAE model."""
    use_quantizer: bool = True  # Whether to use vector quantization
    # Data shape parameters
    num_freq_bands: int = 25          # F: Number of frequency bands
    spatial_rows: int = 7              # R: Spatial grid rows
    spatial_cols: int = 5              # C: Spatial grid cols
    time_samples: int = 250            # T: Time samples per clip
    chunk_dim: int = 50                # ChunkDim: Time chunk length
    orig_channels: int = 32            # Original EEG channels (R*C or separate)
    
    # Encoder parameters
    encoder_2d_channels: list = None   # [32, 64] - 2D conv channels
    encoder_3d_channels: list = None   # [64, 128, 256] - 3D conv channels
    embedding_dim: int = 256           # Final embedding dimension
    
    # VQ parameters
    codebook_size: int = 512           # Number of codebook vectors
    commitment_cost: float = 0.25      # Beta for commitment loss
    ema_decay: float = 0.99            # EMA decay for codebook updates
    epsilon: float = 1e-5              # Small constant for numerical stability
    
    # Decoder parameters
    decoder_channels: list = None      # [256, 128, 64] - Decoder channels
    
    dropout_2d: float = 0.1          # Dropout for 2D encoder
    dropout_3d: float = 0.1          # Dropout for 3D encoder
    dropout_bottleneck: float = 0.2  # Dropout at bottleneck
    dropout_decoder: float = 0.1     # Dropout for decoder
    
    def __post_init__(self):
        if self.encoder_2d_channels is None:
            self.encoder_2d_channels = [8, 16]
        if self.encoder_3d_channels is None:
            self.encoder_3d_channels = [16, 64, 128]
        if self.decoder_channels is None:
            self.decoder_channels = [64, 32, 16]
        
        # Calculate number of chunks
        self.num_chunks = self.time_samples // self.chunk_dim
        
        # Validate configuration
        assert self.time_samples % self.chunk_dim == 0, \
            f"time_samples ({self.time_samples}) must be divisible by chunk_dim ({self.chunk_dim})"



class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average updates.
    
    Implements the VQ layer from VQ-VAE with EMA-based codebook learning,
    which is more stable than direct codebook gradient descent.
    
    Args:
        num_embeddings: Size of the codebook (K)
        embedding_dim: Dimension of each embedding vector (D)
        commitment_cost: Weight for commitment loss (beta)
        decay: EMA decay rate for codebook updates
        epsilon: Small constant for numerical stability
    """
    
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
        
        # Initialize codebook with uniform distribution
        self.register_buffer('embeddings', torch.randn(num_embeddings, embedding_dim))
        self.embeddings.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
        # EMA cluster size and embeddings sum
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_embed_avg', self.embeddings.clone())
        
    def forward(
        self,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through vector quantization.
        
        Args:
            inputs: Tensor of shape (B, D) where B is batch size, D is embedding_dim
            
        Returns:
            quantized: Quantized tensor (B, D)
            encoding_indices: Indices of selected codebook vectors (B,)
            loss_dict: Dictionary containing VQ losses
        """
        # Flatten input if needed
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances to codebook vectors
        # ||z_e - e||^2 = ||z_e||^2 + ||e||^2 - 2 * z_e · e
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self.embeddings.t())
        )
        
        # Find nearest codebook vector for each input
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(encoding_indices, self.embeddings)
        
        # Update codebook with EMA (only during training)
        if self.training:
            self._ema_update(flat_input, encoding_indices)
        
        # Compute losses
        # Commitment loss: encoder commits to codebook
        e_latent_loss = F.mse_loss(quantized.detach(), flat_input)
        
        # Straight-through estimator: copy gradients from decoder to encoder
        quantized = flat_input + (quantized - flat_input).detach()
        
        # Codebook loss (for monitoring; EMA handles actual updates)
        q_latent_loss = F.mse_loss(quantized, flat_input.detach())
        
        # Total VQ loss
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Reshape back to input shape
        quantized = quantized.view(input_shape)
        
        # Perplexity: measure of codebook usage
        avg_probs = torch.bincount(
            encoding_indices,
            minlength=self.num_embeddings
        ).float() / len(encoding_indices)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        loss_dict = {
            'vq_loss': vq_loss,
            'commitment_loss': e_latent_loss,
            'codebook_loss': q_latent_loss,
            'perplexity': perplexity
        }
        print(encoding_indices.unique())
        return quantized, encoding_indices, loss_dict
    
    def _ema_update(self, flat_input: torch.Tensor, encoding_indices: torch.Tensor):
        """
        Update codebook using exponential moving average.
        
        This is more stable than gradient-based updates and prevents codebook collapse.
        """
        # Count how many vectors are assigned to each codebook entry
        encodings_onehot = F.one_hot(
            encoding_indices,
            num_classes=self.num_embeddings
        ).float()
        
        # Update cluster sizes with EMA
        updated_cluster_size = torch.sum(encodings_onehot, dim=0)
        self.ema_cluster_size.data.mul_(self.decay).add_(
            updated_cluster_size, alpha=1 - self.decay
        )
        
        # Laplace smoothing to prevent cluster collapse
        n = torch.sum(self.ema_cluster_size)
        self.ema_cluster_size.data.add_(self.epsilon).div_(
            n + self.num_embeddings * self.epsilon
        ).mul_(n)
        
        # Update embedding averages with EMA
        embed_sum = torch.matmul(encodings_onehot.t(), flat_input)
        self.ema_embed_avg.data.mul_(self.decay).add_(
            embed_sum, alpha=1 - self.decay
        )
        
        # Normalize embeddings
        self.embeddings.data.copy_(
            self.ema_embed_avg / self.ema_cluster_size.unsqueeze(1)
        )


class Encoder2DStage(nn.Module):
    """
    First encoder stage: 2D convolutions over (frequency × time) for each spatial voxel.
    
    Processes each (R×C) spatial location independently with 2D convs across frequency-time.
    """
    
    def __init__(self, config: VQAEConfig):
        super().__init__()
        self.config = config
        
        layers = []
        in_channels = 1  # Single channel input after reshape
        
        for i, out_channels in enumerate(config.encoder_2d_channels):
            layers.extend([
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
                nn.Dropout2d(p=config.dropout_2d) if i < len(config.encoder_2d_channels) - 1 else nn.Identity()
            ])
            in_channels = out_channels
        
        self.conv_net = nn.Sequential(*layers)
        
        # Calculate output dimensions after 2D convolutions
        self.freq_out = config.num_freq_bands
        self.time_out = config.chunk_dim
        for _ in config.encoder_2d_channels:
            self.freq_out = (self.freq_out + 1) // 2
            self.time_out = (self.time_out + 1) // 2
        
        self.out_channels = config.encoder_2d_channels[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape (B*nChunk*R*C, 1, F, ChunkDim)
            
        Returns:
            Shape (B*nChunk*R*C, C_out, F', T')
        """
        return self.conv_net(x)


class Encoder3DStage(nn.Module):
    """
    Second encoder stage: 3D convolutions over (row × col × time) volume.
    
    Processes spatial-temporal structure with frequency bands merged as channels.
    """
    
    def __init__(self, config: VQAEConfig, channels_in: int, time_in: int, **kwargs):
        super().__init__()
        self.config = config
        
        # Note: freq_in is ignored since we merged it into channels_in
        
        layers = []
        in_channels = channels_in  # This is C_out * F_out
        
        for i, out_channels in enumerate(config.encoder_3d_channels):
            layers.extend([
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm3d(out_channels),
                nn.SiLU(inplace=True),
                nn.Dropout3d(p=config.dropout_3d) if i < len(config.encoder_3d_channels) - 1 else nn.Identity()
            ])
            in_channels = out_channels
        
        self.conv_net = nn.Sequential(*layers)
        
        # Calculate output spatial dimensions
        # 3D conv operates over (R, C, T_out) dimensions
        row_out = config.spatial_rows
        col_out = config.spatial_cols
        time_out = time_in
        
        for _ in config.encoder_3d_channels:
            row_out = (row_out + 1) // 2
            col_out = (col_out + 1) // 2
            time_out = (time_out + 1) // 2
        
        # Final projection to embedding dimension
        flatten_dim = (
            config.encoder_3d_channels[-1] * row_out * col_out * time_out
        )
        
        self.projection = nn.Sequential(
            nn.Flatten(),
             nn.Dropout(p=config.dropout_bottleneck),
            nn.Linear(flatten_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape (B*nChunk, C_out*F_out, R, C, T_out)
            
        Returns:
            Shape (B*nChunk, embedding_dim)
        """
        x = self.conv_net(x)
        x = self.projection(x)
        return x



class Decoder(nn.Module):
    """
    Decoder: Reconstructs plain EEG from quantized embeddings.
    
    Maps from compact embedding back to (OrigChannels × ChunkDim) EEG signal.
    """
    
    def __init__(self, config: VQAEConfig):
        super().__init__()
        self.config = config
        
        # Initial projection from embedding to feature map
        # Target shape for transposed convs: (OrigChannels, ChunkDim)
        init_time = config.chunk_dim // (2 ** len(config.decoder_channels))
        init_channels = config.decoder_channels[0]
        
        self.init_dim = init_channels * init_time
        
        self.projection = nn.Sequential(
            nn.Linear(config.embedding_dim, self.init_dim),
            nn.Dropout(p=0.2),
            nn.SiLU(inplace=True)
        )
        
        # Transpose convolutions to upsample
        layers = []
        in_channels = init_channels
        
        for i, out_channels in enumerate(config.decoder_channels[1:]):
            layers.extend([
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm1d(out_channels),
                nn.SiLU(inplace=True),
                nn.Dropout(p=config.dropout_decoder) if i < len(config.decoder_channels[1:]) - 1 else nn.Identity() 
            ])
            in_channels = out_channels
        
        # Final layer to original channels
        layers.append(
            nn.ConvTranspose1d(
                in_channels,
                config.orig_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )
        
        self.decoder_net = nn.Sequential(*layers)
    
    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_q: Shape (B*nChunk, embedding_dim)
            
        Returns:
            Shape (B*nChunk, OrigChannels, ChunkDim)
        """
        batch_size = z_q.shape[0]
        
        # Project to initial feature map
        x = self.projection(z_q)
        x = x.view(batch_size, self.config.decoder_channels[0], -1)
        
        # Upsample to target dimensions
        x = self.decoder_net(x)
        
        # Ensure correct output size
        if x.shape[-1] != self.config.chunk_dim:
            x = F.interpolate(
                x,
                size=self.config.chunk_dim,
                mode='linear',
                align_corners=False
            )
        
        return x

class VQAE(nn.Module):
    def __init__(self, config: VQAEConfig):
        super().__init__()
        
        if isinstance(config, dict):
            config = VQAEConfig(**config)
        elif not isinstance(config, VQAEConfig):
            raise TypeError(f"config must be VQVAEConfig or dict, got {type(config)}")
        
        self.config = config

        # Encoder stages
        self.encoder_2d = Encoder2DStage(config)

        self.encoder_3d = Encoder3DStage(
            config,
            freq_in=self.encoder_2d.freq_out,  # This is now absorbed into channels
            time_in=self.encoder_2d.time_out,
            channels_in=self.encoder_2d.out_channels * self.encoder_2d.freq_out 
        )
        
        # Vector quantization
        self.vq = VectorQuantizer(
            num_embeddings=config.codebook_size,
            embedding_dim=config.embedding_dim,
            commitment_cost=config.commitment_cost,
            decay=config.ema_decay,
            epsilon=config.epsilon
        )
        
        # Decoder
        self.decoder = Decoder(config)

    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode wavelet-transformed EEG to embeddings (before VQ).
        
        Args:
            x: Shape (B, F, R, C, T)
            
        Returns:
            Shape (B*nChunk, embedding_dim)
        """
        B, F, R, C, T = x.shape
        
        # Step 1: Chunk along time dimension
        x_chunked = self._chunk_time(x)  # (B*nChunk, F, R, C, ChunkDim)
        
        # Step 2: Reshape for 2D convolutions
        BnC = x_chunked.shape[0]
        x_2d = x_chunked.permute(0, 2, 3, 1, 4)  # (B*nChunk, R, C, F, ChunkDim)
        x_2d = x_2d.reshape(BnC * R * C, 1, F, self.config.chunk_dim)
        
        # Step 3: 2D encoder
        x_2d = self.encoder_2d(x_2d)  # (B*nChunk*R*C, C_out, F', T')
        
        # Step 4: Reshape back to 3D structure
        C_out, F_out, T_out = (
            self.encoder_2d.out_channels,
            self.encoder_2d.freq_out,
            self.encoder_2d.time_out
        )
        x_3d = x_2d.view(BnC, R, C, C_out*F_out, T_out)  # Shape: (BnC, R, C, C_out*F_out, T_out)
        x_3d = x_3d.permute(0, 3, 1, 2, 4)  # Shape: (BnC, C_out*F_out, R, C, T_out)


        # Step 5: 3D encoder
        z_e = self.encoder_3d(x_3d)  # (B*nChunk, embedding_dim)
        
        return z_e
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized embeddings to plain EEG.
        
        Args:
            z_q: Shape (B*nChunk, embedding_dim)
            
        Returns:
            Shape (B*nChunk, OrigChannels, ChunkDim)
        """
        return self.decoder(z_q)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Full forward pass: encode, quantize, decode.
        
        Args:
            x: Wavelet-transformed EEG, shape (B, F, R, C, T)
            plain_eeg: Plain EEG for reconstruction target, shape (B, OrigChannels, T)
            
        Returns:
            recon: Reconstructed EEG, shape (B*nChunk, OrigChannels, ChunkDim)
            losses: Dictionary of loss components
        """
        # Encode
        z_e = self.encode(x)
        
        # Vector quantization
        if self.config.use_quantizer:
            z_q, indices, vq_losses = self.vq(z_e)
        else:
            z_q = z_e
            indices = torch.zeros(z_e.shape[0], dtype=torch.long, device=z_e.device)
            vq_losses = {'vq_loss': torch.tensor(0.0, device=z_e.device)}   
        
        # Decode
        recon = self.decode(z_q)

        recon = recon.reshape(-1, self.config.orig_channels, self.config.time_samples)  
        return {'reconstruction': recon, 'embeddings': z_e, 'vq_loss': vq_losses['vq_loss']}
    
    def _chunk_time(self, x: torch.Tensor) -> torch.Tensor:
        """
        Chunk tensor along time dimension.
        
        Args:
            x: Shape (B, F, R, C, T)
            
        Returns:
            Shape (B*nChunk, F, R, C, ChunkDim)
        """
        B, F, R, C, T = x.shape
        nChunk = self.config.num_chunks
        ChunkDim = self.config.chunk_dim
        
        x = x.reshape(B, F, R, C, nChunk, ChunkDim)
        x = x.permute(0, 4, 1, 2, 3, 5)  # (B, nChunk, F, R, C, ChunkDim)
        x = x.reshape(B * nChunk, F, R, C, ChunkDim)
        
        return x
    
    def unchunk_time(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Reverse chunking operation to reconstruct full time series.
        
        Args:
            x: Shape (B*nChunk, OrigChannels, ChunkDim)
            batch_size: Original batch size B
            
        Returns:
            Shape (B, OrigChannels, T)
        """
        BnC, C, ChunkDim = x.shape
        nChunk = self.config.num_chunks
        
        x = x.reshape(batch_size, nChunk, C, ChunkDim)
        x = x.permute(0, 2, 1, 3)  # (B, C, nChunk, ChunkDim)
        x = x.reshape(batch_size, C, nChunk * ChunkDim)
        
        return x
    

if __name__ == "__main__":
    # Simple test of VQAE model
    config = VQAEConfig()
    model = VQAE(config)
    
    # Dummy input: batch size 2, F=25, R=8, C=8, T=250
    x = torch.randn(2, config.num_freq_bands, config.spatial_rows, config.spatial_cols, config.time_samples)
    
    outputs = model(x)
    recon = outputs['reconstruction']
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {recon.shape}")
    print(model)
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")