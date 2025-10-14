import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for time series
    More effective than sinusoidal for EEG temporal patterns
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        batch_size, seq_len, _ = x.size()
        return x + self.encoding[:, :seq_len, :]


class SpatialCNN(nn.Module):
    """
    Shallow CNN to extract spatial features from channel connectivity
    """
    def __init__(self, in_channels, hidden_channels, normalization='batch'):
        super(SpatialCNN, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, hidden_channels[0], 
                               kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm3d(hidden_channels[0]) if normalization == 'batch' \
                     else nn.GroupNorm(min(8, hidden_channels[0]), hidden_channels[0])
        
        if len(hidden_channels) > 1:
            self.conv2 = nn.Conv3d(hidden_channels[0], hidden_channels[1], 
                                   kernel_size=3, stride=(2, 2, 1), padding=1)
            self.norm2 = nn.BatchNorm3d(hidden_channels[1]) if normalization == 'batch' \
                         else nn.GroupNorm(min(8, hidden_channels[1]), hidden_channels[1])
            self.has_conv2 = True
        else:
            self.has_conv2 = False
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # x: [B, 50, 7, 5, 250]
        x = self.relu(self.norm1(self.conv1(x)))  # [B, C1, 7, 5, 250]
        
        if self.has_conv2:
            x = self.relu(self.norm2(self.conv2(x)))  # [B, C2, 4, 3, 250]
        
        return x


class SpatialTransposedCNN(nn.Module):
    """
    Decoder CNN to reconstruct spatial dimensions
    """
    def __init__(self, hidden_channels, out_channels, normalization='batch'):
        super(SpatialTransposedCNN, self).__init__()
        
        if len(hidden_channels) > 1:
            self.deconv1 = nn.ConvTranspose3d(
                hidden_channels[0], hidden_channels[1],  # FIXED: Was [-1], [-2]
                kernel_size=3, stride=(2, 2, 1), padding=1, 
                output_padding=(1, 0, 0)
            )
            self.norm1 = nn.BatchNorm3d(hidden_channels[1]) if normalization == 'batch' \
                         else nn.GroupNorm(min(8, hidden_channels[1]), hidden_channels[1])
            self.has_deconv1 = True
            next_in_ch = hidden_channels[1]  # FIXED: Was [-2]
        else:
            self.has_deconv1 = False
            next_in_ch = hidden_channels[0]  # FIXED: Was [-1]
        
        self.deconv2 = nn.Conv3d(next_in_ch, out_channels, 
                                 kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        if self.has_deconv1:
            x = self.relu(self.norm1(self.deconv1(x)))  # [B, C1, 7, 5, 250]
        
        x = self.deconv2(x)  # [B, 50, 7, 5, 250]
        return x



class CNNTTAE(nn.Module):
    """
    Hybrid CNN + Temporal Transformer Autoencoder
    
    Architecture:
    1. Spatial CNN: Extract spatial features from channel connectivity (7x5)
    2. Temporal Transformer: Capture long-range temporal dependencies (250 timepoints)
    3. FC Bottleneck: Compress to latent representation
    4. Symmetric decoder
    """
    
    def __init__(
        self,
        input_channels=50,
        spatial_channels=[64, 96],      # CNN channels
        d_model=128,                     # Transformer dimension
        nhead=8,                         # Number of attention heads
        num_transformer_layers=2,        # Transformer depth
        latent_dim=128,                  # Bottleneck size
        normalization='batch',           # 'batch' or 'group'
        dropout=0.1,
        input_shape=(7, 5, 250)
    ):
        """
        Args:
            input_channels: Number of input channels (50 frequencies)
            spatial_channels: List of CNN channel sizes [64, 96] = 2 conv layers
            d_model: Transformer embedding dimension
            nhead: Number of attention heads in transformer
            num_transformer_layers: Number of transformer encoder layers
            latent_dim: Size of bottleneck embedding
            normalization: 'batch' or 'group'
            dropout: Dropout probability
            input_shape: Spatial dimensions (D, H, W)
        """
        super(CNNTTAE, self).__init__()
        
        self.input_channels = input_channels
        self.spatial_channels = spatial_channels
        self.d_model = d_model
        self.nhead = nhead
        self.num_transformer_layers = num_transformer_layers
        self.latent_dim = latent_dim
        self.normalization = normalization
        self.dropout = dropout
        self.input_shape = input_shape
        
        # ==================== ENCODER ====================
        
        # 1. Spatial CNN
        self.spatial_encoder = SpatialCNN(input_channels, spatial_channels, normalization)
        
        # Calculate spatial shape after CNN
        self.encoded_spatial_shape = self._calculate_cnn_output_shape()
        spatial_features = (spatial_channels[-1] * 
                           self.encoded_spatial_shape[0] * 
                           self.encoded_spatial_shape[1])
        
        # 2. Project to transformer dimension
        self.spatial_to_temporal = nn.Linear(spatial_features, d_model)
        
        # 3. Positional encoding for temporal dimension
        self.pos_encoding = PositionalEncoding(d_model, max_len=input_shape[2])
        
        # 4. Temporal Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,  # [batch, seq, feature]
            activation='gelu'
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        
        # 5. Bottleneck: temporal sequence to latent
        temporal_seq_len = input_shape[2]  # 250
        self.temporal_flatten_size = d_model * temporal_seq_len
        
        self.fc_encoder = nn.Sequential(
            nn.Linear(self.temporal_flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, latent_dim)
        )
        
        # ==================== DECODER ====================
        
        # 1. Bottleneck to temporal sequence
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.temporal_flatten_size)
        )
        
        # 2. Temporal Transformer Decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.temporal_decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_transformer_layers
        )
        
        # 3. Project back to spatial features
        self.temporal_to_spatial = nn.Linear(d_model, spatial_features)
        
        # 4. Spatial Transposed CNN
        self.spatial_decoder = SpatialTransposedCNN(
            spatial_channels[::-1],  # Reverse
            input_channels,
            normalization
        )
    
    def _calculate_cnn_output_shape(self):
        """Calculate spatial shape after CNN"""
        d, h, w = self.input_shape
        
        # After conv1 (stride 1)
        d1, h1 = d, h
        
        # After conv2 (stride 2 in spatial dims if exists)
        if len(self.spatial_channels) > 1:
            d2 = ((d1 + 2 * 1 - 3) // 2) + 1
            h2 = ((h1 + 2 * 1 - 3) // 2) + 1
            return (d2, h2)
        else:
            return (d1, h1)
    
    def encode(self, x):
        """
        Encode input to latent representation
        
        Args:
            x: [B, 50, 7, 5, 250] - Input EEG
        
        Returns:
            embedding: [B, latent_dim] - Latent representation
            spatial_features: For visualization/analysis
        """
        batch_size = x.size(0)
        
        # 1. Spatial CNN: [B, 50, 7, 5, 250] -> [B, C, D', H', 250]
        spatial_features = self.spatial_encoder(x)
        
        # 2. Reshape for temporal processing
        # [B, C, D', H', T] -> [B, T, C*D'*H']
        B, C, D, H, T = spatial_features.shape
        spatial_features_flat = spatial_features.permute(0, 4, 1, 2, 3)  # [B, T, C, D, H]
        spatial_features_flat = spatial_features_flat.reshape(B, T, C * D * H)
        
        # 3. Project to transformer dimension: [B, T, C*D*H] -> [B, T, d_model]
        temporal_input = self.spatial_to_temporal(spatial_features_flat)
        
        # 4. Add positional encoding
        temporal_input = self.pos_encoding(temporal_input)
        
        # 5. Temporal transformer: [B, T, d_model] -> [B, T, d_model]
        temporal_features = self.temporal_encoder(temporal_input)
        
        # 6. Flatten and bottleneck: [B, T, d_model] -> [B, latent_dim]
        temporal_flat = temporal_features.reshape(batch_size, -1)
        embedding = self.fc_encoder(temporal_flat)
        
        return embedding, spatial_features
    
    def decode(self, embedding):
        """
        Decode from latent representation to output
        
        Args:
            embedding: [B, latent_dim] - Latent representation
        
        Returns:
            reconstruction: [B, 50, 7, 5, 250] - Reconstructed EEG
        """
        batch_size = embedding.size(0)
        
        # 1. Expand from bottleneck: [B, latent_dim] -> [B, T*d_model]
        temporal_flat = self.fc_decoder(embedding)
        
        # 2. Reshape to temporal sequence: [B, T*d_model] -> [B, T, d_model]
        temporal_features = temporal_flat.reshape(batch_size, self.input_shape[2], self.d_model)
        
        # 3. Add positional encoding
        temporal_features = self.pos_encoding(temporal_features)
        
        # 4. Temporal transformer decoder: [B, T, d_model] -> [B, T, d_model]
        temporal_output = self.temporal_decoder(temporal_features)
        
        # 5. Project back to spatial features: [B, T, d_model] -> [B, T, C*D*H]
        spatial_features_flat = self.temporal_to_spatial(temporal_output)
        
        # 6. Reshape to spatial structure: [B, T, C*D*H] -> [B, C, D, H, T]
        C = self.spatial_channels[-1]
        D, H = self.encoded_spatial_shape
        spatial_features = spatial_features_flat.reshape(batch_size, self.input_shape[2], C, D, H)
        spatial_features = spatial_features.permute(0, 2, 3, 4, 1)  # [B, C, D, H, T]
        
        # 7. Spatial transposed CNN: [B, C, D, H, T] -> [B, 50, 7, 5, 250]
        reconstruction = self.spatial_decoder(spatial_features)
        
        return reconstruction
    
    def forward(self, x):
        """Full forward pass"""
        embedding, _ = self.encode(x)
        reconstruction = self.decode(embedding)
        return reconstruction, embedding
    
    def count_parameters(self):
        """Count parameters in each component"""
        spatial_enc_params = sum(p.numel() for p in self.spatial_encoder.parameters())
        transformer_enc_params = sum(p.numel() for p in self.temporal_encoder.parameters()) + \
                                sum(p.numel() for p in self.pos_encoding.parameters())
        bottleneck_params = sum(p.numel() for p in self.fc_encoder.parameters()) + \
                           sum(p.numel() for p in self.fc_decoder.parameters())
        transformer_dec_params = sum(p.numel() for p in self.temporal_decoder.parameters())
        spatial_dec_params = sum(p.numel() for p in self.spatial_decoder.parameters())
        
        projection_params = sum(p.numel() for p in self.spatial_to_temporal.parameters()) + \
                           sum(p.numel() for p in self.temporal_to_spatial.parameters())
        
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'spatial_cnn': spatial_enc_params + spatial_dec_params,
            'temporal_transformer': transformer_enc_params + transformer_dec_params,
            'projection_layers': projection_params,
            'bottleneck': bottleneck_params,
            'total': total_params
        }
    
if __name__ == "__main__":
    # Example usage
    model = CNNTTAE(
        input_channels=50,
        spatial_channels=[64, 96],
        d_model=128,
        nhead=8,
        num_transformer_layers=2,
        latent_dim=128,
        normalization='batch',
        dropout=0.1,
        input_shape=(7, 5, 250)
    )
    print(model.count_parameters())

    # Test with dummy data
    x = torch.randn(2, 50, 7, 5, 250)  # Batch of 2 samples
    recon, embed = model(x)
    print("Input shape:", x.shape)
    print("Reconstruction shape:", recon.shape)
    print("Embedding shape:", embed.shape)
