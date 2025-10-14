import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DAE(nn.Module):
    """
    Flexible 3D Convolutional Autoencoder with independent encoder/decoder
    
    Control everything:
    - Encoder depth and channels
    - Decoder depth and channels  
    - Embedding size
    - Normalization type
    """
    
    def __init__(
        self,
        input_channels=25,
        encoder_channels=[64, 128, 256],      # Encoder architecture
        decoder_channels=[256, 128, 64],      # Decoder architecture (independent!)
        latent_dim=128,
        normalization='group',  # 'batch' or 'group'
        num_groups=8,
        input_shape=(7, 5, 250),
        dropout=0.0
    ):
        """
        Args:
            input_channels: Number of input channels (e.g., 50 frequencies)
            encoder_channels: List of channels for ENCODER layers
                             [64, 128, 256] = 3 encoder layers
            decoder_channels: List of channels for DECODER layers (can be different!)
                             [256, 192, 128, 64] = 4 decoder layers
            latent_dim: Size of bottleneck embedding
            normalization: 'batch' or 'group'
            num_groups: Number of groups for GroupNorm
            input_shape: Spatial dimensions (D, H, W)
            dropout: Dropout probability in bottleneck
        """
        super(Conv3DAE, self).__init__()
        
        self.input_channels = input_channels
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.latent_dim = latent_dim
        self.normalization = normalization
        self.num_groups = num_groups
        self.input_shape = input_shape
        self.dropout = dropout
        
        # Validate decoder input matches encoder output
        if decoder_channels[0] != encoder_channels[-1]:
            raise ValueError(
                f"Decoder first channel ({decoder_channels[0]}) must match "
                f"encoder last channel ({encoder_channels[-1]})"
            )
        
        # ===================== ENCODER =====================
        self.encoder_blocks = nn.ModuleList()
        in_ch = input_channels
        
        for i, out_ch in enumerate(encoder_channels):
            stride = 1 if i == 0 else 2
            self.encoder_blocks.append(
                self._make_conv_block(in_ch, out_ch, stride)
            )
            in_ch = out_ch
        
        # Calculate encoded spatial shape
        self.encoded_shape = self._calculate_encoded_shape()
        self.flat_size = encoder_channels[-1] * \
                        self.encoded_shape[0] * \
                        self.encoded_shape[1] * \
                        self.encoded_shape[2]
        
        # ===================== BOTTLENECK =====================
        self.fc_encoder = nn.Linear(self.flat_size, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, self.flat_size)
        
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = nn.Identity()
        
        # ===================== DECODER =====================
        
        # Calculate target shapes for decoder upsampling
        self.decoder_target_shapes = self._calculate_decoder_targets()
        
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(len(decoder_channels) - 1):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1]
            
            # Determine stride and output_padding
            current_shape = self.decoder_target_shapes[i]
            target_shape = self.decoder_target_shapes[i + 1]
            
            # Check if we need to upsample
            needs_upsample = any(t > c for c, t in zip(current_shape, target_shape))
            
            if needs_upsample:
                stride = 2
                output_padding = self._calculate_output_padding(
                    current_shape, target_shape, stride
                )
            else:
                stride = 1
                output_padding = (0, 0, 0)
            
            self.decoder_blocks.append(
                self._make_deconv_block(in_ch, out_ch, stride, output_padding)
            )
        
        # Final reconstruction layer (always stride 1)
        self.final_conv = nn.Conv3d(
            decoder_channels[-1],
            input_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
    
    def _make_conv_block(self, in_channels, out_channels, stride):
        """Create encoder convolutional block"""
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1)
        ]
        
        if self.normalization == 'batch':
            layers.append(nn.BatchNorm3d(out_channels))
        elif self.normalization == 'group':
            groups = self._get_num_groups(out_channels)
            layers.append(nn.GroupNorm(groups, out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def _make_deconv_block(self, in_channels, out_channels, stride, output_padding):
        """Create decoder deconvolutional block"""
        layers = []
        
        if stride > 1:
            layers.append(
                nn.ConvTranspose3d(
                    in_channels, out_channels,
                    kernel_size=3, stride=stride, padding=1,
                    output_padding=output_padding
                )
            )
        else:
            layers.append(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                         stride=1, padding=1)
            )
        
        if self.normalization == 'batch':
            layers.append(nn.BatchNorm3d(out_channels))
        elif self.normalization == 'group':
            groups = self._get_num_groups(out_channels)
            layers.append(nn.GroupNorm(groups, out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def _get_num_groups(self, num_channels):
        """Calculate valid number of groups for GroupNorm"""
        groups = min(self.num_groups, num_channels)
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return groups
    
    def _calculate_encoded_shape(self):
        """Calculate spatial shape after encoder"""
        d, h, w = self.input_shape
        
        for i in range(len(self.encoder_channels)):
            stride = 1 if i == 0 else 2
            d = ((d + 2 * 1 - 3) // stride) + 1
            h = ((h + 2 * 1 - 3) // stride) + 1
            w = ((w + 2 * 1 - 3) // stride) + 1
        
        return (d, h, w)
    
    def _calculate_decoder_targets(self):
        """
        Calculate target spatial shapes for each decoder layer
        Works backwards from input_shape to encoded_shape
        """
        # Calculate all encoder shapes (forward)
        encoder_shapes = [self.input_shape]
        d, h, w = self.input_shape
        
        for i in range(len(self.encoder_channels)):
            stride = 1 if i == 0 else 2
            d = ((d + 2 * 1 - 3) // stride) + 1
            h = ((h + 2 * 1 - 3) // stride) + 1
            w = ((w + 2 * 1 - 3) // stride) + 1
            encoder_shapes.append((d, h, w))
        
        # Reverse for decoder (decoder shape progression)
        # We need to map decoder depth to encoder depth
        num_decoder_layers = len(self.decoder_channels)
        num_encoder_layers = len(self.encoder_channels)
        
        # Start from encoded shape
        decoder_shapes = [self.encoded_shape]
        
        # Interpolate between encoded shape and input shape
        # based on number of decoder layers
        if num_decoder_layers >= num_encoder_layers:
            # More or equal decoder layers: use encoder shapes in reverse
            reverse_encoder = encoder_shapes[::-1]
            step = len(reverse_encoder) / num_decoder_layers
            
            for i in range(1, num_decoder_layers):
                idx = min(int(i * step), len(reverse_encoder) - 1)
                decoder_shapes.append(reverse_encoder[idx])
        else:
            # Fewer decoder layers: sample from encoder shapes
            reverse_encoder = encoder_shapes[::-1]
            indices = [int(i * (len(reverse_encoder) - 1) / (num_decoder_layers - 1)) 
                      for i in range(num_decoder_layers)]
            decoder_shapes = [reverse_encoder[i] for i in indices]
        
        # Ensure final shape matches input
        decoder_shapes.append(self.input_shape)
        
        return decoder_shapes
    
    def _calculate_output_padding(self, input_shape, target_shape, stride):
        """Calculate output_padding for ConvTranspose3d"""
        op_d = target_shape[0] - ((input_shape[0] - 1) * stride - 2 * 1 + 3)
        op_h = target_shape[1] - ((input_shape[1] - 1) * stride - 2 * 1 + 3)
        op_w = target_shape[2] - ((input_shape[2] - 1) * stride - 2 * 1 + 3)
        
        return (max(0, op_d), max(0, op_h), max(0, op_w))
    
    def encode(self, x):
        """Encode input to latent embedding"""
        for block in self.encoder_blocks:
            x = block(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout_layer(x)
        embedding = self.fc_encoder(x)
        
        return embedding
    
    def decode(self, embedding):
        """Decode from latent embedding to output"""
        x = self.fc_decoder(embedding)
        x = self.dropout_layer(x)
        x = x.view(-1, self.encoder_channels[-1], *self.encoded_shape)
        
        for block in self.decoder_blocks:
            x = block(x)
        
        x = self.final_conv(x)
        return x
    
    def forward(self, x):
        """Full forward pass"""
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return reconstruction, embedding
    
if __name__ == "__main__":
    # Example usage
    model = Conv3DAE(
        input_channels=25,
        encoder_channels=[64, 256],
        decoder_channels=[128, 256, 128, 64],
        latent_dim=128,
        normalization='batch',
        num_groups=8,
        input_shape=(7, 5, 250),
        dropout=0.1
    )
    
    print(model)
    
    # Test with dummy data
    x = torch.randn(2, 25, 7, 5, 250)  # Batch of 2 samples
    recon, embed = model(x)
    print("Input shape:", x.shape)
    print("Reconstruction shape:", recon.shape)
    print("Embedding shape:", embed.shape)