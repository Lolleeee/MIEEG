import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth



class LayerScaler(nn.Module):
    def __init__(self, init_value: float, dimensions: int):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones((dimensions)), requires_grad=True)


    def forward(self, x):
        return self.gamma[None, ..., None, None, None] * x



class ConvNeXtBlock3D(nn.Module):
    def __init__(
        self,
        in_features: int,
        expansion: int = 4,
        kernel_size: int = 7,
        drop_p: float = 0.0,
        layer_scaler_init_value: float = 1e-6,
    ):
        """ConvNeXt block for 3D data"""
        super().__init__()
        expanded_features = in_features * expansion
        
        self.block = nn.Sequential(
            nn.Conv3d(
                in_features, 
                in_features, 
                kernel_size=kernel_size, 
                padding=kernel_size // 2, 
                groups=in_features,
                bias=True
            ),
            nn.GroupNorm(num_groups=1, num_channels=in_features),
            nn.Conv3d(in_features, expanded_features, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(expanded_features, in_features, kernel_size=1),
        )
        
        self.layer_scaler = LayerScaler(layer_scaler_init_value, in_features)
        self.drop_path = StochasticDepth(drop_p, mode="batch")


    def forward(self, x):
        res = x
        x = self.block(x)
        x = self.layer_scaler(x)
        x = self.drop_path(x)
        x += res
        return x



class Conv3DAE(nn.Module):
    def __init__(
        self, 
        in_channels=25, 
        input_spatial=(7, 5, 64),  
        latent_dim=128, 
        hidden_dims=None,
        use_convnext=False,
        convnext_expansion=4,
        drop_p=0.0,
        encoder_skip_connections=None,
        decoder_skip_connections=None
    ):
        """
        3D Convolutional Autoencoder with optional skip connections
        
        Args:
            in_channels: Number of input channels (default: 25)
            input_spatial: Spatial dimensions as (H, W, D) tuple (default: (7, 5, 64))
            latent_dim: Size of the latent embedding space (default: 128)
            hidden_dims: List of channel dimensions for encoder layers (default: [64, 128])
            use_convnext: Whether to use ConvNeXt blocks instead of classic conv blocks (default: False)
            convnext_expansion: Expansion ratio for ConvNeXt blocks (default: 4)
            drop_p: Stochastic depth drop probability (default: 0.0)
            encoder_skip_connections: List of tuples (from_layer, to_layer) for encoder skip connections.
                                     Layers are 0-indexed. Example: [(0, 2)] connects layer 0 to layer 2
            decoder_skip_connections: List of tuples (from_layer, to_layer) for decoder skip connections.
                                     Layers are 0-indexed. Example: [(0, 2)] connects layer 0 to layer 2
        """
        super(Conv3DAE, self).__init__()
        self.convnext_expansion = convnext_expansion
        self.in_channels = in_channels
        self.input_spatial = input_spatial
        self.embedding_dim = latent_dim
        self.use_convnext = use_convnext
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [64, 128]
        
        self.hidden_dims = hidden_dims
        self.encoder_skip_connections = encoder_skip_connections or []
        self.decoder_skip_connections = decoder_skip_connections or []
        
        # Validate skip connections
        self._validate_skip_connections(self.encoder_skip_connections, len(hidden_dims), "encoder")
        self._validate_skip_connections(self.decoder_skip_connections, len(hidden_dims), "decoder")
        
        # Track spatial dimensions at each encoder layer
        self.encoder_spatial_dims = self._compute_encoder_spatial_dims(input_spatial, len(hidden_dims))
        
        # Build encoder with modular layers
        self.encoder_layers = nn.ModuleList()
        self._build_encoder(hidden_dims, drop_p)
        
        # Calculate flattened size after encoder
        final_spatial = self.encoder_spatial_dims[-1]
        self.flat_size = hidden_dims[-1] * final_spatial[0] * final_spatial[1] * final_spatial[2]
        
        # Bottleneck (embedding space)
        self.fc_encoder = nn.Linear(self.flat_size, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, self.flat_size)
        
        # Build decoder with modular layers
        self.decoder_layers = nn.ModuleList()
        self._build_decoder(hidden_dims, drop_p)
        
        # Create projection layers for skip connections
        self.encoder_skip_projections = nn.ModuleDict()
        self.decoder_skip_projections = nn.ModuleDict()
        self._build_skip_projections()


    def _validate_skip_connections(self, skip_conns, num_layers, prefix):
        """Validate that skip connection indices are within bounds"""
        for from_idx, to_idx in skip_conns:
            if from_idx < 0 or from_idx >= num_layers:
                raise ValueError(f"{prefix} skip connection from_layer {from_idx} out of bounds [0, {num_layers-1}]")
            if to_idx < 0 or to_idx >= num_layers:
                raise ValueError(f"{prefix} skip connection to_layer {to_idx} out of bounds [0, {num_layers-1}]")
            if from_idx >= to_idx:
                raise ValueError(f"{prefix} skip connection must go forward: from_layer ({from_idx}) < to_layer ({to_idx})")


    def _build_encoder(self, hidden_dims, drop_p):
        """Build encoder layers as separate modules"""
        prev_channels = self.in_channels
        drop_probs = [x.item() for x in torch.linspace(0, drop_p, len(hidden_dims))]
        
        for i, h_dim in enumerate(hidden_dims):
            layer_modules = []
            
            # Downsampling layer (or stride-1 conv for first layer)
            if i == 0:
                if self.use_convnext:
                    layer_modules.extend([
                        nn.Conv3d(prev_channels, h_dim, kernel_size=3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=1, num_channels=h_dim),
                        nn.GELU()
                    ])
                else:
                    layer_modules.extend([
                        nn.Conv3d(prev_channels, h_dim, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm3d(h_dim),
                        nn.ReLU()
                    ])
            else:
                if self.use_convnext:
                    layer_modules.extend([
                        nn.GroupNorm(num_groups=1, num_channels=prev_channels),
                        nn.Conv3d(prev_channels, h_dim, kernel_size=2, stride=2)
                    ])
                else:
                    layer_modules.extend([
                        nn.Conv3d(prev_channels, h_dim, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm3d(h_dim),
                        nn.ReLU()
                    ])
            
            # Add ConvNeXt block or standard conv block
            if self.use_convnext:
                layer_modules.append(
                    ConvNeXtBlock3D(h_dim, expansion=self.convnext_expansion, drop_p=drop_probs[i])
                )
            else:
                layer_modules.extend([
                    nn.Conv3d(h_dim, h_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.ReLU()
                ])
            
            self.encoder_layers.append(nn.Sequential(*layer_modules))
            prev_channels = h_dim


    def _build_decoder(self, hidden_dims, drop_p):
        """Build decoder layers as separate modules"""
        reversed_hidden_dims = list(reversed(hidden_dims))
        drop_probs = [x.item() for x in torch.linspace(0, drop_p, len(hidden_dims))]
        
        for i in range(len(reversed_hidden_dims)):
            current_channels = reversed_hidden_dims[i]
            layer_modules = []
            
            # Add ConvNeXt block or standard conv block
            if self.use_convnext:
                layer_modules.append(
                    ConvNeXtBlock3D(
                        current_channels,
                        expansion=self.convnext_expansion,
                        drop_p=drop_probs[len(reversed_hidden_dims) - 1 - i]
                    )
                )
            else:
                layer_modules.extend([
                    nn.Conv3d(current_channels, current_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(current_channels),
                    nn.ReLU()
                ])
            
            # Upsampling (except for last layer)
            if i < len(reversed_hidden_dims) - 1:
                next_channels = reversed_hidden_dims[i + 1]
                
                spatial_idx_in = len(self.encoder_spatial_dims) - 1 - i
                spatial_idx_out = spatial_idx_in - 1
                
                input_spatial = self.encoder_spatial_dims[spatial_idx_in]
                target_spatial = self.encoder_spatial_dims[spatial_idx_out]
                
                output_padding = self._compute_output_padding(input_spatial, target_spatial)
                
                if self.use_convnext:
                    layer_modules.extend([
                        nn.GroupNorm(num_groups=1, num_channels=current_channels),
                        nn.ConvTranspose3d(
                            current_channels,
                            next_channels,
                            kernel_size=2,
                            stride=2,
                            output_padding=output_padding
                        )
                    ])
                else:
                    layer_modules.extend([
                        nn.ConvTranspose3d(
                            current_channels,
                            next_channels,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=output_padding
                        ),
                        nn.BatchNorm3d(next_channels),
                        nn.ReLU()
                    ])
            
            self.decoder_layers.append(nn.Sequential(*layer_modules))
        
        # Final reconstruction layer
        self.final_conv = nn.Conv3d(reversed_hidden_dims[-1], self.in_channels, kernel_size=3, stride=1, padding=1)


    def _build_skip_projections(self):
        """Create 1x1x1 convolutions to match channel dimensions for skip connections"""
        # Encoder skip projections
        for from_idx, to_idx in self.encoder_skip_connections:
            from_channels = self.hidden_dims[from_idx]
            to_channels = self.hidden_dims[to_idx]
            
            if from_channels != to_channels:
                key = f"{from_idx}_to_{to_idx}"
                self.encoder_skip_projections[key] = nn.Conv3d(
                    from_channels, to_channels, kernel_size=1, bias=False
                )
        
        # Decoder skip projections
        reversed_hidden_dims = list(reversed(self.hidden_dims))
        for from_idx, to_idx in self.decoder_skip_connections:
            from_channels = reversed_hidden_dims[from_idx]
            to_channels = reversed_hidden_dims[to_idx]
            
            if from_channels != to_channels:
                key = f"{from_idx}_to_{to_idx}"
                self.decoder_skip_projections[key] = nn.Conv3d(
                    from_channels, to_channels, kernel_size=1, bias=False
                )


    def _compute_encoder_spatial_dims(self, input_shape, num_layers):
        """Compute spatial dimensions at each encoder layer"""
        dims = [input_shape]
        h, w, d = input_shape
        
        for i in range(num_layers):
            if i == 0:
                h_new, w_new, d_new = h, w, d
            else:
                if self.use_convnext:
                    h_new = (h + 2 * 0 - 2) // 2 + 1
                    w_new = (w + 2 * 0 - 2) // 2 + 1
                    d_new = (d + 2 * 0 - 2) // 2 + 1
                else:
                    h_new = (h + 2 * 1 - 3) // 2 + 1
                    w_new = (w + 2 * 1 - 3) // 2 + 1
                    d_new = (d + 2 * 1 - 3) // 2 + 1
            
            h, w, d = h_new, w_new, d_new
            dims.append((h, w, d))
        
        return dims
    
    def _compute_output_padding(self, input_spatial, target_spatial):
        """Calculate output padding for transposed convolution"""
        out_pad = []
        for inp, targ in zip(input_spatial, target_spatial):
            if self.use_convnext:
                expected = (inp - 1) * 2 - 2 * 0 + 2
            else:
                expected = (inp - 1) * 2 - 2 * 1 + 3
            pad = targ - expected
            out_pad.append(pad)
        
        return tuple(out_pad)


    def encode(self, x):
        """Encode input to latent embedding with skip connections"""
        skip_outputs = {}
        
        # Forward through encoder layers
        for i, layer in enumerate(self.encoder_layers):
            # Add skip connection input if this layer receives one
            for from_idx, to_idx in self.encoder_skip_connections:
                if to_idx == i and from_idx in skip_outputs:
                    skip_feat = skip_outputs[from_idx]
                    
                    # Project if channels don't match
                    key = f"{from_idx}_to_{to_idx}"
                    if key in self.encoder_skip_projections:
                        skip_feat = self.encoder_skip_projections[key](skip_feat)
                    
                    # Interpolate if spatial dimensions don't match
                    if skip_feat.shape[2:] != x.shape[2:]:
                        skip_feat = nn.functional.interpolate(
                            skip_feat, size=x.shape[2:], mode='trilinear', align_corners=False
                        )
                    
                    x = x + skip_feat
            
            x = layer(x)
            skip_outputs[i] = x
        
        x = x.view(x.size(0), -1)  # Flatten
        embedding = self.fc_encoder(x)
        return embedding


    def decode(self, embedding):
        """Decode from latent embedding to output with skip connections"""
        x = self.fc_decoder(embedding)
        final_spatial = self.encoder_spatial_dims[-1]
        x = x.view(-1, self.hidden_dims[-1], *final_spatial)  # Reshape
        
        skip_outputs = {}
        
        # Forward through decoder layers
        for i, layer in enumerate(self.decoder_layers):
            # Add skip connection input if this layer receives one
            for from_idx, to_idx in self.decoder_skip_connections:
                if to_idx == i and from_idx in skip_outputs:
                    skip_feat = skip_outputs[from_idx]
                    
                    # Project if channels don't match
                    key = f"{from_idx}_to_{to_idx}"
                    if key in self.decoder_skip_projections:
                        skip_feat = self.decoder_skip_projections[key](skip_feat)
                    
                    # Interpolate if spatial dimensions don't match
                    if skip_feat.shape[2:] != x.shape[2:]:
                        skip_feat = nn.functional.interpolate(
                            skip_feat, size=x.shape[2:], mode='trilinear', align_corners=False
                        )
                    
                    x = x + skip_feat
            
            x = layer(x)
            skip_outputs[i] = x
        
        x = self.final_conv(x)
        return x


    def forward(self, x):
        """Full forward pass through autoencoder"""
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return reconstruction


if __name__ == "__main__":
    # Example 1: No skip connections
    print("=" * 60)
    print("Example 1: Standard autoencoder (no skip connections)")
    print("=" * 60)
    model1 = Conv3DAE(use_convnext=True, drop_p=0.1, hidden_dims=[64, 128, 256])
    print(model1)
    total_params = sum(p.numel() for p in model1.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Example 2: Encoder skip connections
    print("\n" + "=" * 60)
    print("Example 2: Encoder with skip connections [0->2]")
    print("=" * 60)
    model2 = Conv3DAE(
        use_convnext=True, 
        drop_p=0.1, 
        hidden_dims=[64, 128, 256],
        encoder_skip_connections=[(0, 2)]  # Skip from layer 0 to layer 2
    )
    print(f"Encoder skip connections: {model2.encoder_skip_connections}")
    total_params = sum(p.numel() for p in model2.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Example 3: Decoder skip connections
    print("\n" + "=" * 60)
    print("Example 3: Decoder with skip connections [0->1, 1->2]")
    print("=" * 60)
    model3 = Conv3DAE(
        use_convnext=True, 
        drop_p=0.1, 
        hidden_dims=[64, 128, 256],
        decoder_skip_connections=[(0, 1), (1, 2)]  # Sequential skips
    )
    print(f"Decoder skip connections: {model3.decoder_skip_connections}")
    total_params = sum(p.numel() for p in model3.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Example 4: Both encoder and decoder skip connections
    print("\n" + "=" * 60)
    print("Example 4: Both encoder and decoder skip connections")
    print("=" * 60)
    model4 = Conv3DAE(
        use_convnext=True, 
        drop_p=0.1, 
        hidden_dims=[64, 128, 256],
        encoder_skip_connections=[(0, 2)],
        decoder_skip_connections=[(0, 2)]
    )
    print(f"Encoder skip connections: {model4.encoder_skip_connections}")
    print(f"Decoder skip connections: {model4.decoder_skip_connections}")
    total_params = sum(p.numel() for p in model4.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Quick forward pass to verify shapes
    print("\n" + "=" * 60)
    print("Forward pass test")
    print("=" * 60)
    bs = 2
    c = model4.in_channels
    h, w, d = model4.input_spatial
    x = torch.randn(bs, c, h, w, d)
    with torch.no_grad():
        out = model4(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("✓ Forward pass successful!")