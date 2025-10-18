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
        drop_p=0.0
    ):
        """
        3D Convolutional Autoencoder
        
        Args:
            in_channels: Number of input channels (default: 25)
            input_spatial: Spatial dimensions as (H, W, D) tuple (default: (7, 5, 64))
            latent_dim: Size of the latent embedding space (default: 128)
            hidden_dims: List of channel dimensions for encoder layers (default: [64, 128])
            use_convnext: Whether to use ConvNeXt blocks instead of classic conv blocks (default: False)
            convnext_expansion: Expansion ratio for ConvNeXt blocks (default: 4)
            drop_p: Stochastic depth drop probability (default: 0.0)
        """
        super(Conv3DAE, self).__init__()

        self.in_channels = in_channels
        self.input_spatial = input_spatial  # NEW: Store input spatial dimensions
        self.embedding_dim = latent_dim
        self.use_convnext = use_convnext
        
        # Default hidden dimensions if not provided - reduced from 3 to 2 layers for smaller input
        if hidden_dims is None:
            hidden_dims = [64, 128]
        
        self.hidden_dims = hidden_dims
        
        # Track spatial dimensions at each encoder layer
        self.encoder_spatial_dims = self._compute_encoder_spatial_dims(input_spatial, len(hidden_dims))
        
        # ENCODER - Build dynamically based on hidden_dims
        encoder_layers = []
        prev_channels = in_channels
        
        # Create drop path probabilities
        drop_probs = [x.item() for x in torch.linspace(0, drop_p, len(hidden_dims))]
        
        for i, h_dim in enumerate(hidden_dims):
            # Downsampling layer (or stride-1 conv for first layer)
            if i == 0:
                if use_convnext:
                    encoder_layers.extend([
                        nn.Conv3d(prev_channels, h_dim, kernel_size=3, stride=1, padding=1),
                        nn.GroupNorm(num_groups=1, num_channels=h_dim),
                        nn.GELU()
                    ])
                else:
                    encoder_layers.extend([
                        nn.Conv3d(prev_channels, h_dim, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm3d(h_dim),
                        nn.ReLU()
                    ])
            else:
                # Separate downsampling
                if use_convnext:
                    encoder_layers.extend([
                        nn.GroupNorm(num_groups=1, num_channels=prev_channels),
                        nn.Conv3d(prev_channels, h_dim, kernel_size=2, stride=2)
                    ])
                else:
                    encoder_layers.extend([
                        nn.Conv3d(prev_channels, h_dim, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm3d(h_dim),
                        nn.ReLU()
                    ])
            
            # Add ConvNeXt block or standard conv block
            if use_convnext:
                encoder_layers.append(
                    ConvNeXtBlock3D(
                        h_dim, 
                        expansion=convnext_expansion,
                        drop_p=drop_probs[i]
                    )
                )
            else:
                encoder_layers.extend([
                    nn.Conv3d(h_dim, h_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.ReLU()
                ])
            
            prev_channels = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate flattened size after encoder
        final_spatial = self.encoder_spatial_dims[-1]
        self.flat_size = hidden_dims[-1] * final_spatial[0] * final_spatial[1] * final_spatial[2]
        
        # Bottleneck (embedding space)
        self.fc_encoder = nn.Linear(self.flat_size, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, self.flat_size)
        
        # DECODER - Build dynamically (reverse of encoder)
        decoder_layers = []
        reversed_hidden_dims = list(reversed(hidden_dims))
        
        for i in range(len(reversed_hidden_dims) - 1):
            current_channels = reversed_hidden_dims[i]
            next_channels = reversed_hidden_dims[i + 1]
            
            # Add ConvNeXt block or standard conv block
            if use_convnext:
                decoder_layers.append(
                    ConvNeXtBlock3D(
                        current_channels,
                        expansion=convnext_expansion,
                        drop_p=drop_probs[len(reversed_hidden_dims) - 1 - i]
                    )
                )
            else:
                decoder_layers.extend([
                    nn.Conv3d(current_channels, current_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(current_channels),
                    nn.ReLU()
                ])
            
            # Upsampling
            spatial_idx_in = len(self.encoder_spatial_dims) - 1 - i
            spatial_idx_out = spatial_idx_in - 1
            
            input_spatial = self.encoder_spatial_dims[spatial_idx_in]
            target_spatial = self.encoder_spatial_dims[spatial_idx_out]
            
            output_padding = self._compute_output_padding(input_spatial, target_spatial)
            
            if use_convnext:
                decoder_layers.extend([
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
                decoder_layers.extend([
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
        
        # Final ConvNeXt or conv block
        if use_convnext:
            decoder_layers.append(
                ConvNeXtBlock3D(
                    reversed_hidden_dims[-1],
                    expansion=convnext_expansion,
                    drop_p=drop_probs[0]
                )
            )
        else:
            decoder_layers.extend([
                nn.Conv3d(reversed_hidden_dims[-1], reversed_hidden_dims[-1], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(reversed_hidden_dims[-1]),
                nn.ReLU()
            ])
        
        # Final layer to reconstruct original input channels
        decoder_layers.append(
            nn.Conv3d(reversed_hidden_dims[-1], in_channels, kernel_size=3, stride=1, padding=1)
        )
        
        self.decoder = nn.Sequential(*decoder_layers)


    def _compute_encoder_spatial_dims(self, input_shape, num_layers):
        """Compute spatial dimensions at each encoder layer"""
        dims = [input_shape]
        h, w, d = input_shape
        
        for i in range(num_layers):
            if i == 0:
                h_new, w_new, d_new = h, w, d
            else:
                if self.use_convnext:
                    # ConvNeXt uses kernel_size=2, stride=2 for downsampling
                    h_new = (h + 2 * 0 - 2) // 2 + 1
                    w_new = (w + 2 * 0 - 2) // 2 + 1
                    d_new = (d + 2 * 0 - 2) // 2 + 1
                else:
                    # Classic conv uses kernel_size=3, stride=2, padding=1
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
                # ConvNeXt: kernel_size=2, stride=2, padding=0
                expected = (inp - 1) * 2 - 2 * 0 + 2
            else:
                # Classic: kernel_size=3, stride=2, padding=1
                expected = (inp - 1) * 2 - 2 * 1 + 3
            pad = targ - expected
            out_pad.append(pad)
        
        return tuple(out_pad)


    def encode(self, x):
        """Encode input to latent embedding"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        embedding = self.fc_encoder(x)
        return embedding


    def decode(self, embedding):
        """Decode from latent embedding to output"""
        x = self.fc_decoder(embedding)
        final_spatial = self.encoder_spatial_dims[-1]
        x = x.view(-1, self.hidden_dims[-1], *final_spatial)  # Reshape
        x = self.decoder(x)
        return x


    def forward(self, x):
        """Full forward pass through autoencoder"""
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return reconstruction

if __name__ == "__main__":
    model = Conv3DAE(use_convnext=True, drop_p=0.1)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Quick forward pass to verify shapes
    bs = 1
    c = model.in_channels
    h, w, d = model.input_spatial
    x = torch.randn(bs, c, h, w, d)
    with torch.no_grad():
        out = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)