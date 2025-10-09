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
        '''
        ConvNeXt block for 3D data.
        
        Architecture:
        1. Depthwise 3D convolution (7x7x7 by default)
        2. LayerNorm
        3. Pointwise expansion (1x1x1)
        4. GELU activation
        5. Pointwise projection (1x1x1)
        6. Layer scaling
        7. Stochastic depth
        '''
        super().__init__()
        expanded_features = in_features * expansion
        
        self.block = nn.Sequential(
            # Depthwise 3D convolution with large kernel
            nn.Conv3d(
                in_features, 
                in_features, 
                kernel_size=kernel_size, 
                padding=kernel_size // 2, 
                groups=in_features,
                bias=True
            ),
            # LayerNorm (using GroupNorm with num_groups=1)
            nn.GroupNorm(num_groups=1, num_channels=in_features),
            # Pointwise expansion (inverted bottleneck)
            nn.Conv3d(in_features, expanded_features, kernel_size=1),
            nn.GELU(),
            # Pointwise projection back to input dimensions
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


class Conv3DVAE(nn.Module):
    def __init__(
        self, 
        in_channels=50, 
        latent_dim=128, 
        hidden_dims=None,
        use_convnext=False,
        convnext_expansion=4,
        drop_p=0.0
    ):
        '''
        Convolutional Variational Autoencoder for 5D tensors with optional ConvNeXt blocks.
        
        Args:
            in_channels: Number of input channels (default: 50)
            latent_dim: Dimensionality of latent space (default: 128)
            hidden_dims: List of channel dimensions for encoder layers (default: [32, 64, 128])
            use_convnext: Whether to use ConvNeXt blocks instead of classic conv blocks (default: False)
            convnext_expansion: Expansion ratio for ConvNeXt blocks (default: 4)
            drop_p: Stochastic depth drop probability (default: 0.0)
        
        Input shape: (batch, in_channels, 7, 5, 250)
        '''
        super(Conv3DVAE, self).__init__()
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.use_convnext = use_convnext
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]
        
        self.hidden_dims = hidden_dims
        
        # Track spatial dimensions at each encoder layer
        self.encoder_spatial_dims = self._compute_encoder_spatial_dims((7, 5, 250), len(hidden_dims))
        
        # ENCODER - Build dynamically based on hidden_dims
        encoder_layers = []
        prev_channels = in_channels
        
        # Create drop path probabilities
        drop_probs = [x.item() for x in torch.linspace(0, drop_p, len(hidden_dims))]
        
        for i, h_dim in enumerate(hidden_dims):
            # Downsampling layer (or stride-1 conv for first layer)
            if i == 0:
                encoder_layers.extend([
                    nn.Conv3d(prev_channels, h_dim, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=1, num_channels=h_dim),
                    nn.GELU()
                ])
            else:
                # Separate downsampling (ConvNeXt style)
                encoder_layers.extend([
                    nn.GroupNorm(num_groups=1, num_channels=prev_channels),
                    nn.Conv3d(prev_channels, h_dim, kernel_size=2, stride=2)
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
                    nn.GroupNorm(num_groups=1, num_channels=h_dim),
                    nn.GELU()
                ])
            
            prev_channels = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate flattened dimension after encoder
        final_spatial = self.encoder_spatial_dims[-1]
        self.flatten_dim = hidden_dims[-1] * final_spatial[0] * final_spatial[1] * final_spatial[2]
        
        # Latent space mapping
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        
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
                    nn.GroupNorm(num_groups=1, num_channels=current_channels),
                    nn.GELU()
                ])
            
            # Upsampling
            spatial_idx_in = len(self.encoder_spatial_dims) - 1 - i
            spatial_idx_out = spatial_idx_in - 1
            
            input_spatial = self.encoder_spatial_dims[spatial_idx_in]
            target_spatial = self.encoder_spatial_dims[spatial_idx_out]
            
            output_padding = self._compute_output_padding(input_spatial, target_spatial)
            
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
                nn.GroupNorm(num_groups=1, num_channels=reversed_hidden_dims[-1]),
                nn.GELU()
            ])
        
        # Final layer to reconstruct original input channels
        decoder_layers.append(
            nn.Conv3d(reversed_hidden_dims[-1], in_channels, kernel_size=3, stride=1, padding=1)
        )
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def _compute_encoder_spatial_dims(self, input_shape, num_layers):
        '''Compute spatial dimensions at each encoder layer'''
        dims = [input_shape]
        h, w, d = input_shape
        
        for i in range(num_layers):
            if i == 0:
                h_new, w_new, d_new = h, w, d
            else:
                h_new = (h + 2 * 0 - 2) // 2 + 1
                w_new = (w + 2 * 0 - 2) // 2 + 1
                d_new = (d + 2 * 0 - 2) // 2 + 1
            
            h, w, d = h_new, w_new, d_new
            dims.append((h, w, d))
        
        return dims
    
    def _compute_output_padding(self, input_spatial, target_spatial):
        '''Calculate output padding for transposed convolution'''
        out_pad = []
        for inp, targ in zip(input_spatial, target_spatial):
            expected = (inp - 1) * 2 - 2 * 0 + 2
            pad = targ - expected
            out_pad.append(pad)
        
        return tuple(out_pad)
    
    def encode(self, x):
        '''Encode input to latent space parameters'''
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        '''Reparameterization trick: z = mu + sigma * epsilon'''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        '''Decode latent vector to reconstruction'''
        h = self.decoder_input(z)
        final_spatial = self.encoder_spatial_dims[-1]
        h = h.view(-1, self.hidden_dims[-1], *final_spatial)
        reconstruction = self.decoder(h)
        return reconstruction
    
    def forward(self, x):
        '''Full forward pass through VAE'''
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
