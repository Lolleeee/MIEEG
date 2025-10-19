import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth


class SeparableConv3D(nn.Module):
    """Depthwise Separable 3D Convolution (Depthwise + Pointwise)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # Depthwise: each input channel convolved separately
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=in_channels,  # Key: groups=in_channels
            bias=False
        )
        # Pointwise: 1x1x1 conv to mix channels
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class FactorizedConv3D(nn.Module):
    """(2+1)D Factorized Convolution: 2D spatial + 1D temporal"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 spatial_kernel=(3, 3), temporal_kernel=3):
        super().__init__()
        # Intermediate channels for factorization
        mid_channels = (in_channels + out_channels) // 2
        
        # 2D spatial convolution (H x W dimensions)
        spatial_pad = tuple(k // 2 for k in spatial_kernel)
        self.spatial_conv = nn.Conv3d(
            in_channels, mid_channels,
            kernel_size=(spatial_kernel[0], spatial_kernel[1], 1),
            stride=(stride, stride, 1),
            padding=(*spatial_pad, 0),
            bias=False
        )
        self.spatial_norm = nn.GroupNorm(num_groups=1, num_channels=mid_channels)
        self.spatial_act = nn.GELU()
        
        # 1D temporal convolution (T dimension)
        temporal_pad = temporal_kernel // 2
        self.temporal_conv = nn.Conv3d(
            mid_channels, out_channels,
            kernel_size=(1, 1, temporal_kernel),
            stride=(1, 1, stride if stride > 1 else 1),
            padding=(0, 0, temporal_pad),
            bias=False
        )
        
    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.spatial_norm(x)
        x = self.spatial_act(x)
        x = self.temporal_conv(x)
        return x


class InvertedResidualBlock3D(nn.Module):
    """MobileNet-style Inverted Residual with Expansion"""
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        stride=1,
        expansion=4,
        use_factorized=True,
        drop_p=0.0
    ):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expansion
        
        layers = []
        
        # Expansion phase (pointwise)
        if expansion != 1:
            layers.extend([
                nn.Conv3d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
                nn.GELU()
            ])
        
        # Depthwise/Factorized convolution
        if use_factorized:
            # Use (2+1)D for efficiency
            layers.append(
                FactorizedConv3D(
                    hidden_dim, hidden_dim,
                    stride=stride,
                    spatial_kernel=(3, 3),
                    temporal_kernel=3
                )
            )
        else:
            # Use depthwise separable
            layers.extend([
                nn.Conv3d(
                    hidden_dim, hidden_dim, 
                    kernel_size=3, 
                    stride=stride, 
                    padding=1, 
                    groups=hidden_dim,
                    bias=False
                ),
                nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
                nn.GELU()
            ])
        
        # Projection phase (pointwise)
        layers.extend([
            nn.Conv3d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        self.drop_path = StochasticDepth(drop_p, mode="batch") if drop_p > 0 else nn.Identity()
        
    def forward(self, x):
        if self.use_residual:
            return x + self.drop_path(self.conv(x))
        else:
            return self.conv(x)


class Conv3DAE(nn.Module):
    def __init__(
        self, 
        in_channels=25, 
        input_spatial=(7, 5, 64),
        latent_dim=128,
        hidden_dims=None,
        expansion=4,
        use_factorized=True,
        drop_p=0.0
    ):
        """
        Efficient 3D Convolutional Autoencoder with:
        - Factorized (2+1)D convolutions
        - Depthwise separable convolutions
        - Inverted residual bottlenecks
        
        Args:
            in_channels: Input channels (25 for your wavelet EEG)
            input_spatial: (H, W, D) = (7, 5, 64)
            latent_dim: Bottleneck embedding size
            hidden_dims: Channel dimensions per stage
            expansion: Expansion ratio for inverted bottleneck
            use_factorized: Use (2+1)D factorization vs depthwise separable
            drop_p: Stochastic depth probability
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.input_spatial = input_spatial
        self.embedding_dim = latent_dim
        self.use_factorized = use_factorized
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 96]  # Smaller than before for efficiency
        
        self.hidden_dims = hidden_dims
        self.encoder_spatial_dims = self._compute_encoder_spatial_dims(input_spatial, len(hidden_dims))
        
        # ========== ENCODER ==========
        encoder_blocks = []
        drop_probs = [x.item() for x in torch.linspace(0, drop_p, len(hidden_dims))]
        
        # Stem: lightweight initial convolution
        encoder_blocks.append(
            nn.Sequential(
                nn.Conv3d(in_channels, hidden_dims[0], kernel_size=(3, 3, 3), 
                         stride=1, padding=(1, 1, 1), bias=False),
                nn.GroupNorm(num_groups=1, num_channels=hidden_dims[0]),
                nn.GELU()
            )
        )
        
        # Inverted residual blocks with downsampling
        for i in range(len(hidden_dims)):
            in_ch = hidden_dims[i]
            out_ch = hidden_dims[i] if i == len(hidden_dims) - 1 else hidden_dims[i + 1]
            
            # Main inverted residual block
            encoder_blocks.append(
                InvertedResidualBlock3D(
                    in_ch, in_ch,
                    stride=1,
                    expansion=expansion,
                    use_factorized=use_factorized,
                    drop_p=drop_probs[i]
                )
            )
            
            # Downsampling transition (except last layer)
            if i < len(hidden_dims) - 1:
                encoder_blocks.append(
                    InvertedResidualBlock3D(
                        in_ch, out_ch,
                        stride=2,
                        expansion=expansion,
                        use_factorized=use_factorized,
                        drop_p=drop_probs[i]
                    )
                )
        
        self.encoder = nn.Sequential(*encoder_blocks)
        
        # Bottleneck
        final_spatial = self.encoder_spatial_dims[-1]
        self.flat_size = hidden_dims[-1] * final_spatial[0] * final_spatial[1] * final_spatial[2]

        print(f"Debug - Final spatial: {final_spatial}, Flat size: {self.flat_size}")  # Debug line

        self.fc_encoder = nn.Sequential(
            nn.Linear(self.flat_size, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

            
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, self.flat_size)
        )
        
        # ========== DECODER ==========
        decoder_blocks = []
        reversed_hidden_dims = list(reversed(hidden_dims))

        for i in range(len(reversed_hidden_dims)):
            in_ch = reversed_hidden_dims[i]
            out_ch = reversed_hidden_dims[i + 1] if i < len(reversed_hidden_dims) - 1 else hidden_dims[0]
            
            # Main inverted residual block
            decoder_blocks.append(
                InvertedResidualBlock3D(
                    in_ch, in_ch,
                    stride=1,
                    expansion=expansion,
                    use_factorized=use_factorized,
                    drop_p=drop_probs[len(reversed_hidden_dims) - 1 - i]
                )
            )
            
            # Upsampling transition (except last layer)
            if i < len(reversed_hidden_dims) - 1:
                # Get target spatial dimensions for exact upsampling
                target_spatial = self.encoder_spatial_dims[-(i + 2)]
                
                # Use trilinear interpolation + convolution for exact size matching
                decoder_blocks.extend([
                    nn.Upsample(size=target_spatial, mode='trilinear', align_corners=False),
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(num_groups=1, num_channels=out_ch),
                    nn.GELU()
                ])

        # Reconstruction head
        decoder_blocks.append(
            nn.Conv3d(hidden_dims[0], in_channels, kernel_size=1)
        )

        self.decoder = nn.Sequential(*decoder_blocks)

        
    def _compute_encoder_spatial_dims(self, input_shape, num_layers):
        """
        Compute spatial dimensions at each encoder stage.
        Tracks dimensions after each downsampling operation.
        """
        dims = [input_shape]  # Initial input
        h, w, d = input_shape
        
        # After stem convolution (stride=1, preserves dimensions with padding=1)
        # No change: (h+2*1-3)//1+1 = h
        
        # Track dimensions after each downsampling stage
        for i in range(num_layers - 1):  # Only downsampling stages
            if self.use_factorized:
                # FactorizedConv3D with stride=2:
                # Spatial conv: kernel=(3,3,1), stride=(2,2,1), padding=(1,1,0)
                # Temporal conv: kernel=(1,1,3), stride=(1,1,2), padding=(0,0,1)
                h = (h + 2*1 - 3) // 2 + 1  # Spatial dimension
                w = (w + 2*1 - 3) // 2 + 1  # Spatial dimension  
                d = (d + 2*1 - 3) // 2 + 1  # Temporal dimension
            else:
                # Depthwise separable: kernel=3, stride=2, padding=1 (uniform)
                h = (h + 2*1 - 3) // 2 + 1
                w = (w + 2*1 - 3) // 2 + 1
                d = (d + 2*1 - 3) // 2 + 1
            
            dims.append((h, w, d))
        
        return dims

    def _compute_output_padding(self, input_spatial, target_spatial):
        """
        Calculate output padding for ConvTranspose3d.
        Formula: output = (input - 1) * stride - 2*padding + kernel + output_padding
        For kernel=2, stride=2, padding=0:
            output = input * 2 + output_padding
        Therefore: output_padding = target - input * 2
        """
        out_pad = []
        for inp, targ in zip(input_spatial, target_spatial):
            expected = inp * 2  # What we get without output_padding
            pad = targ - expected  # Additional padding needed
            out_pad.append(max(0, pad))
        return tuple(out_pad)

    
    def encode(self, x):
        """Encode to latent embedding"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        embedding = self.fc_encoder(x)
        return embedding
    
    def decode(self, embedding):
        """Decode from latent embedding"""
        x = self.fc_decoder(embedding)
        final_spatial = self.encoder_spatial_dims[-1]
        x = x.view(-1, self.hidden_dims[-1], *final_spatial)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        """Full autoencoder forward pass"""
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return reconstruction


if __name__ == "__main__":
    import time
    
    # Compare original vs optimized
    print("=" * 60)
    print("EFFICIENT MODEL (Factorized)")
    print("=" * 60)
    model_efficient = Conv3DAE(
        use_factorized=True, 
        expansion=4, 
        drop_p=0.1,
        latent_dim=1867  # For 30x compression
    )
    
    total_params = sum(p.numel() for p in model_efficient.parameters())
    trainable_params = sum(p.numel() for p in model_efficient.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_efficient = model_efficient.to(device)
    
    bs = 4
    c = model_efficient.in_channels
    h, w, d = model_efficient.input_spatial
    x = torch.randn(bs, c, h, w, d).to(device)
    
    # Warmup
    with torch.no_grad():
        _ = model_efficient(x)
    
    # Benchmark
    n_iters = 50
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    
    with torch.no_grad():
        for _ in range(n_iters):
            out = model_efficient(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    elapsed = time.time() - start
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Embedding size: {model_efficient.embedding_dim}")
    print(f"Compression ratio: {(c * h * w * d) / model_efficient.embedding_dim:.2f}x")
    print(f"\nAverage inference time: {elapsed / n_iters * 1000:.2f} ms")
    print(f"Throughput: {bs * n_iters / elapsed:.2f} samples/sec")
