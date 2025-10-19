import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth


class SqueezeExcitation3D(nn.Module):
    """Squeeze-and-Excitation block for 3D tensors"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, reduced_channels, 1, bias=True),
            nn.GELU(),
            nn.Conv3d(reduced_channels, channels, 1, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)


class DepthwiseSeparableConv3D(nn.Module):
    """
    Proper Depthwise Separable 3D Convolution with normalization and activation.
    Follows MobileNet standard: Conv -> Norm -> Activation pattern.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv3d(
                in_channels, in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=False
            ),
            nn.GroupNorm(num_groups=1, num_channels=in_channels),
            nn.GELU()
        )
        
        # Pointwise convolution
        self.pointwise = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class FactorizedConv3D(nn.Module):
    """
    (2+1)D Factorized Convolution: 2D spatial + 1D temporal.
    More parameter-efficient than full 3D convolution.
    """
    def __init__(self, in_channels, out_channels, stride=1, mid_channels=None):
        super().__init__()
        
        if mid_channels is None:
            mid_channels = max(in_channels, out_channels)
        
        # 2D spatial convolution (H x W)
        self.spatial = nn.Sequential(
            nn.Conv3d(
                in_channels, mid_channels,
                kernel_size=(3, 3, 1),
                stride=(stride, stride, 1),
                padding=(1, 1, 0),
                bias=False
            ),
            nn.GroupNorm(num_groups=1, num_channels=mid_channels),
            nn.GELU()
        )
        
        # 1D temporal convolution (T dimension)
        self.temporal = nn.Sequential(
            nn.Conv3d(
                mid_channels, out_channels,
                kernel_size=(1, 1, 3),
                stride=(1, 1, stride if stride > 1 else 1),
                padding=(0, 0, 1),
                bias=False
            ),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        x = self.spatial(x)
        x = self.temporal(x)
        return x


class InvertedResidual3D(nn.Module):
    """
    MobileNetV2-style Inverted Residual Block with Linear Bottleneck.
    Key features:
    - Expansion phase (narrow -> wide)
    - Depthwise/Factorized convolution (efficient spatial processing)
    - Linear bottleneck (wide -> narrow, NO activation)
    - Squeeze-and-Excitation (optional)
    - Skip connection when stride=1 and in_ch=out_ch
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expansion=6,
        use_factorized=True,
        use_se=True,
        drop_p=0.0
    ):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_channels = in_channels * expansion
        
        layers = []
        
        # 1. Expansion phase (only if expansion > 1)
        if expansion > 1:
            layers.extend([
                nn.Conv3d(in_channels, hidden_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=1, num_channels=hidden_channels),
                nn.GELU()
            ])
        else:
            hidden_channels = in_channels
        
        # 2. Depthwise or Factorized convolution
        if use_factorized:
            layers.append(
                FactorizedConv3D(hidden_channels, hidden_channels, stride=stride)
            )
        else:
            layers.extend([
                nn.Conv3d(
                    hidden_channels, hidden_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_channels,
                    bias=False
                ),
                nn.GroupNorm(num_groups=1, num_channels=hidden_channels),
                nn.GELU()
            ])
        
        # 3. Squeeze-and-Excitation (optional)
        if use_se:
            layers.append(SqueezeExcitation3D(hidden_channels))
        
        # 4. Linear Bottleneck (projection, NO activation)
        layers.extend([
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels)
            # NOTE: No activation here! This is the "linear" bottleneck
        ])
        
        self.conv = nn.Sequential(*layers)
        self.drop_path = StochasticDepth(drop_p, mode="batch") if drop_p > 0 else nn.Identity()
    
    def forward(self, x):
        if self.use_residual:
            return x + self.drop_path(self.conv(x))
        else:
            return self.conv(x)


class EfficientConv3DAE(nn.Module):
    """
    Efficient 3D Convolutional Autoencoder optimized for EEG wavelet data.
    
    Architecture features:
    - Inverted residual blocks (MobileNetV2-style)
    - (2+1)D factorized convolutions for efficiency
    - Squeeze-and-Excitation for better feature representation
    - Linear bottlenecks to preserve information
    - Trilinear upsampling for exact dimension reconstruction
    - Stochastic depth for regularization
    
    Args:
        in_channels: Input channels (25 for wavelet-transformed EEG)
        input_spatial: (H, W, T) spatial dimensions (7, 5, 64)
        latent_dim: Compressed bottleneck size
        hidden_dims: Channel dimensions per encoder stage
        expansion: Expansion ratio for inverted residual blocks
        use_factorized: Use (2+1)D factorization vs depthwise separable
        use_se: Use Squeeze-and-Excitation blocks
        drop_p: Maximum stochastic depth probability
    """
    def __init__(
        self,
        in_channels=25,
        input_spatial=(7, 5, 64),
        latent_dim=128,
        hidden_dims=None,
        expansion=6,
        use_factorized=True,
        use_se=True,
        drop_p=0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.input_spatial = input_spatial
        self.latent_dim = latent_dim
        self.use_factorized = use_factorized
        
        if hidden_dims is None:
            # Efficient channel progression
            hidden_dims = [32, 64, 128]
        
        self.hidden_dims = hidden_dims
        
        # Compute spatial dimensions after each downsampling
        self.encoder_spatial_dims = self._compute_spatial_dims(
            input_spatial, len(hidden_dims) - 1
        )
        
        # ========== ENCODER ==========
        encoder_layers = []
        
        # Stem: Initial lightweight convolution
        encoder_layers.append(
            nn.Sequential(
                nn.Conv3d(
                    in_channels, hidden_dims[0],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                ),
                nn.GroupNorm(num_groups=1, num_channels=hidden_dims[0]),
                nn.GELU()
            )
        )
        
        # Progressive stochastic depth probabilities
        total_blocks = len(hidden_dims) * 2 - 1  # Each stage has 1-2 blocks
        drop_probs = torch.linspace(0, drop_p, total_blocks).tolist()
        drop_idx = 0
        
        # Build encoder stages
        for i in range(len(hidden_dims)):
            in_ch = hidden_dims[i]
            out_ch = hidden_dims[i + 1] if i < len(hidden_dims) - 1 else hidden_dims[i]
            
            # Inverted residual block (stride=1)
            encoder_layers.append(
                InvertedResidual3D(
                    in_ch, in_ch,
                    stride=1,
                    expansion=expansion,
                    use_factorized=use_factorized,
                    use_se=use_se,
                    drop_p=drop_probs[drop_idx]
                )
            )
            drop_idx += 1
            
            # Downsampling block (stride=2, except last stage)
            if i < len(hidden_dims) - 1:
                encoder_layers.append(
                    InvertedResidual3D(
                        in_ch, out_ch,
                        stride=2,
                        expansion=expansion,
                        use_factorized=use_factorized,
                        use_se=use_se,
                        drop_p=drop_probs[drop_idx]
                    )
                )
                drop_idx += 1
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Bottleneck
        final_spatial = self.encoder_spatial_dims[-1]
        self.flat_size = hidden_dims[-1] * final_spatial[0] * final_spatial[1] * final_spatial[2]
        
        # Bottleneck MLPs with GELU activation
        self.bottleneck_encoder = nn.Sequential(
            nn.Linear(self.flat_size, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        self.bottleneck_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, self.flat_size)
        )
        
        # ========== DECODER ==========
        decoder_layers = []
        reversed_hidden = list(reversed(hidden_dims))
        
        # Reverse drop probabilities for decoder
        reversed_drop_probs = list(reversed(drop_probs[:drop_idx]))
        rev_drop_idx = 0
        
        for i in range(len(reversed_hidden)):
            in_ch = reversed_hidden[i]
            out_ch = reversed_hidden[i + 1] if i < len(reversed_hidden) - 1 else reversed_hidden[-1]
            
            # Inverted residual block
            decoder_layers.append(
                InvertedResidual3D(
                    in_ch, in_ch,
                    stride=1,
                    expansion=expansion,
                    use_factorized=use_factorized,
                    use_se=use_se,
                    drop_p=reversed_drop_probs[rev_drop_idx] if rev_drop_idx < len(reversed_drop_probs) else 0
                )
            )
            rev_drop_idx += 1
            
            # Upsampling (except last stage)
            if i < len(reversed_hidden) - 1:
                target_spatial = self.encoder_spatial_dims[-(i + 2)]
                
                decoder_layers.extend([
                    nn.Upsample(size=target_spatial, mode='trilinear', align_corners=False),
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(num_groups=1, num_channels=out_ch),
                    nn.GELU()
                ])
        
        # Final reconstruction head
        decoder_layers.append(
            nn.Conv3d(hidden_dims[0], in_channels, kernel_size=1)
        )
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _compute_spatial_dims(self, input_spatial, num_downsamples):
        """
        Compute spatial dimensions after each downsampling operation.
        Formula for conv with kernel=3, stride=2, padding=1:
        output = (input + 2*1 - 3) // 2 + 1
        """
        dims = [input_spatial]
        h, w, t = input_spatial
        
        for _ in range(num_downsamples):
            h = (h + 2 - 3) // 2 + 1
            w = (w + 2 - 3) // 2 + 1
            t = (t + 2 - 3) // 2 + 1
            dims.append((h, w, t))
        
        return dims
    
    def _initialize_weights(self):
        """Initialize weights following best practices"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x):
        """Encode input to latent representation"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck_encoder(x)
        return x
    
    def decode(self, z):
        """Decode latent representation to reconstruction"""
        x = self.bottleneck_decoder(z)
        final_spatial = self.encoder_spatial_dims[-1]
        x = x.view(-1, self.hidden_dims[-1], *final_spatial)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        """Full autoencoder forward pass"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon
    
    def get_compression_ratio(self):
        """Calculate compression ratio"""
        input_size = self.in_channels * self.input_spatial[0] * self.input_spatial[1] * self.input_spatial[2]
        return input_size / self.latent_dim


if __name__ == "__main__":
    import time
    
    print("=" * 70)
    print("EFFICIENT 3D CONVOLUTIONAL AUTOENCODER")
    print("=" * 70)
    
    # Create model with 30x compression
    model = EfficientConv3DAE(
        in_channels=25,
        input_spatial=(7, 5, 64),
        latent_dim=747,  # 30x compression: 25*7*5*64 / 30 ≈ 747
        hidden_dims=[32, 64, 96],
        expansion=4,  # Lower than MobileNetV2's 6 for efficiency
        use_factorized=True,
        use_se=True,
        drop_p=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Compression ratio: {model.get_compression_ratio():.2f}x")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    bs = 4
    x = torch.randn(bs, 25, 7, 5, 64).to(device)
    
    # Warmup
    with torch.no_grad():
        _ = model(x)
    
    # Benchmark
    n_iters = 50
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            out = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    elapsed = time.time() - start
    
    print(f"\nInput shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Latent dim: {model.latent_dim}")
    print(f"\nAverage inference time: {elapsed / n_iters * 1000:.2f} ms")
    print(f"Throughput: {bs * n_iters / elapsed:.2f} samples/sec")
    
    # Check dimension correctness
    if tuple(out.shape) == tuple(x.shape):
        print("✓ Output dimensions match input!")
    else:
        print("✗ Dimension mismatch!")
