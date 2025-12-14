import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


# -----------------------
# Utility blocks
# -----------------------


class DWConv2dPW(nn.Module):
    """Depthwise followed by pointwise (separable) 2D conv."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, use_bn=True, groups_gn=4):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, stride=s, padding=p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        if use_bn:
            self.norm = nn.GroupNorm(min(groups_gn, out_ch), out_ch)
        else:
            self.norm = nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        return self.act(x)


class SE3D(nn.Module):
    """Lightweight 3D Squeeze-and-Excitation."""
    def __init__(self, channels: int, reduction: int = 8, min_channels: int = 8):
        super().__init__()
        mid = max(min_channels, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, mid, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv3d(mid, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        s = self.pool(x)
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        s = self.gate(s)
        return x * s


class Lite3DBlock(nn.Module):
    """Small 3D conv block with stride on time/space to reduce dims."""
    def __init__(self, cin, cout, stride=(1,2,2), use_gn=True, groups=4):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False)
        if use_gn:
            self.norm = nn.GroupNorm(min(groups, cout), cout)
        else:
            self.norm = nn.BatchNorm3d(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Lite3DUpsample(nn.Module):
    """Upsample then 3D conv to refine, with SE attention."""
    def __init__(self, cin, cout, use_gn=True, groups=4):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, 3, padding=1, bias=False)
        if use_gn:
            self.norm = nn.GroupNorm(min(groups, cout), cout)
        else:
            self.norm = nn.BatchNorm3d(cout)
        self.se = SE3D(cout, reduction=8)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x, size: Tuple[int,int,int]):
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        x = self.conv(x)
        x = self.norm(x)
        x = self.se(x)
        return self.act(x)


class UpPixelShuffle2D(nn.Module):
    """
    Sub-pixel upsampling along width (time-flattened) only.
    Input:  (B, C, H, W)
    Output: (B, C_out, H, W*2)
    """
    def __init__(self, cin: int, cout: int, use_bn: bool = True, groups_gn: int = 4):
        super().__init__()
        # First expand channels for sub-pixel rearrangement
        self.expand = nn.Conv2d(cin, cin * 2, kernel_size=1, bias=False)
        # We will reshape and call PixelShuffle with upscale_factor=2 treating width as height
        self.refine = DWConv2dPW(cin, cout, k=3, s=1, p=1, use_bn=use_bn, groups_gn=groups_gn)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = self.expand(x)          # (B, 2C, H, W)
        # Treat (H, W) -> (1, H*W) so that PixelShuffle(2) doubles "width" = H*W
        x = x.view(B, 2*C, 1, H*W)  # (B, 2C, 1, H*W)
        x = F.pixel_shuffle(x, upscale_factor=2)  # (B, C, 2, H*W)
        # Now interpret the 2 as width-upscale, restore H, W*2
        x = x.view(B, C, H, W*2)    # (B, C, H, 2W)
        x = self.refine(x)
        return x


# -----------------------
# Config
# -----------------------


@dataclass
class VQAELightConfig:
    use_quantizer: bool = False
    use_cwt: bool = True

    # CWT parameters
    cwt_frequencies: tuple = None
    chunk_samples: int = 160
    normalize_outputs: bool = True
    learnable_norm: bool = True 

    # Data shape parameters
    num_input_channels: int = 2   # real + imag
    num_freq_bands: int = 25
    spatial_rows: int = 7
    spatial_cols: int = 5
    time_samples: int = 160
    orig_channels: int = 32

    # Encoder-Decoder channels
    encoder_2d_channels: list = None  # small
    encoder_3d_channels: list = None  # tiny
    embedding_dim: int = 48           # small bottleneck

    # VQ
    codebook_size: int = 128
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    epsilon: float = 1e-5

    # Norms
    use_group_norm: bool = True
    num_groups: int = 4

    def __post_init__(self):
        if self.encoder_2d_channels is None:
            self.encoder_2d_channels = [32, 48]
        if self.encoder_3d_channels is None:
            self.encoder_3d_channels = [64, 96]
        if self.cwt_frequencies is None:
            frequencies = np.logspace(np.log10(0.5), np.log10(79.9), 25)
            self.cwt_frequencies = tuple(frequencies)


# -----------------------
# Vector Quantizer (EMA)
# -----------------------


class VectorQuantizerLight(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.register_buffer('embeddings', torch.randn(num_embeddings, embedding_dim))
        nn.init.xavier_uniform_(self.embeddings)
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_embed_avg', self.embeddings.clone())

    def forward(self, inputs):
        flat = inputs.view(-1, self.embedding_dim)
        f_norm = F.normalize(flat, p=2, dim=1)
        e_norm = F.normalize(self.embeddings, p=2, dim=1)
        distances = 1.0 - torch.matmul(f_norm, e_norm.t())
        idx = torch.argmin(distances, dim=1)
        quant = F.embedding(idx, self.embeddings)
        if self.training:
            self._ema_update(flat, idx)
        e_latent = F.mse_loss(quant.detach(), flat)
        quant = flat + (quant - flat).detach()
        q_latent = F.mse_loss(quant, flat.detach())
        vq_loss = q_latent + self.commitment_cost * e_latent
        quant = quant.view_as(inputs)
        avg_probs = torch.bincount(idx, minlength=self.num_embeddings).float() / len(idx)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        usage = (avg_probs > 0).float().mean()
        return quant, idx, {'vq_loss': vq_loss, 'perplexity': perplexity, 'codebook_usage': usage}

    def _ema_update(self, flat, idx):
        enc_onehot = F.one_hot(idx, num_classes=self.num_embeddings).float()
        cluster = torch.sum(enc_onehot, dim=0)
        self.ema_cluster_size.data.mul_(self.decay).add_(cluster, alpha=1 - self.decay)
        n = torch.sum(self.ema_cluster_size)
        self.ema_cluster_size.data.add_(self.epsilon).div_(n + self.num_embeddings * self.epsilon).mul_(n)
        embed_sum = torch.matmul(enc_onehot.t(), flat)
        self.ema_embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
        self.embeddings.data.copy_(self.ema_embed_avg / self.ema_cluster_size.unsqueeze(1))


# -----------------------
# Encoder (2D → 3D) and Decoder (3D → 2D)
# -----------------------


class Encoder2D(nn.Module):
    """2D downsampling on (grid cell, freq, time)."""
    def __init__(self, config: VQAELightConfig):
        super().__init__()
        self.config = config
        use_gn = config.use_group_norm
        g = config.num_groups
        ch_in = config.num_input_channels * config.num_freq_bands  # 2*F

        layers = []
        c = ch_in
        for i, outc in enumerate(config.encoder_2d_channels):
            layers.append(DWConv2dPW(c, outc, k=3, s=(1,2), p=1, use_bn=use_gn, groups_gn=g))
            c = outc
        self.net = nn.Sequential(*layers)

        self.h_out = config.spatial_rows
        self.w_out = config.spatial_cols
        self.t_out = config.time_samples
        for _ in config.encoder_2d_channels:
            self.t_out = (self.t_out + 1) // 2
        self.c_out = c

    def forward(self, x_cwt):
        # x_cwt: (B, 2, F, 7, 5, T)
        B, two, F, H, W, T = x_cwt.shape
        x = x_cwt.reshape(B, two*F, H, W*T)
        return self.net(x)  # (B, C2d, 7, 5*T')


class Encoder3D(nn.Module):
    """Tiny 3D encoder mixing (H, W, T') with minimal channels."""
    def __init__(self, config: VQAELightConfig, c2d: int, t_out: int):
        super().__init__()
        self.config = config
        use_gn = config.use_group_norm
        g = config.num_groups
        self.in_ch = c2d
        self.t_out = t_out

        self.enc3d = nn.Sequential(
            Lite3DBlock(self.in_ch, config.encoder_3d_channels[0], stride=(1,2,2), use_gn=use_gn, groups=g),
            Lite3DBlock(config.encoder_3d_channels[0], config.encoder_3d_channels[1], stride=(1,2,2), use_gn=use_gn, groups=g),
        )
        self.h_out = (config.spatial_rows + 1) // 2
        self.h_out = (self.h_out + 1) // 2
        self.w_out = (config.spatial_cols + 1) // 2
        self.w_out = (self.w_out + 1) // 2
        self.t3_out = t_out
        self.c3_out = config.encoder_3d_channels[-1]

        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
            nn.Linear(self.c3_out, config.embedding_dim, bias=False),
            nn.LayerNorm(config.embedding_dim),
        )

    def forward(self, x2d):
        # x2d: (B, C2d, H=7, W=5*T')
        B, C2d, H, WT = x2d.shape
        W = self.config.spatial_cols
        Tprime = WT // W
        assert WT % W == 0, "Width must be divisible by spatial_cols"
        x = x2d.view(B, C2d, H, W, Tprime)
        h = self.enc3d(x)
        z = self.proj(h)
        return h, z, (H, W, Tprime)


class Decoder3D(nn.Module):
    """Enhanced 3D decoder with SE attention."""
    def __init__(self, config: VQAELightConfig, enc3_shape, c3_out: int):
        super().__init__()
        self.config = config
        use_gn = config.use_group_norm
        g = config.num_groups
        H, W, Tprime = enc3_shape
        self.HWT = (H, W, Tprime)
        self.c3 = c3_out

        self.from_z = nn.Sequential(
            nn.Linear(config.embedding_dim, self.c3, bias=False),
            nn.SiLU(inplace=True),
        )

        self.up1 = Lite3DUpsample(self.c3, config.encoder_3d_channels[0], use_gn=use_gn, groups=g)
        self.up2 = Lite3DUpsample(config.encoder_3d_channels[0], config.encoder_3d_channels[0], use_gn=use_gn, groups=g)

        self.out2d = nn.Conv3d(config.encoder_3d_channels[0], self.c3, 1, bias=False)
        self.post_se = SE3D(self.c3, reduction=8)

    def forward(self, z):
        B = z.shape[0]
        H, W, Tprime = self.HWT
        h0 = self.from_z(z).view(B, self.c3, 1, 1, 1)
        h = self.up1(h0, size=((H+1)//2, (W+1)//2, Tprime))
        h = self.up2(h,  size=(H, W, Tprime))
        h = self.out2d(h)
        h = self.post_se(h)
        x2d = h.view(B, self.c3, H, W*Tprime)
        return x2d


class Decoder2D(nn.Module):
    """2D upsampling back to CWT layout (B, 2, F, 7, 5, T) with PixelShuffle-style upsampling."""
    def __init__(self, config: VQAELightConfig, c2d_out: int):
        super().__init__()
        self.config = config
        use_gn = config.use_group_norm
        g = config.num_groups
        C_in = c2d_out
        layers = []

        chans = list(reversed(config.encoder_2d_channels))
        for i, outc in enumerate(chans):
            layers.append(UpPixelShuffle2D(C_in, outc, use_bn=use_gn, groups_gn=g))
            C_in = outc

        layers.append(nn.Conv2d(C_in, config.num_input_channels * config.num_freq_bands, 1, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x2d):
        x = self.net(x2d)  # (B, 2F, 7, 5*T)
        B, CF, H, WT = x.shape
        two = self.config.num_input_channels
        F = self.config.num_freq_bands
        W = self.config.spatial_cols
        T = WT // W
        x = x.view(B, two, F, H, W, T)
        return x


# -----------------------
# VQAE (CWT-space)
# -----------------------


class VQAELight(nn.Module):
    def __init__(self, config: VQAELightConfig | Dict):
        super().__init__()
        if isinstance(config, dict):
            config = VQAELightConfig(**config)
        self.config = config
        self.use_cwt = config.use_cwt
        self.chunk_samples = config.chunk_samples

        if self.use_cwt:
            from packages.models.wavelet_head import CWTHead
            self.cwt_head = CWTHead(
                frequencies=config.cwt_frequencies,
                fs=160,
                num_channels=config.orig_channels,
                n_cycles=5.0,
                trainable=False,
                chunk_samples=config.chunk_samples,
                normalize_outputs=config.normalize_outputs,
                learnable_norm=config.learnable_norm
            )

        self.enc2d = Encoder2D(config)
        self.enc3d = Encoder3D(config, c2d=self.enc2d.c_out, t_out=self.enc2d.t_out)

        self.vq = VectorQuantizerLight(
            config.codebook_size, config.embedding_dim,
            config.commitment_cost, config.ema_decay, config.epsilon
        )

        self.dec3d = Decoder3D(
            config,
            enc3_shape=(self.config.spatial_rows, self.config.spatial_cols, self.enc2d.t_out),
            c3_out=self.enc3d.c3_out
        )
        self.dec2d = Decoder2D(config, c2d_out=self.enc3d.c3_out)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if self.use_cwt and hasattr(self, "cwt_head") and m is getattr(self.cwt_head, "conv", None):
            return
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                          nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d)):
            if getattr(m, "weight", None) is not None:
                nn.init.constant_(m.weight, 1)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)

    def encode(self, x_cwt):
        x2d = self.enc2d(x_cwt)
        h3, z, shape = self.enc3d(x2d)
        return h3, z, (x2d.shape, shape)

    def decode(self, z):
        x2d = self.dec3d(z)
        xhat = self.dec2d(x2d)
        return x2d, xhat

    def forward(self, x):
        if not self.use_cwt:
            raise ValueError("Assumes use_cwt=True")
        x_cwt = self.cwt_head(x)
        h3, z, _ = self.encode(x_cwt)

        if self.config.use_quantizer:
            z_q, indices, vq_losses = self.vq(z)
        else:
            z_q = z
            indices = torch.zeros(z.shape[0], device=z.device, dtype=torch.long)
            vq_losses = {
                "vq_loss": torch.tensor(0., device=z.device),
                "perplexity": torch.tensor(0., device=z.device),
                "codebook_usage": torch.tensor(1., device=z.device),
            }

        _, recon_cwt = self.decode(z_q)
        return {
            "reconstruction": recon_cwt,
            "target": x_cwt,
            "embeddings": z,
            "quantized": z_q,
            "indices": indices,
            **vq_losses
        }


if __name__ == "__main__":
    config = VQAELightConfig()
    model = VQAELight(config)
    x = torch.randn(4, 32, 640)
    outputs = model(x)
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")
        else:
            print(f"{k}: {v}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(model)
