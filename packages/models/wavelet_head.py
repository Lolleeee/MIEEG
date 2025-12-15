import torch
import torch.nn as nn
import numpy as np

class CWTHead(nn.Module):
    def __init__(
        self,
        frequencies: list | np.ndarray,
        fs: int,
        num_channels: int = 32,
        n_cycles: float = 6.0,
        trainable: bool = False,
        chunk_samples: int = None,
        normalize_outputs: bool = True,
        learnable_norm: bool = True
    ):
        """
        CWT Head returning Real/Imag on a 7x5 grid.

        Returns:
            (B, 2, F, 7, 5, T) where 2 = (real, imag)
        """
        super().__init__()
        self.num_channels = num_channels
        self.frequencies = np.array(frequencies)
        self.num_freqs = len(frequencies)
        self.chunk_size_samples = chunk_samples

        self.normalize_outputs = normalize_outputs
        self.learnable_norm = learnable_norm

        # 1) Setup filters (same as your version)
        f_min = np.min(frequencies)
        if f_min < 4.0:
            cycles_at_min = max(2.0, n_cycles * (f_min / 4.0))
        else:
            cycles_at_min = n_cycles

        sigma_max = cycles_at_min / (2 * np.pi * f_min)
        kernel_size = int(8 * sigma_max * fs)
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2

        # 2) Morlet conv (same)
        self.conv = nn.Conv1d(
            in_channels=num_channels,
            out_channels=num_channels * self.num_freqs * 2,  # Real + Imag
            kernel_size=kernel_size,
            padding=padding,
            padding_mode='reflect',
            groups=num_channels,
            bias=False
        )

        weights = self._create_morlet_weights(frequencies, fs, kernel_size, n_cycles_max=n_cycles)
        self.conv.weight.data = weights
        self.conv.weight.requires_grad = trainable

        # 3) Spatial mapping (same)
        mapping_matrix = np.array([
            [-1,  0, -1,  1, -1],
            [ 2,  3,  4,  5,  6],
            [ 7,  8, 13,  9, 10],
            [11, 12, 18, 14, 15],
            [16, 17, 19, 20, 21],
            [22, 23, 24, 25, 26],
            [27, 28, 29, 30, 31]
        ])

        rows, cols = [], []
        for ch in range(32):
            coords = np.where(mapping_matrix == ch)
            rows.append(coords[0][0])
            cols.append(coords[1][0])

        self.register_buffer('rows', torch.tensor(rows, dtype=torch.long))
        self.register_buffer('cols', torch.tensor(cols, dtype=torch.long))
        self.grid_h, self.grid_w = 7, 5

        # 4) Per-frequency normalization (now for real/imag)
        if self.normalize_outputs:
            shift0 = torch.zeros(self.num_freqs, dtype=torch.float32)
            scale0 = torch.ones(self.num_freqs, dtype=torch.float32)

            if learnable_norm:
                self.ri_shift = nn.Parameter(shift0)  # (F,)
                self.ri_scale = nn.Parameter(scale0)  # (F,)
            else:
                self.register_buffer('ri_shift', shift0)
                self.register_buffer('ri_scale', scale0)

    def forward_pre_chunk(self, x):
        """
        Real/Imag CWT transform (no mag/phase).

        Args:
            x: (B, 32, T)

        Returns:
            canvas: (B, 2, F, 7, 5, T) where 2=(real, imag)
        """
        B, C, T = x.shape

        cwt_raw = self.conv(x)  # (B, 32*F*2, T)
        cwt_reshaped = cwt_raw.view(B, C, self.num_freqs, 2, T)

        real = cwt_reshaped[..., 0, :]  # (B, 32, F, T)
        imag = cwt_reshaped[..., 1, :]  # (B, 32, F, T)

        # Per-frequency affine normalization (applied to both real and imag)
        if self.normalize_outputs:
            real = (real + self.ri_shift[None, None, :, None]) * self.ri_scale[None, None, :, None]
            imag = (imag + self.ri_shift[None, None, :, None]) * self.ri_scale[None, None, :, None]

        # Stack real + imag
        features = torch.stack([real, imag], dim=-1)      # (B, 32, F, T, 2)
        features = features.permute(0, 4, 2, 1, 3)        # (B, 2, F, 32, T)

        # Map 32 channels to 7×5 spatial grid
        canvas = torch.zeros(
            B, 2, self.num_freqs, self.grid_h, self.grid_w, T,
            device=x.device, dtype=x.dtype
        )
        canvas[:, :, :, self.rows, self.cols, :] = features

        # Fill padding positions (same as your version)
        canvas[:, :, :, 0, 0, :] = canvas[:, :, :, 1, 0, :]
        canvas[:, :, :, 0, 2, :] = canvas[:, :, :, 1, 2, :]
        canvas[:, :, :, 0, 4, :] = canvas[:, :, :, 1, 4, :]

        return canvas

    def forward(self, x):
        B, C, Total_Time = x.shape
        full_cwt = self.forward_pre_chunk(x)

        if self.chunk_size_samples is None:
            return full_cwt

        if Total_Time % self.chunk_size_samples != 0:
            raise ValueError(
                f"Total time {Total_Time} not divisible by chunk size {self.chunk_size_samples}"
            )

        self.num_chunks = Total_Time // self.chunk_size_samples

        cwt_permuted = full_cwt.permute(0, 5, 1, 2, 3, 4)
        chunks = cwt_permuted.view(
            B, self.num_chunks, self.chunk_size_samples,
            2, self.num_freqs, self.grid_h, self.grid_w
        )

        chunks_merged = chunks.reshape(
            B * self.num_chunks, self.chunk_size_samples,
            2, self.num_freqs, self.grid_h, self.grid_w
        )

        output = chunks_merged.permute(0, 2, 3, 4, 5, 1)
        return output

    def _create_morlet_weights(self, freqs, fs, K, n_cycles_max=6.0):
        weights = torch.zeros(self.num_channels * self.num_freqs * 2, 1, K)
        t = np.linspace(-K/2/fs, K/2/fs, K)

        for i, f in enumerate(freqs):
            if f < 1.0:
                cycles = 3.0 + (f / 1.0) * (4.0 - 3.0)
            elif f < 4.0:
                cycles = 4.0 + ((f - 1.0) / 3.0) * (n_cycles_max - 4.0)
            else:
                cycles = n_cycles_max

            sigma = cycles / (2 * np.pi * f)
            sine = np.exp(2j * np.pi * f * t)
            gauss = np.exp(-t**2 / (2 * sigma**2))
            wavelet = sine * gauss
            wavelet /= np.linalg.norm(wavelet)

            for c in range(self.num_channels):
                idx_real = (c * self.num_freqs * 2) + (i * 2)
                idx_imag = idx_real + 1
                weights[idx_real, 0, :] = torch.from_numpy(np.real(wavelet))
                weights[idx_imag, 0, :] = torch.from_numpy(np.imag(wavelet))

        return weights

    def _unchunk(self, x):
        Bn, C, chunk_T = x.shape
        if self.chunk_size_samples is None:
            return x
        x = x.reshape(-1, C, self.num_chunks * chunk_T)
        return x

class InverseCWTHead(nn.Module):
    """
    Improved inverse head:
      - Positive per-frequency gain via exp(log_gain)
      - Optional mean-normalization of gains
      - Grouped ConvTranspose1d initialized from analysis weights
      - Optional depthwise+pointwise refinement in time domain
    """
    def __init__(
        self,
        cwt_head,
        train_synthesis: bool = True,
        refine: bool = True,
        refine_kernel: int = 65,          # bigger default
        normalize_gain_mean: bool = True, # keep mean gain ~1
        gain_min: float = 1e-4,           # avoid vanishing gains
    ):
        super().__init__()
        self.cwt_head = cwt_head
        self.F = cwt_head.num_freqs
        self.C = cwt_head.num_channels
        self.K = cwt_head.conv.kernel_size[0]
        self.padding = cwt_head.conv.padding[0]

        self.normalize_gain_mean = normalize_gain_mean
        self.gain_min = float(gain_min)

        # log-gains initialized to 0 => gain = 1
        self.freq_gain_log = nn.Parameter(torch.zeros(self.F))

        # Synthesis filterbank: in_ch = C*(2F), out_ch = C, groups=C
        self.synth = nn.ConvTranspose1d(
            in_channels=self.C * (2 * self.F),
            out_channels=self.C,
            kernel_size=self.K,
            stride=1,
            padding=self.padding,
            groups=self.C,
            bias=False
        )

        with torch.no_grad():
            self.synth.weight.copy_(self.cwt_head.conv.weight)

        self.synth.weight.requires_grad = bool(train_synthesis)

        self.refine = None
        if refine:
            k = int(refine_kernel)
            if k % 2 == 0:
                k += 1
            self.refine = nn.Sequential(
                nn.Conv1d(self.C, self.C, kernel_size=k, padding=k // 2, groups=self.C, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv1d(self.C, self.C, kernel_size=1, bias=True),
            )

    def _freq_gain(self):
        # positive gains
        g = torch.exp(self.freq_gain_log).clamp_min(self.gain_min)  # (F,)
        if self.normalize_gain_mean:
            g = g / (g.mean() + 1e-12)
        return g

    def forward(self, cwt_grid):
        """
        cwt_grid: (B*,2,F,7,5,T) -> (B*,32,T)
        """
        B, two, Freqs, H, W, T = cwt_grid.shape
        assert two == 2 and Freqs == self.F

        # (B,2,F,32,T)
        feat = cwt_grid[:, :, :, self.cwt_head.rows, self.cwt_head.cols, :]

        # invert normalization: y=(x+shift)*scale -> x=y/scale - shift
        if getattr(self.cwt_head, "normalize_outputs", False):
            scale = self.cwt_head.ri_scale[None, None, :, None, None]
            shift = self.cwt_head.ri_shift[None, None, :, None, None]
            feat = feat / (scale + 1e-12)
            feat = feat - shift

        # per-frequency synthesis weighting
        g = self._freq_gain()[None, None, :, None, None]  # (1,1,F,1,1)
        feat = feat * g

        # pack to (B, C*(2F), T)
        feat = feat.permute(0, 3, 1, 2, 4).contiguous()  # (B,32,2,F,T)
        z = feat.view(B, self.C * (2 * self.F), T)

        # synthesize
        x_hat = self.synth(z)

        # refine
        if self.refine is not None:
            x_hat = self.refine(x_hat)

        return x_hat

if __name__ == "__main__":

    InverseCWTHead_test = InverseCWTHead(
        cwt_head=CWTHead(
            frequencies=np.linspace(1, 50, 30),
            fs=250,
            num_channels=32,
            n_cycles=6.0,
            trainable=False,
            chunk_samples=None,
            normalize_outputs=True,
            learnable_norm=True
        ),
        train_synthesis=True,
        refine=True,
        refine_kernel=9
    )
    x = torch.randn(2, 32, 1000)
    cwt_out = InverseCWTHead_test.cwt_head(x)
    x_rec = InverseCWTHead_test(cwt_out)
    print(x_rec.shape)  # Expected: (2, 32, 1000
    