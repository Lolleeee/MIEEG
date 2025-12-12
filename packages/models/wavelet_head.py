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
        use_log_compression: bool = True,   # kept for API compatibility (unused in real/imag mode)
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

        # kept but not used in real/imag output
        self.use_log_compression = use_log_compression

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



import torch
import torch.nn as nn

class WaveletSynthesisHead(nn.Module):
    """
    Synthesis: (real, imag, freq, grid_h, grid_w, time) -> (channels, time)

    Input:  coeffs (B_chunk, 2, F, 7, 5, T)
    Output: x_hat  (B_chunk, 32, T)
    """
    def __init__(self, cwt_head: "CWTHead", learn_freq_gains: bool = True):
        super().__init__()
        self.cwt_head = cwt_head
        self.C = cwt_head.num_channels
        self.F = cwt_head.num_freqs

        # Optional scale/frequency weighting (cheap, usually helps)
        if learn_freq_gains:
            self.freq_gain = nn.Parameter(torch.ones(self.F, dtype=torch.float32))
        else:
            self.register_buffer("freq_gain", torch.ones(self.F, dtype=torch.float32))

        # Learnable synthesis filterbank (grouped by EEG channel)
        # Not a true inverse in general, but a strong learnable decoder. [web:84]
        self.synth = nn.ConvTranspose1d(
            in_channels=self.C * (2 * self.F),
            out_channels=self.C,
            kernel_size=cwt_head.conv.kernel_size[0],
            padding=cwt_head.conv.padding[0],
            groups=self.C,
            bias=False,
        )

        # Good init: start from analysis kernels (adjoint-like init)
        self.synth.weight.data = cwt_head.conv.weight.data.clone()

    def forward(self, coeffs):
        """
        coeffs: (B_chunk, 2, F, 7, 5, T)
        """
        Bc, two, F, H, W, T = coeffs.shape
        assert two == 2 and F == self.F

        # Grid -> 32 channels: (B_chunk, 2, F, 32, T)
        feat = coeffs[:, :, :, self.cwt_head.rows, self.cwt_head.cols, :]

        # Undo CWTHead's real/imag normalization if enabled (so synth sees "raw-ish" coeffs)
        # Your CWTHead does: (x + shift) * scale, so invert: x = x/scale - shift
        if getattr(self.cwt_head, "normalize_outputs", False):
            scale = self.cwt_head.ri_scale[None, None, :, None, None]  # (1,1,F,1,1)
            shift = self.cwt_head.ri_shift[None, None, :, None, None]  # (1,1,F,1,1)
            feat = feat / (scale + 1e-12)
            feat = feat - shift

        # Apply frequency gains to both real and imag
        feat = feat * self.freq_gain[None, None, :, None, None]

        # Pack into grouped ConvTranspose1d format:
        # (B_chunk, 2, F, 32, T) -> (B_chunk, 32, 2*F, T) -> (B_chunk, 32*(2F), T)
        feat = feat.permute(0, 3, 1, 2, 4).contiguous()          # (B_chunk, 32, 2, F, T)
        feat = feat.view(Bc, self.C, 2 * self.F, T)              # (B_chunk, 32, 2F, T)
        z = feat.reshape(Bc, self.C * (2 * self.F), T)           # (B_chunk, 32*(2F), T)

        x_hat = self.synth(z)                                    # (B_chunk, 32, T)
        return x_hat

if __name__ == "__main__":
    # Print model parameters
    synth = WaveletSynthesisHead(
        cwt_head=CWTHead(
            frequencies=[4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
            fs=250,
            num_channels=32,
            n_cycles=6.0,
            trainable=False,
            chunk_samples=250,
            use_log_compression=True,
            normalize_outputs=True,
            learnable_norm=True
        ),
        learn_freq_gains=True
    )
    total_params = sum(p.numel() for p in synth.parameters() if p.requires_grad)
    print(f"Total trainable parameters in WaveletSynthesisHead: {total_params}")