import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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
        use_log_compression: bool = True,
        normalize_outputs: bool = True,
        learnable_norm: bool = True
    ):
        """
        CWT Head with per-frequency normalization for balanced power/phase gradients.
        
        Args:
            frequencies: Array of frequencies to analyze (e.g., 0.5 to 80 Hz)
            fs: Sampling rate (Hz)
            num_channels: Number of EEG channels (default: 32)
            n_cycles: Maximum number of wavelet cycles (default: 6.0)
            trainable: Whether Morlet kernels are learnable (default: False)
            chunk_samples: If set, chunks time dimension (e.g., 160 for 640→4×160)
            use_log_compression: Apply log1p to power (default: True)
            normalize_outputs: Enable per-frequency normalization (default: True)
            learnable_norm: Make normalization parameters learnable (default: True)
        """
        super().__init__()
        self.num_channels = num_channels
        self.frequencies = np.array(frequencies)
        self.num_freqs = len(frequencies)
        self.chunk_size_samples = chunk_samples
        self.use_log_compression = use_log_compression
        self.normalize_outputs = normalize_outputs
        self.learnable_norm = learnable_norm
        
        # 1. Setup Filters (Dynamic Cycles for Low Frequencies)
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
        
        # 2. Convolution Layer (Morlet Wavelets)
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
        
        # 3. Spatial Mapping (32 channels → 7×5 grid)
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
        
        # 4. Per-Frequency Normalization Parameters
        if self.normalize_outputs:
            freqs_tensor = torch.tensor(self.frequencies, dtype=torch.float32)
            
            # Initialize based on 1/f power distribution
            # Higher frequencies have lower power → more negative log-power
            expected_log_power = -torch.log(freqs_tensor / freqs_tensor.max())
            expected_std = torch.ones(self.num_freqs) * 1.5  # Empirical estimate
            
            if learnable_norm:
                # Learnable parameters (optimized via backprop)
                self.power_shift = nn.Parameter(-expected_log_power)  # (25,)
                self.power_scale = nn.Parameter(1.0 / expected_std)   # (25,)
                self.phase_scale = nn.Parameter(torch.tensor(1.0 / np.pi))  # Scalar
            else:
                # Fixed buffers (hand-tuned)
                self.register_buffer('power_shift', -expected_log_power)
                self.register_buffer('power_scale', 1.0 / expected_std)
                self.register_buffer('phase_scale', torch.tensor(1.0 / np.pi))


    def forward_pre_chunk(self, x):
        """
        CWT transform with per-frequency normalization.
        
        Args:
            x: (B, 32, T) - Raw EEG (per-channel z-scored)
            
        Returns:
            (B, 2, 25, 7, 5, T) - Power/Phase on spatial grid
        """
        B, C, T = x.shape
        
        # Apply Morlet wavelets
        cwt_raw = self.conv(x)  # (B, 32*25*2, T)
        cwt_reshaped = cwt_raw.view(B, C, self.num_freqs, 2, T)
        
        real = cwt_reshaped[..., 0, :]  # (B, 32, 25, T)
        imag = cwt_reshaped[..., 1, :]  # (B, 32, 25, T)
        
        # === POWER CHANNEL ===
        mag = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-8)
        
        # Log-compression (handles 1/f distribution)
        if self.use_log_compression:
            mag = torch.log1p(mag)  # log(1 + x) for numerical stability
        
        # Per-frequency normalization
        if self.normalize_outputs:
            # Broadcast shapes: (B, 32, 25, T) + (25,) → (B, 32, 25, T)
            mag = mag + self.power_shift[None, None, :, None]
            mag = mag * self.power_scale[None, None, :, None]
        
        # === PHASE CHANNEL ===
        phase = torch.atan2(imag, real)  # (B, 32, 25, T) in [-π, π]
        
        # Scale to [-1, 1] for balanced gradients with power
        if self.normalize_outputs:
            phase = phase * self.phase_scale
        
        # Stack power + phase
        features = torch.stack([mag, phase], dim=-1)  # (B, 32, 25, T, 2)
        features = features.permute(0, 4, 2, 1, 3)     # (B, 2, 25, 32, T)
        
        # Map 32 channels to 7×5 spatial grid
        canvas = torch.zeros(
            B, 2, self.num_freqs, self.grid_h, self.grid_w, T,
            device=x.device, dtype=x.dtype
        )
        canvas[:, :, :, self.rows, self.cols, :] = features

        # === FILL PADDING POSITIONS ===
        # Position [0,0] → nearest valid channel at [1,0] (channel 2)
        canvas[:, :, :, 0, 0, :] = canvas[:, :, :, 1, 0, :]
        
        # Position [0,2] → nearest valid channel at [1,2] (channel 4)
        canvas[:, :, :, 0, 2, :] = canvas[:, :, :, 1, 2, :]
        
        # Position [0,4] → nearest valid channel at [1,4] (channel 6)
        canvas[:, :, :, 0, 4, :] = canvas[:, :, :, 1, 4, :]
        
        return canvas  # (B, 2, 25, 7, 5, T)


    def forward(self, x):
        """
        Main forward pass with optional chunking.
        
        Args:
            x: (B, 32, Total_Time) - Raw EEG, e.g., (B, 32, 640)
            
        Returns:
            If chunk_samples=None: (B, 2, 25, 7, 5, Total_Time)
            If chunk_samples=160:  (B*4, 2, 25, 7, 5, 160) for 640 samples
        """
        B, C, Total_Time = x.shape
        full_cwt = self.forward_pre_chunk(x)
        
        if self.chunk_size_samples is None:
            return full_cwt
            
        if Total_Time % self.chunk_size_samples != 0:
            raise ValueError(
                f"Total time {Total_Time} not divisible by chunk size {self.chunk_size_samples}"
            )
            
        self.num_chunks = Total_Time // self.chunk_size_samples
        
        # Reshape: (B, 2, F, H, W, T) → (B, T, 2, F, H, W)
        cwt_permuted = full_cwt.permute(0, 5, 1, 2, 3, 4)
        
        # Split time into chunks: (B, num_chunks, chunk_T, 2, F, H, W)
        chunks = cwt_permuted.view(
            B, self.num_chunks, self.chunk_size_samples, 
            2, self.num_freqs, self.grid_h, self.grid_w
        )
        
        # Merge batch and chunks: (B*num_chunks, chunk_T, 2, F, H, W)
        chunks_merged = chunks.reshape(
            B * self.num_chunks, self.chunk_size_samples, 
            2, self.num_freqs, self.grid_h, self.grid_w
        )
        
        # Final permute: (B*num_chunks, 2, F, H, W, chunk_T)
        output = chunks_merged.permute(0, 2, 3, 4, 5, 1)
        
        return output


    def _create_morlet_weights(self, freqs, fs, K, n_cycles_max=6.0):
        """Creates Morlet wavelets with dynamic cycle scaling."""
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
        """Reverses chunking operation."""
        Bn, C, chunk_T = x.shape
        if self.chunk_size_samples is None:
            return x
        x = x.view(-1, C, self.num_chunks * chunk_T)
        return x

    
class InverseCWTHead(nn.Module):
    def __init__(self, encoder_head):
        super().__init__()
        self.frequencies = encoder_head.frequencies
        self.num_channels = encoder_head.num_channels
        self.num_freqs = encoder_head.num_freqs
        self.use_log_compression = getattr(encoder_head, 'use_log_compression', False)
        
        in_ch = encoder_head.conv.out_channels
        out_ch = encoder_head.num_channels
        
        # Standard Inverse Conv
        self.inv_conv = nn.ConvTranspose1d(
            in_channels=in_ch, out_channels=out_ch,
            kernel_size=encoder_head.conv.kernel_size,
            padding=encoder_head.conv.padding,
            groups=encoder_head.num_channels, 
            bias=False
        )
        
        self.inv_conv.weight.data = encoder_head.conv.weight.data
        self.inv_conv.weight.requires_grad = False 
        self.scale = nn.Parameter(torch.ones(1, out_ch, 1))

    def forward(self, x):
        # x: (B, 32*F*2, T)
        B, C_total, T = x.shape
        
        # 1. Reshape [B, 32, F, 2, T]
        x_reshaped = x.view(B, self.num_channels, self.num_freqs, 2, T)
        
        mag_raw = x_reshaped[..., 0, :]
        phase = x_reshaped[..., 1, :]
        
        # 2. FIX: Enforce Positive Magnitude before inversion
        # If the decoder outputs -5.0, expm1(-5) is -0.99 (Invalid Magnitude)
        # Softplus ensures we are always in the positive domain before expm1
        if self.use_log_compression:
             # Softplus makes negative logits small positive numbers, preserving gradient
            mag_positive = F.softplus(mag_raw)
            mag = torch.expm1(mag_positive)
        else:
            mag = F.softplus(mag_raw) # Simple positivity constraint

        # 3. Polar -> Rectangular
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        
        # 4. Interleave for Morlet Weight compatibility
        rect_features = torch.stack([real, imag], dim=3)
        rect_flat = rect_features.view(B, C_total, T)
        
        return self.inv_conv(rect_flat) * self.scale
