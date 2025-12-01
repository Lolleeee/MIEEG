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
        chunk_samples: int = None 
    ):
        super().__init__()
        self.num_channels = num_channels
        self.frequencies = frequencies
        self.num_freqs = len(frequencies)
        self.chunk_size_samples = chunk_samples 
        
        # 1. Setup Filters (Corrected for Dynamic Cycles)
        # We need to calculate sigma_max using the same logic as create_weights
        # At lowest freq (e.g. 1Hz), we might scale down cycles to 2.0
        f_min = np.min(frequencies)
        if f_min < 4.0:
            cycles_at_min = max(2.0, n_cycles * (f_min / 4.0))
        else:
            cycles_at_min = n_cycles
            
        sigma_max = cycles_at_min / (2 * np.pi * f_min)
        
        # Kernel size: 8 sigmas covers >99.9% of energy
        kernel_size = int(8 * sigma_max * fs) 
        if kernel_size % 2 == 0: kernel_size += 1 
        
        padding = kernel_size // 2 
        
        # 2. Convolution Layer
        self.conv = nn.Conv1d(
            in_channels=num_channels,
            out_channels=num_channels * self.num_freqs * 2, 
            kernel_size=kernel_size,
            padding=padding,
            padding_mode='reflect', 
            groups=num_channels,    
            bias=False
        )
        
        # Pass n_cycles_max to the weight creator
        weights = self._create_morlet_weights(frequencies, fs, kernel_size, n_cycles_max=n_cycles)
        self.conv.weight.data = weights
        self.conv.weight.requires_grad = trainable
        
        # 3. Spatial Mapping Vectors
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
            
        self.register_buffer('rows', torch.tensor(rows).long())
        self.register_buffer('cols', torch.tensor(cols).long())
        
        self.grid_h, self.grid_w = 7, 5

    def forward_pre_chunk(self, x):
        """Standard CWT Forward Pass (No splitting)."""
        B, C, T = x.shape
        cwt_raw = self.conv(x) # (B, 32*F*2, T)
        cwt_reshaped = cwt_raw.view(B, C, self.num_freqs, 2, T)
        
        real = cwt_reshaped[..., 0, :] 
        imag = cwt_reshaped[..., 1, :] 
        
        mag = torch.sqrt(real.pow(2) + imag.pow(2))
        phase = torch.atan2(imag, real)
        
        features = torch.stack([mag, phase], dim=-1)
        features = features.permute(0, 4, 2, 1, 3)
        
        canvas = torch.zeros(
            B, 2, self.num_freqs, self.grid_h, self.grid_w, T,
            device=x.device, dtype=x.dtype
        )
        canvas[:, :, :, self.rows, self.cols, :] = features
        return canvas

    def forward(self, x):
        """Main forward pass with optional chunking."""
        B, C, Total_Time = x.shape
        full_cwt = self.forward_pre_chunk(x)
        
        if self.chunk_size_samples is None:
            return full_cwt
            
        if Total_Time % self.chunk_size_samples != 0:
            raise ValueError(f"Total Time {Total_Time} not divisible by {self.chunk_size_samples}")
            
        num_chunks = Total_Time // self.chunk_size_samples
        cwt_permuted = full_cwt.permute(0, 5, 1, 2, 3, 4)
        chunks = cwt_permuted.view(B, num_chunks, self.chunk_size_samples, 2, self.num_freqs, self.grid_h, self.grid_w)
        chunks_merged = chunks.reshape(B * num_chunks, self.chunk_size_samples, 2, self.num_freqs, self.grid_h, self.grid_w)
        output = chunks_merged.permute(0, 2, 3, 4, 5, 1)
        
        return output

    def _create_morlet_weights(self, freqs, fs, K, n_cycles_max=6.0):
        """
        Creates Morlet kernels with Dynamic Cycle Scaling for low frequencies.
        This allows low-frequency wavelets (0.5-4Hz) to fit inside 4s windows.
        """
        weights = torch.zeros(self.num_channels * self.num_freqs * 2, 1, K)
        t = np.linspace(-K/2/fs, K/2/fs, K)
        
        for i, f in enumerate(freqs):
            # --- DYNAMIC CYCLE SCALING (IMPROVED) ---
            # At 0.5Hz: cycles = 3.0 (fits in ~6s, okay for 4s window with padding)
            # At 1.0Hz: cycles = 4.0 (fits in 4s window)
            # At 4.0Hz: cycles = 6.0 (full resolution)
            # Above 4Hz: cycles = 6.0 (stays at max)
            
            if f < 1.0:
                # Scale from 3.0 at 0.5Hz to 4.0 at 1.0Hz
                cycles = 3.0 + (f / 1.0) * (4.0 - 3.0)
            elif f < 4.0:
                # Scale from 4.0 at 1Hz to 6.0 at 4Hz
                cycles = 4.0 + ((f - 1.0) / 3.0) * (6.0 - 4.0)
            else:
                # Full resolution for high frequencies
                cycles = n_cycles_max
                
            sigma = cycles / (2 * np.pi * f)
            
            sine = np.exp(2j * np.pi * f * t)
            gauss = np.exp(-t**2 / (2 * sigma**2))
            wavelet = sine * gauss
            
            # Normalize
            wavelet /= np.linalg.norm(wavelet)
            
            for c in range(self.num_channels):
                idx_real = (c * self.num_freqs * 2) + (i * 2)
                idx_imag = idx_real + 1
                weights[idx_real, 0, :] = torch.from_numpy(np.real(wavelet))
                weights[idx_imag, 0, :] = torch.from_numpy(np.imag(wavelet))
                    
        return weights


