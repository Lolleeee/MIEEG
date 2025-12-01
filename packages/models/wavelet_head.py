import torch
import torch.nn as nn
import numpy as np

class CWTHead(nn.Module):
    def __init__(
        self, 
        frequencies: list, 
        fs: int, 
        num_channels: int = 32,
        n_cycles: float = 5.0,
        trainable: bool = False
    ):
        super().__init__()
        self.num_channels = num_channels
        self.frequencies = frequencies
        self.num_freqs = len(frequencies)
        
        # 1. Setup Filters
        # We make the kernel size odd so padding is symmetric (no phase shift)
        kernel_size = int(4 * fs / np.min(frequencies)) 
        if kernel_size % 2 == 0: kernel_size += 1
        
        padding = kernel_size // 2 
        
        # 2. Convolution Layer (The CWT Engine)
        self.conv = nn.Conv1d(
            in_channels=num_channels,
            out_channels=num_channels * self.num_freqs * 2, # Real + Imag parts
            kernel_size=kernel_size,
            padding=padding,
            groups=num_channels, # IMPORTANT: Keeps channels separate
            bias=False
        )
        
        # Initialize with Morlet weights
        weights = self._create_morlet_weights(frequencies, fs, kernel_size, n_cycles=n_cycles)
        self.conv.weight.data = weights
        self.conv.weight.requires_grad = trainable
        
        # 3. Spatial Mapping Vectors (The "Advanced Indexing" setup)
        # Mapping: Channel ID -> (Row, Col)
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
            
        # Register as buffers so they move to GPU with the model
        self.register_buffer('rows', torch.tensor(rows).long())
        self.register_buffer('cols', torch.tensor(cols).long())
        
        self.grid_h, self.grid_w = 7, 5

    def forward(self, x):
        """
        x: (Batch, 32, Time)
        Returns: (Batch, 2, Freq, 7, 5, Time)
        """
        B, C, T = x.shape
        
        # Step 1: Convolve
        # Output: (B, 32*F*2, T)
        cwt_raw = self.conv(x)
        
        # Step 2: Reshape
        # (B, Channel, Freq, Real/Imag, T)
        cwt_reshaped = cwt_raw.view(B, C, self.num_freqs, 2, T)
        
        # Step 3: Mag & Phase
        real = cwt_reshaped[..., 0, :] # (B, C, F, T)
        imag = cwt_reshaped[..., 1, :] # (B, C, F, T)
        
        mag = torch.sqrt(real.pow(2) + imag.pow(2))
        phase = torch.atan2(imag, real)
        
        # Stack Mag/Phase. 
        # Let's stack on last dim first to be safe, then permute.
        # stack -> (B, C, F, T, 2)
        features = torch.stack([mag, phase], dim=-1)
        
        # We need target shape: (B, 2, F, 32, T) for the canvas assignment
        # Current: (B, 32, F, T, 2)
        # Permute: (0, 4, 2, 1, 3) -> (B, 2, F, 32, T)
        features = features.permute(0, 4, 2, 1, 3)
        
        # Step 4: Map to Grid
        canvas = torch.zeros(
            B, 2, self.num_freqs, self.grid_h, self.grid_w, T,
            device=x.device, dtype=x.dtype
        )
        
        # Now shapes match perfectly:
        # canvas slice: (B, 2, F, 32, T)
        # features:     (B, 2, F, 32, T)
        canvas[:, :, :, self.rows, self.cols, :] = features
        
        return canvas


    def _create_morlet_weights(self, freqs, fs, K, n_cycles=5.0):
        """
        Args:
            n_cycles (float): Number of cycles in the Gaussian envelope. 
                              Standard for EEG is 3.0 to 6.0.
        """
        weights = torch.zeros(self.num_channels * self.num_freqs * 2, 1, K)
        t = np.linspace(-K/2/fs, K/2/fs, K)
        
        for i, f in enumerate(freqs):
            # --- CONSTANT Q TRANSFORMATION ---
            # sigma scales with frequency to keep number of cycles constant
            # sigma = n_cycles / (2 * pi * f)
            # This ensures we capture ~n_cycles of the oscillation
            sigma = n_cycles / (2 * np.pi * f)
            
            # Complex Morlet
            sine = np.exp(2j * np.pi * f * t)
            gauss = np.exp(-t**2 / (2 * sigma**2))
            wavelet = sine * gauss
            
            # Energy Normalization
            # For CWT, we usually normalize by 1/sqrt(sigma) or L2 norm
            # L2 norm is safest for Neural Networks to keep gradients stable
            wavelet /= np.linalg.norm(wavelet)
            
            for c in range(self.num_channels):
                idx_real = (c * self.num_freqs * 2) + (i * 2)
                idx_imag = idx_real + 1
                weights[idx_real, 0, :] = torch.from_numpy(np.real(wavelet))
                weights[idx_imag, 0, :] = torch.from_numpy(np.imag(wavelet))
                
        return weights


if __name__ == "__main__":
    # Test Code
    fs = 100
    freqs = [10] # Just test 10Hz
    model = CWTHead(frequencies=freqs, fs=fs)

    # 1. Create a pure 10Hz Sine Wave on Channel 3
    t = torch.linspace(0, 1, fs) # 1 second
    sine_wave = torch.sin(2 * np.pi * 10 * t).unsqueeze(0).unsqueeze(0) # (1, 1, 100)
    data = torch.zeros(1, 32, 100)
    data[:, 3, :] = sine_wave # Put signal ONLY on Channel 3

    # 2. Run Model
    out = model(data) # (1, 2, 1, 7, 5, 100)
    print("Output shape:", out.shape)
    # 3. Verify
    # Channel 3 corresponds to Row 1, Col 1 in your map.
    # We expect high magnitude there, and 0 magnitude elsewhere.
    target_mag = out[0, 0, 0, 1, 1, :].mean().item()
    empty_mag = out[0, 0, 0, 0, 0, :].mean().item() # Spot (-1)

    print(f"Magnitude at Ch3 (1,1): {target_mag:.4f} (Should be > 0)")
    print(f"Magnitude at Empty (0,0): {empty_mag:.4f} (Should be 0.0)")

    if target_mag > 0.1 and empty_mag == 0.0:
        print("✅ SUCCESS: Mapping and CWT logic are correct.")
    else:
        print("❌ FAILURE: Something is wrong.")
