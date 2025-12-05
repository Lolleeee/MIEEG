import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from packages.models.vqae_light import VQAELight, VQAELightConfig
def quick_frequency_test(model, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Fast test: Generate pure frequency signals and check reconstruction quality.
    Returns: Dictionary with SNR per frequency band.
    """
    model.eval()
    model = model.to(device)
    
    # Test frequencies spanning your CWT range
    test_freqs = [2, 5, 10, 15, 20, 30, 40, 50]  # Hz
    fs = 160  # Your sampling rate
    duration = 4.0  # seconds (640 samples)
    T = int(duration * fs)
    
    results = {}
    
    print("=" * 60)
    print("FREQUENCY RECONSTRUCTION TEST")
    print("=" * 60)
    
    for freq in test_freqs:
        # Generate pure sinusoid on all 32 channels
        t = torch.arange(T, device=device) / fs
        signal = torch.sin(2 * np.pi * freq * t)
        
        # Broadcast to all channels: (1, 32, 640)
        x = signal.unsqueeze(0).unsqueeze(0).repeat(1, 32, 1)
        
        # Add tiny noise to avoid numerical issues
        x = x + torch.randn_like(x) * 0.01
        
        with torch.no_grad():
            out = model(x)
            recon = out['reconstruction']  # (1, 32, 640)
        
        # Compute SNR in dB
        signal_power = (x ** 2).mean()
        noise_power = ((x - recon) ** 2).mean() + 1e-10
        snr_db = 10 * torch.log10(signal_power / noise_power)
        
        # Compute frequency-domain correlation
        X_fft = torch.fft.rfft(x[0, 0])  # First channel
        R_fft = torch.fft.rfft(recon[0, 0])
        
        # Find peak frequency in reconstruction
        power_spectrum = torch.abs(R_fft) ** 2
        freqs = torch.fft.rfftfreq(T, 1/fs)
        peak_idx = torch.argmax(power_spectrum)
        peak_freq = freqs[peak_idx].item()
        
        # Check if peak is within 2 Hz of target
        freq_accurate = abs(peak_freq - freq) < 2.0
        
        results[freq] = {
            'snr_db': snr_db.item(),
            'peak_freq': peak_freq,
            'accurate': freq_accurate
        }
        
        # Visual indicator
        status = "✓" if snr_db > 10 and freq_accurate else "✗"
        print(f"{status} {freq:5.1f} Hz: SNR = {snr_db:6.2f} dB | "
              f"Peak at {peak_freq:5.1f} Hz | "
              f"{'GOOD' if freq_accurate else 'FAIL'}")
    
    print("=" * 60)
    
    # Summary statistics
    low_freqs = [f for f in test_freqs if f < 15]
    high_freqs = [f for f in test_freqs if f >= 30]
    
    avg_snr_low = np.mean([results[f]['snr_db'] for f in low_freqs])
    avg_snr_high = np.mean([results[f]['snr_db'] for f in high_freqs])
    
    print(f"\nSUMMARY:")
    print(f"  Low freq (<15Hz) avg SNR:  {avg_snr_low:6.2f} dB")
    print(f"  High freq (≥30Hz) avg SNR: {avg_snr_high:6.2f} dB")
    print(f"  SNR drop: {avg_snr_low - avg_snr_high:6.2f} dB")
    
    if avg_snr_high < 5:
        print("\n⚠️  HIGH FREQUENCY RECONSTRUCTION FAILED")
        print("   Model cannot reconstruct frequencies above 30 Hz")
    elif avg_snr_low - avg_snr_high > 15:
        print("\n⚠️  SEVERE HIGH-FREQUENCY ATTENUATION")
        print("   Check decoder upsampling and normalization")
    else:
        print("\n✓ Frequency reconstruction looks OK")
    
    return results


def visualize_spectrum_comparison(model, freq=40, config=None, device='cpu'):
    """
    Detailed visualization for a single frequency.
    Use this to debug WHERE the high frequencies are lost.
    """
    model.eval()
    model = model.to(device)
    
    fs = 160
    T = 640
    t = torch.arange(T, device=device) / fs
    
    # Generate test signal
    signal = torch.sin(2 * np.pi * freq * t)
    x = signal.unsqueeze(0).unsqueeze(0).repeat(1, 32, 1) + torch.randn(1, 32, T, device=device) * 0.01
    
    with torch.no_grad():
        out = model(x)
        recon = out['reconstruction']
    
    # Move to CPU for plotting
    x_np = x[0, 0].cpu().numpy()
    recon_np = recon[0, 0].cpu().numpy()
    
    # Compute spectra
    freqs = np.fft.rfftfreq(T, 1/fs)
    X_fft = np.abs(np.fft.rfft(x_np))
    R_fft = np.abs(np.fft.rfft(recon_np))
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time domain
    axes[0].plot(t.cpu().numpy()[:320], x_np[:320], label='Original', alpha=0.7)
    axes[0].plot(t.cpu().numpy()[:320], recon_np[:320], label='Reconstructed', alpha=0.7)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Time Domain: {freq} Hz Signal')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Frequency domain
    axes[1].semilogy(freqs, X_fft, label='Original', alpha=0.7, linewidth=2)
    axes[1].semilogy(freqs, R_fft, label='Reconstructed', alpha=0.7, linewidth=2)
    axes[1].axvline(freq, color='red', linestyle='--', alpha=0.5, label=f'Target: {freq} Hz')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title('Frequency Spectrum (Log Scale)')
    axes[1].set_xlim(0, 80)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'freq_test_{freq}hz.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to freq_test_{freq}hz.png")
    plt.close()


def trace_frequency_through_decoder(model, freq=40, device='cpu'):
    """
    Hook-based analysis: Track where high frequencies disappear in decoder.
    """
    model.eval()
    model = model.to(device)
    
    # Generate test signal
    fs = 160
    T = 640
    t = torch.arange(T, device=device) / fs
    signal = torch.sin(2 * np.pi * freq * t)
    x = signal.unsqueeze(0).unsqueeze(0).repeat(1, 32, 1)
    
    # Storage for intermediate activations
    activations = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor) and output.dim() == 3:
                # Compute power in high-freq band (30-80 Hz)
                fft = torch.fft.rfft(output[0, 0])  # First channel
                power = torch.abs(fft) ** 2
                freqs = torch.fft.rfftfreq(output.shape[-1], 1/fs)
                
                high_freq_mask = (freqs >= 30) & (freqs <= 80)
                low_freq_mask = (freqs >= 1) & (freqs < 15)
                
                high_power = power[high_freq_mask].mean().item()
                low_power = power[low_freq_mask].mean().item()
                ratio = high_power / (low_power + 1e-10)
                
                activations[name] = {
                    'shape': output.shape,
                    'high_power': high_power,
                    'low_power': low_power,
                    'ratio': ratio
                }
        return hook
    
    # Register hooks on decoder
    hooks = []
    for name, module in model.decoder.named_modules():
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    with torch.no_grad():
        _ = model(x)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Print results
    print("\n" + "=" * 80)
    print(f"DECODER FREQUENCY TRACE ({freq} Hz input)")
    print("=" * 80)
    print(f"{'Layer':<40} {'Shape':<20} {'High/Low Ratio':<15} {'Status'}")
    print("-" * 80)
    
    for name, stats in activations.items():
        ratio = stats['ratio']
        status = "✓" if ratio > 0.01 else "✗ LOST"
        print(f"{name:<40} {str(stats['shape']):<20} {ratio:>8.6f}       {status}")
    
    print("=" * 80)
    
    return activations

# ====================
# USAGE
# ====================

if __name__ == "__main__":
    # Your existing model setup
    config = VQAELightConfig(
        num_input_channels=2, num_freq_bands=25, spatial_rows=7, spatial_cols=5, 
        time_samples=160, use_cwt=True, chunk_samples=160, 
        use_inverse_cwt=False,  # Test without iCWT first
        embedding_dim=1024  # Your 1024-dim test
    )
    
    model = VQAELight(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Quick overall test (10 seconds)
    print("\n1. RUNNING QUICK FREQUENCY TEST...")
    results = quick_frequency_test(model, config, device=device)
    
    # 2. Detailed visualization for high frequency
    print("\n2. GENERATING DETAILED 40 Hz VISUALIZATION...")
    visualize_spectrum_comparison(model, freq=40, config=config, device=device)
    
    # 3. Trace where frequencies are lost
    print("\n3. TRACING FREQUENCY THROUGH DECODER...")
    trace = trace_frequency_through_decoder(model, freq=40, device=device)
