import torch
import numpy as np

def sliding_window_vae_decode(model, latents, window_size=256, overlap=64):
    """
    Decode latents using sliding windows with overlap-add reconstruction.
    
    Args:
        model: AudioLDM model with VAE decoder
        latents: Input latent tensor [B, C, T, F]
        window_size: Size of each window in latent frames (default: 256 ~= 10s)
        overlap: Overlap between windows in frames (default: 64 ~= 2.5s)
    
    Returns:
        Reconstructed mel spectrogram
    """
    device = latents.device
    B, C, T, F = latents.shape
    
    print(f"  🪟 Sliding Window VAE Decode:")
    print(f"    Input shape: {latents.shape}")
    print(f"    Window size: {window_size} frames (~{window_size/25.6:.1f}s)")
    print(f"    Overlap size: {overlap} frames (~{overlap/25.6:.1f}s)")
    
    # If input is smaller than window, use regular decoding
    if T <= window_size:
        print(f"    Input smaller than window - using regular decode")
        return model.decode_first_stage(latents)
    
    # Calculate windows
    step_size = window_size - overlap
    num_windows = (T - overlap + step_size - 1) // step_size
    
    print(f"    Step size: {step_size} frames")
    print(f"    Number of windows: {num_windows}")
    
    # Process each window
    windows = []
    window_positions = []
    
    for i in range(num_windows):
        start = i * step_size
        end = min(start + window_size, T)
        actual_window_size = end - start
        
        # Extract window
        window_latent = latents[:, :, start:end, :]
        
        # Decode window
        with torch.no_grad():
            window_mel = model.decode_first_stage(window_latent)
        
        windows.append(window_mel)
        window_positions.append((start, end, actual_window_size))
        
        print(f"      Window {i+1}: latent[{start}:{end}] -> mel shape {window_mel.shape}")
    
    # Calculate output dimensions
    mel_scale_factor = windows[0].shape[2] / window_positions[0][2]  # mel_frames / latent_frames
    total_mel_frames = int(T * mel_scale_factor)
    mel_channels = windows[0].shape[1]
    mel_freq_bins = windows[0].shape[3]
    
    print(f"    Mel scale factor: {mel_scale_factor:.2f}x")
    print(f"    Output mel shape: [{B}, {mel_channels}, {total_mel_frames}, {mel_freq_bins}]")
    
    # Initialize output tensor and weight accumulator
    output_mel = torch.zeros((B, mel_channels, total_mel_frames, mel_freq_bins), 
                            device=device, dtype=windows[0].dtype)
    weight_sum = torch.zeros_like(output_mel)
    
    # Create overlap-add weights (linear fade in/out)
    overlap_mel_size = int(overlap * mel_scale_factor)
    
    for i, (window_mel, (start, end, actual_size)) in enumerate(zip(windows, window_positions)):
        # Calculate mel positions
        mel_start = int(start * mel_scale_factor)
        mel_end = int(end * mel_scale_factor)
        actual_mel_size = window_mel.shape[2]
        
        # Create weight mask for this window
        weight_mask = torch.ones((1, 1, actual_mel_size, 1), device=device)
        
        # Apply fade-in at the beginning (except for first window)
        if i > 0 and overlap_mel_size > 0:
            fade_in_size = min(overlap_mel_size, actual_mel_size // 2)
            fade_in = torch.linspace(0, 1, fade_in_size, device=device)
            weight_mask[:, :, :fade_in_size, :] = fade_in.view(1, 1, -1, 1)
        
        # Apply fade-out at the end (except for last window)
        if i < len(windows) - 1 and overlap_mel_size > 0:
            fade_out_size = min(overlap_mel_size, actual_mel_size // 2)
            fade_out = torch.linspace(1, 0, fade_out_size, device=device)
            weight_mask[:, :, -fade_out_size:, :] = fade_out.view(1, 1, -1, 1)
        
        # Add weighted window to output
        mel_end_actual = mel_start + actual_mel_size
        output_mel[:, :, mel_start:mel_end_actual, :] += window_mel * weight_mask
        weight_sum[:, :, mel_start:mel_end_actual, :] += weight_mask
        
        print(f"      Added window {i+1} to output[{mel_start}:{mel_end_actual}]")
    
    # Normalize by weight sum to complete overlap-add
    output_mel = output_mel / torch.clamp(weight_sum, min=1e-8)
    
    print(f"    ✅ Sliding window VAE decode complete")
    
    return output_mel