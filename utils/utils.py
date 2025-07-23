import os
import numpy as np
import torch
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch 
import time
from audioldm import build_model
from audioldm.latent_diffusion.ddim import DDIMSampler
from audioldm.pipeline import duration_to_latent_t_size



# Visualization functions for spectrograms
def plot_latent_spectrogram(latent_tensor, title="Latent Spectrogram"):
    """Plot latent tensor as spectrogram"""
    # Convert to numpy and take first batch/channel
    if len(latent_tensor.shape) == 4:  # [batch, channels, time, freq]
        spec = latent_tensor[0, 0].cpu().numpy()  # Take first batch, first channel
    else:
        spec = latent_tensor.cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(spec.T, x_axis='time', y_axis='linear', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{title} - Shape: {spec.shape}')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.tight_layout()
    plt.show()
    
    print(f"Latent stats - Min: {spec.min():.3f}, Max: {spec.max():.3f}, Mean: {spec.mean():.3f}, Std: {spec.std():.3f}")

def plot_mel_spectrogram(mel_tensor, title="Mel Spectrogram", sr=16000):
    """Plot mel spectrogram"""
    # Convert to numpy and take first batch
    if len(mel_tensor.shape) == 4:  # [batch, channels, freq, time] 
        spec = mel_tensor[0, 0].cpu().numpy()  # Take first batch, first channel
    elif len(mel_tensor.shape) == 3:  # [batch, freq, time]
        spec = mel_tensor[0].cpu().numpy()  # Take first batch
    else:
        spec = mel_tensor.cpu().numpy()
    
    # Debug shape info
    print(f"DEBUG: mel_tensor.shape = {mel_tensor.shape}, spec.shape = {spec.shape}")
    
    # Ensure we have a 2D spectrogram [freq, time]
    if len(spec.shape) == 1:
        # If 1D, reshape based on expected mel dimensions (typically 80 mel bins)
        expected_mel_bins = 80
        time_frames = len(spec) // expected_mel_bins
        spec = spec[:expected_mel_bins * time_frames].reshape(expected_mel_bins, time_frames)
        print(f"DEBUG: Reshaped 1D to 2D: {spec.shape}")
    
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(spec, x_axis='time', y_axis='linear', sr=sr, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{title} - Shape: {spec.shape}')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    plt.show()
    
    print(f"Mel stats - Min: {spec.min():.3f}, Max: {spec.max():.3f}, Mean: {spec.mean():.3f}, Std: {spec.std():.3f}")

def check_for_nan_inf(tensor, name):
    """Check if tensor contains NaN or Inf values"""
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    
    if has_nan:
        print(f"⚠️  WARNING: {name} contains NaN values!")
    if has_inf:
        print(f"⚠️  WARNING: {name} contains Inf values!")
    if not has_nan and not has_inf:
        print(f"✅ {name} is clean (no NaN/Inf)")
    
    return has_nan, has_inf

# MultiDiffusion Helper Functions
def should_use_multidiffusion(total_frames, chunk_size):
    """Determine if MultiDiffusion is needed"""
    return total_frames > chunk_size

def create_multidiffusion_chunks(total_frames, chunk_size, overlap_ratio=0.75):
    """Create overlapping chunks for MultiDiffusion following the paper approach"""
    overlap_frames = int(chunk_size * overlap_ratio)
    advance_step = chunk_size - overlap_frames
    
    print(f"DEBUG: total_frames={total_frames}, chunk_size={chunk_size}, overlap_frames={overlap_frames}, advance_step={advance_step}")
    
    chunks = []
    start = 0
    while start < total_frames:
        end = min(start + chunk_size, total_frames)
        chunks.append((start, end))
        print(f"DEBUG: chunk {len(chunks)}: ({start}, {end}) - frames: {end-start}")
        if end >= total_frames:
            break
        start += advance_step
    
    print(f"DEBUG: Created {len(chunks)} chunks covering frames 0-{chunks[-1][1]}")
    return chunks, overlap_frames, advance_step

def pad_chunk_to_size(x_chunk, target_frames):
    """Pad chunk to target size if needed"""
    current_frames = x_chunk.shape[2]  # Assuming shape [batch, channels, time, freq]
    if current_frames < target_frames:
        # Pad with zeros on the time dimension
        pad_frames = target_frames - current_frames
        padding = torch.zeros(x_chunk.shape[0], x_chunk.shape[1], pad_frames, x_chunk.shape[3], 
                            device=x_chunk.device, dtype=x_chunk.dtype)
        x_chunk = torch.cat([x_chunk, padding], dim=2)
    return x_chunk

def unpad_chunk_result(result, original_frames):
    """Remove padding from chunk result"""
    if result.shape[2] > original_frames:
        result = result[:, :, :original_frames, :]
    return result

def overlap_average_noise_predictions(noise_predictions, full_shape):
    """Average overlapping noise predictions from chunks"""
    device = noise_predictions[0][2].device
    weight_sum = torch.zeros(full_shape, device=device)
    weighted_sum = torch.zeros(full_shape, device=device)
    
    for start_frame, end_frame, noise_pred in noise_predictions:
        weighted_sum[:, :, start_frame:end_frame, :] += noise_pred
        weight_sum[:, :, start_frame:end_frame, :] += 1.0
    
    return weighted_sum / weight_sum

def ddim_step_full_tensor(x_full, noise_pred_full, timestep, sampler, index, eta, unconditional_guidance_scale=1.0):
    """Apply DDIM step to full tensor (extracted from p_sample_ddim)"""
    
    # Extract DDIM parameters for this timestep
    a_t = sampler.ddim_alphas[index]
    a_prev = sampler.ddim_alphas_prev[index] 
    sigma_t = sampler.ddim_sigmas[index]
    sqrt_one_minus_at = sampler.ddim_sqrt_one_minus_alphas[index]
    
    # Convert to proper tensor shapes
    b = x_full.shape[0]
    device = x_full.device
    a_t = torch.full((b, 1, 1, 1), a_t, device=device)
    a_prev = torch.full((b, 1, 1, 1), a_prev, device=device)
    sigma_t = torch.full((b, 1, 1, 1), sigma_t, device=device)
    sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_at, device=device)
    
    # DDIM math (same as p_sample_ddim)
    pred_x0 = (x_full - sqrt_one_minus_at * noise_pred_full) / a_t.sqrt()
    dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * noise_pred_full
    noise = sigma_t * torch.randn_like(x_full) if index > 0 else 0.
    
    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
    
    return x_prev

def chunked_noise_prediction(model, x_full, timestep, conditioning, unconditional_conditioning, 
                           unconditional_guidance_scale, chunks, chunk_frames):
    """Get noise predictions by applying model to chunks, then overlap average"""
    
    noise_predictions = []
    
    for start_frame, end_frame in chunks:
        # Extract chunk
        x_chunk = x_full[:, :, start_frame:end_frame, :]
        original_chunk_frames = end_frame - start_frame
        
        # Pad chunk to consistent size for U-Net processing
        x_chunk_padded = pad_chunk_to_size(x_chunk, chunk_frames)
        
        # Apply model to padded chunk with CFG
        if unconditional_guidance_scale == 1.0:
            noise_pred_padded = model.apply_model(x_chunk_padded, timestep, conditioning)
        else:
            # Batch conditional and unconditional
            x_in = torch.cat([x_chunk_padded] * 2)
            t_in = torch.cat([timestep] * 2)
            c_in = torch.cat([unconditional_conditioning, conditioning])
            
            noise_uncond, noise_cond = model.apply_model(x_in, t_in, c_in).chunk(2)
            
            # CFG
            noise_pred_padded = noise_uncond + unconditional_guidance_scale * (noise_cond - noise_uncond)
        
        # Remove padding to get back to original chunk size
        noise_pred = unpad_chunk_result(noise_pred_padded, original_chunk_frames)
        
        noise_predictions.append((start_frame, end_frame, noise_pred))
    
    # Overlap average all noise predictions
    full_noise_pred = overlap_average_noise_predictions(noise_predictions, x_full.shape)
    
    return full_noise_pred

def multidiffusion_sample_clean(sampler, shape, conditioning, unconditional_conditioning,
                               unconditional_guidance_scale, eta, x_T, S=200,
                               chunk_size=256, overlap_ratio=0.75,
                               chunk_frames=None, overlap_frames=None):
    """
    Clean MultiDiffusion: Proper implementation following Bar-Tal et al. paper
    - Fixed chunk size with 75% overlap
    - Only use MultiDiffusion when total_frames > chunk_size
    - Chunk expensive UNet, use standard scheduling on full tensors
    """
    model = sampler.model
    device = model.device

    # Prepare DDIM schedule
    sampler.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=False)

    # Alias compatibility parameters
    if chunk_frames is not None:
        chunk_size = chunk_frames
    if overlap_frames is not None:
        overlap_ratio = overlap_frames / chunk_size if overlap_frames > 1 else overlap_frames

    # Unpack shape
    batch_size, channels, total_frames, freq_bins = shape

    # Determine chunking strategy
    if not should_use_multidiffusion(total_frames, chunk_size):
        chunks = [(0, total_frames)]
        actual_chunk_size = total_frames
        ov_frames = 0
        print(f"SHORT AUDIO: {total_frames} frames <= {chunk_size} chunk size - using standard DDIM")
    else:
        chunks, ov_frames, _ = create_multidiffusion_chunks(total_frames, chunk_size, overlap_ratio)
        actual_chunk_size = chunk_size
        print(f"LONG AUDIO: Using MultiDiffusion with {len(chunks)} chunks, overlap={ov_frames}")

    # Timesteps and reverse order for DDIM
    timesteps = sampler.ddim_timesteps[:S]
    time_sequence = np.flip(timesteps)

    # Initialize current latent
    x = x_T.clone()

    # Main denoising loop
    for i, step in enumerate(time_sequence):
        index = len(timesteps) - i - 1
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)

        if i < 5 or i % 20 == 0:
            mode = "MultiDiffusion" if len(chunks)>1 else "Standard"
            print(f"Step {i+1}/{len(timesteps)} ({mode}), timestep: {step}")

        # 1. Chunked noise prediction
        noise_pred = chunked_noise_prediction(
            model, x, t, conditioning, unconditional_conditioning,
            unconditional_guidance_scale, chunks, actual_chunk_size
        )

        # 2. DDIM scheduler step
        x = ddim_step_full_tensor(x, noise_pred, t, sampler, index, eta)

        if i % 20 == 0:
            print(f"  ✅ Completed step {i+1}")

    return x
