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
from tqdm import tqdm

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

def plot_mel_spectrogram(mel_tensor, title="Mel Spectrogram", sr=16000, hop_length=160):
    """Plot mel spectrogram with robust dimension handling."""
    # Convert to numpy and remove batch/channel dimensions
    spec = mel_tensor.squeeze().cpu().numpy()

    # If 3D, take the first channel
    if spec.ndim > 2:
        spec = spec[0]

    # Check if the first dimension is time (larger than frequency)
    if spec.shape[0] > spec.shape[1]:
        spec = spec.T  # Transpose to [frequency, time]

    plt.figure(figsize=(12, 6))
    # Pass the correct hop_length to specshow
    librosa.display.specshow(spec, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{title} - Shape: {spec.shape}')
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

def ensure_correct_dimensions(tensor, expected_shape):
    """Ensure the tensor has the correct dimensions, permuting if necessary."""
    if tensor.shape != expected_shape:
        print(f"⚠️ Dimension mismatch: Expected {expected_shape}, got {tensor.shape}. Permuting...")
        # Assuming the mismatch is between time and frequency dimensions
        tensor = tensor.permute(0, 1, 3, 2)  # Swap time and frequency
        print(f"✅ Dimensions corrected: {tensor.shape}")
    return tensor

# Update chunked_noise_prediction to ensure correct dimensions
def chunked_noise_prediction(model, x_full, timestep, conditioning, unconditional_conditioning,
                           unconditional_guidance_scale, chunks, chunk_frames):
    """Get noise predictions by applying model to chunks, then overlap average"""

    # Ensure x_full has correct dimensions by swapping time and frequency if needed
    if x_full.shape[2] < x_full.shape[3]:
        x_full = x_full.permute(0, 1, 3, 2)  # Swap time and frequency

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
    with tqdm(total=len(time_sequence), desc="Diffusion Steps") as pbar:
        for i, step in enumerate(time_sequence):
            index = len(timesteps) - i - 1
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)

            # 1. Chunked noise prediction
            noise_pred = chunked_noise_prediction(
                model, x, t, conditioning, unconditional_conditioning,
                unconditional_guidance_scale, chunks, actual_chunk_size
            )

            # 2. DDIM scheduler step
            x = ddim_step_full_tensor(x, noise_pred, t, sampler, index, eta)

            # Update progress bar
            pbar.update(1)

    return x

def multidiffusion_sample_temporal(sampler, shape, segment_embeddings, prompt_segments, 
                                 unconditional_conditioning, unconditional_guidance_scale, 
                                 eta, x_T, S=200, chunk_frames=256, overlap_frames=192, duration=30.0,
                                 full_prompt_emb=None, full_prompt_cfg_scale=None):
    """
    Temporal MultiDiffusion with clean segment boundaries (no crossfading).
    Uses simple, separate prompts with overlap averaging but NO crossfading weights.
    """
    model = sampler.model
    device = model.device
    sampler.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=False)

    batch_size, channels, total_frames, freq_bins = shape
    
    # --- 1. Create chunks and assign a simple prompt to each ---
    advance_step = chunk_frames - overlap_frames
    chunks = []
    chunk_conditionings = []
    
    # Use simple prompts, assign based on chunk CENTER with proper boundaries
    for i in range(0, total_frames, advance_step):
        start, end = i, min(i + chunk_frames, total_frames)
        if end - start < chunk_frames // 4 and len(chunks) > 0: 
            continue  # Skip tiny final chunks

        chunks.append((start, end))
        
        # Use chunk CENTER for assignment with inclusive boundaries
        chunk_center_time = ((start + end) / 2) / 25.6  # frames to seconds
        
        # Find which segment this chunk center belongs to
        assigned_segment = len(prompt_segments) - 1  # Default to last segment
        segment_name = prompt_segments[-1][2]
        
        for j, (seg_start, seg_end, seg_prompt) in enumerate(prompt_segments):
            # Use <= for end boundary to ensure even distribution
            if seg_start <= chunk_center_time <= seg_end:
                assigned_segment = j
                segment_name = seg_prompt
                break
        
        chunk_conditionings.append(segment_embeddings[assigned_segment])
        
        print(f"  Chunk {len(chunks)}: frames[{start}:{end}] time[{start/25.6:.1f}-{end/25.6:.1f}s] CENTER={chunk_center_time:.1f}s -> '{segment_name}' (seg {assigned_segment})")

    print(f"🎭 TEMPORAL MULTIDIFFUSION: Processing {len(chunks)} chunks with clean boundaries (no crossfading).")
    
    timesteps = np.flip(sampler.ddim_timesteps[:S])
    x = x_T.clone()

    with tqdm(total=len(timesteps), desc="Temporal Diffusion") as pbar:
        for i, step in enumerate(timesteps):
            index = len(timesteps) - 1 - i
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            
            # --- 2. Predict noise for each chunk using its assigned simple prompt ---
            noise_predictions = []
            for (start, end), cond in zip(chunks, chunk_conditionings):
                x_chunk = x[:, :, start:end, :]
                original_frames = x_chunk.shape[2]
                
                x_chunk_padded = pad_chunk_to_size(x_chunk, chunk_frames)

                # Batch conditional and unconditional for efficiency
                x_in = torch.cat([x_chunk_padded] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, cond])
                
                noise_uncond, noise_cond = model.apply_model(x_in, t_in, c_in).chunk(2)
                noise_pred_padded = noise_uncond + unconditional_guidance_scale * (noise_cond - noise_uncond)

                noise_pred = unpad_chunk_result(noise_pred_padded, original_frames)
                noise_predictions.append((start, end, noise_pred))

            # --- 3. Use standard overlap averaging (NO crossfading weights) ---
            full_noise_pred = overlap_average_noise_predictions(noise_predictions, x.shape)
            
            # --- 4. Perform DDIM step on the full tensor ---
            x = ddim_step_full_tensor(x, full_noise_pred, t, sampler, index, eta)
            pbar.update(1)
    
    return x
    """
    Temporal MultiDiffusion: Apply different prompts to different time segments
    
    Args:
        sampler: DDIM sampler
        shape: [batch, channels, time_frames, freq_bins]
        segment_embeddings: List of conditioning embeddings for each prompt segment
        prompt_segments: List of (start_time, end_time, prompt) tuples
        unconditional_conditioning: Unconditional embedding for CFG
        unconditional_guidance_scale: CFG scale
        eta: DDIM eta parameter
        x_T: Initial noise tensor
        S: Number of diffusion steps
        chunk_frames: Size of each chunk
        overlap_frames: Overlap between chunks
        duration: Total duration in seconds
    
    Returns:
        Denoised latent tensor
    """
    model = sampler.model
    device = model.device
    
    # Prepare DDIM schedule
    sampler.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=False)
    
    # Unpack shape
    batch_size, channels, total_frames, freq_bins = shape
    
    # Create chunk-to-prompt mapping
    def get_chunk_conditioning(chunk_start_frame, chunk_end_frame):
        """Determine which prompt segment(s) this chunk overlaps with"""
        # Convert chunk frames to time
        chunk_start_time = chunk_start_frame / 25.6  # frames to seconds
        chunk_end_time = chunk_end_frame / 25.6
        chunk_mid_time = (chunk_start_time + chunk_end_time) / 2
        
        # Find which segment the chunk center belongs to
        for i, (seg_start, seg_end, _) in enumerate(prompt_segments):
            if seg_start <= chunk_mid_time < seg_end:
                return segment_embeddings[i]
        
        # Fallback to last segment if beyond range
        return segment_embeddings[-1]
    
    # Create chunks with overlap
    advance_step = chunk_frames - overlap_frames
    chunks = []
    chunk_conditionings = []
    
    for i in range(0, total_frames, advance_step):
        start = i
        end = min(start + chunk_frames, total_frames)
        if end - start < chunk_frames // 2:  # Skip very small final chunks
            break
        chunks.append((start, end))
        
        # Get conditioning for this chunk
        chunk_cond = get_chunk_conditioning(start, end)
        chunk_conditionings.append(chunk_cond)
        
        # Convert frames to time for logging
        start_time = start / 25.6
        end_time = end / 25.6
        segment_idx = None
        for j, (seg_start, seg_end, _) in enumerate(prompt_segments):
            if seg_start <= (start_time + end_time) / 2 < seg_end:
                segment_idx = j
                break
        
        print(f"  Chunk {len(chunks)}: frames[{start}:{end}] time[{start_time:.1f}-{end_time:.1f}s] -> prompt segment {segment_idx + 1 if segment_idx is not None else 'N/A'}")
    
    print(f"🎭 TEMPORAL MULTIDIFFUSION: {len(chunks)} chunks with temporal conditioning")
    
    # Timesteps and reverse order for DDIM
    timesteps = sampler.ddim_timesteps[:S]
    time_sequence = np.flip(timesteps)
    
    # Initialize current latent
    x = x_T.clone()
    
    # Main denoising loop
    with tqdm(total=len(time_sequence), desc="Temporal Diffusion") as pbar:
        for i, step in enumerate(time_sequence):
            index = len(timesteps) - i - 1
            t_tensor = torch.full((batch_size,), step, device=device, dtype=torch.long)
            
            # Process each chunk with its specific conditioning
            noise_predictions = []
            for chunk_idx, ((start, end), chunk_cond) in enumerate(zip(chunks, chunk_conditionings)):
                # Extract chunk
                x_chunk = x[:, :, start:end, :]
                
                # Apply model with chunk-specific conditioning
                noise_uncond = model.apply_model(x_chunk, t_tensor, unconditional_conditioning)
                noise_cond = model.apply_model(x_chunk, t_tensor, chunk_cond)
                
                # Classifier-free guidance
                noise_pred = noise_uncond + unconditional_guidance_scale * (noise_cond - noise_uncond)
                
                noise_predictions.append((start, end, noise_pred))
            
            # Overlap-average the noise predictions
            full_noise_pred = torch.zeros_like(x)
            weight_sum = torch.zeros_like(x)
            
            for start, end, noise_pred in noise_predictions:
                # Create weight mask with linear fade for overlaps
                chunk_length = end - start
                weights = torch.ones_like(noise_pred)
                
                # Apply fade-in at start (except for first chunk)
                if start > 0 and overlap_frames > 0:
                    fade_length = min(overlap_frames, chunk_length // 2)
                    fade_in = torch.linspace(0, 1, fade_length, device=device)
                    weights[:, :, :fade_length, :] *= fade_in[None, None, :, None]
                
                # Apply fade-out at end (except for last chunk)
                if end < total_frames and overlap_frames > 0:
                    fade_length = min(overlap_frames, chunk_length // 2)
                    fade_out = torch.linspace(1, 0, fade_length, device=device)
                    weights[:, :, -fade_length:, :] *= fade_out[None, None, :, None]
                
                # Add weighted prediction
                full_noise_pred[:, :, start:end, :] += noise_pred * weights
                weight_sum[:, :, start:end, :] += weights
            
        # Normalize by weights
        full_noise_pred = full_noise_pred / torch.clamp(weight_sum, min=1e-8)
        
        # DDIM step on full tensor
        x = ddim_step_full_tensor(x, full_noise_pred, t_tensor, sampler, index, eta)
        
        pbar.update(1)
    
    return x