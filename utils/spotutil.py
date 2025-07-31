import os
import numpy as np
import torch
import time
from audioldm import build_model
from audioldm.latent_diffusion.ddim import DDIMSampler
from audioldm.pipeline import duration_to_latent_t_size
from tqdm import tqdm
import random

class SpotDiffusionConfig:
    """Configuration for SpotDiffusion parameters"""
    def __init__(self, 
                 tile_size_time=256,     # Tile size in time dimension
                 min_overlap=32,         # Minimum overlap between tiles (not used in non-overlapping)
                 max_shift_ratio=0.5):   # Maximum shift as ratio of tile size
        self.tile_size_time = tile_size_time
        self.min_overlap = min_overlap
        self.max_shift_ratio = max_shift_ratio

def generate_random_shift(shape, config, step):
    """Generate random shift for SpotDiffusion at given timestep"""
    batch_size, channels, time_frames, freq_frames = shape
    
    # Calculate maximum shift based on tile size (time only)
    max_shift_time = int(config.tile_size_time * config.max_shift_ratio)
    
    # Generate positive-only random shifts (as per SpotDiffusion paper: s(t) ~ U(0, W))
    # Use step to vary the random state but don't override global seed
    rng = torch.Generator()
    rng.manual_seed(hash(step) % (2**32))
    
    shift_time = torch.randint(0, max_shift_time + 1, (1,), generator=rng).item()
    
    return shift_time

def apply_shift(latent, shift_time):
    """Apply circular shift to latent tensor (time only)"""
    if shift_time != 0:
        latent = torch.roll(latent, shifts=shift_time, dims=2)  # Time dimension
    return latent

def reverse_shift(latent, shift_time):
    """Reverse the circular shift (time only)"""
    if shift_time != 0:
        latent = torch.roll(latent, shifts=-shift_time, dims=2)
    return latent

def create_tiles(shape, config):
    """Create tile coordinates for SpotDiffusion (time only, full frequency)"""
    batch_size, channels, time_frames, freq_frames = shape
    
    tiles = []
    
    # Calculate number of tiles needed (time only)
    time_tiles = max(1, (time_frames + config.tile_size_time - 1) // config.tile_size_time)
    
    # Generate non-overlapping tiles (full frequency spectrum per tile)
    for t_idx in range(time_tiles):
        # Calculate tile boundaries
        t_start = t_idx * config.tile_size_time
        t_end = min(t_start + config.tile_size_time, time_frames)
        
        # Always use full frequency spectrum
        f_start = 0
        f_end = freq_frames
        
        # Only add tile if it has meaningful size
        if t_end > t_start:
            tiles.append((t_start, t_end, f_start, f_end))
    
    return tiles

def spotdiffusion_denoise_step(sampler, x_t, t, conditioning, unconditional_conditioning, 
                              unconditional_guidance_scale, eta, config, step):
    """Single denoising step using SpotDiffusion tiling with GLOBAL noise coherence and reflection padding"""
    
    # Generate random shift for this timestep (time only)
    shift_time = generate_random_shift(x_t.shape, config, step)
    
    # REFLECTION PADDING: Pad before shifting to avoid unnatural circular boundaries
    pad_size = 64  # Sufficient padding for convolution windows
    x_padded = torch.nn.functional.pad(x_t, (0, 0, pad_size, pad_size), mode='reflect')
    
    # Apply shift to padded tensor (now wrap-around happens in natural padded region)
    x_shifted = apply_shift(x_padded, shift_time)
    
    # Create tiles
    tiles = create_tiles(x_shifted.shape, config)
    
    # Step 1: Predict noise for each tile (WITHOUT applying DDIM step yet)
    noise_predictions = []
    
    for t_start, t_end, f_start, f_end in tiles:
        # Extract tile
        tile = x_shifted[:, :, t_start:t_end, f_start:f_end]
        
        # Create timestep tensor for this tile
        t_tensor = torch.full((tile.shape[0],), t, device=tile.device, dtype=torch.long)
        
        # Predict noise using model (CFG if needed)
        with torch.no_grad():
            if unconditional_guidance_scale != 1.0:
                # Batch conditional and unconditional
                tile_input = torch.cat([tile, tile])
                t_input = torch.cat([t_tensor, t_tensor])
                cond_input = torch.cat([unconditional_conditioning, conditioning])
                
                # Get noise predictions
                noise_pred_combined = sampler.model.apply_model(tile_input, t_input, cond_input)
                noise_pred_uncond, noise_pred_cond = noise_pred_combined.chunk(2)
                
                # Apply classifier-free guidance
                noise_pred_tile = noise_pred_uncond + unconditional_guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # No CFG
                noise_pred_tile = sampler.model.apply_model(tile, t_tensor, conditioning)
        
        noise_predictions.append((noise_pred_tile, t_start, t_end, f_start, f_end))
    
    # Step 2: Assemble full noise prediction tensor (this will also be padded)
    full_noise_pred = torch.zeros_like(x_shifted)
    tile_counts = torch.zeros_like(x_shifted)
    
    for noise_pred_tile, t_start, t_end, f_start, f_end in noise_predictions:
        full_noise_pred[:, :, t_start:t_end, f_start:f_end] += noise_pred_tile
        tile_counts[:, :, t_start:t_end, f_start:f_end] += 1
    
    # Average overlapping regions (though tiles are non-overlapping)
    full_noise_pred = full_noise_pred / torch.clamp(tile_counts, min=1)
    
    # Step 3: Reverse shift on padded noise prediction
    full_noise_pred_unshifted = reverse_shift(full_noise_pred, shift_time)
    
    # Step 4: CROP back to original size (remove reflection padding)
    full_noise_pred_cropped = full_noise_pred_unshifted[:, :, pad_size:-pad_size, :]
    
    # Step 5: Find DDIM parameters for this timestep
    timestep_indices = torch.where(torch.tensor(sampler.ddim_timesteps, device=x_t.device) == t)[0]
    if len(timestep_indices) > 0:
        index = timestep_indices[0].item()
    else:
        # Fallback: find closest timestep
        index = torch.argmin(torch.abs(torch.tensor(sampler.ddim_timesteps, device=x_t.device) - t)).item()
    
    # Get DDIM coefficients
    a_t = sampler.ddim_alphas[index]
    a_prev = sampler.ddim_alphas_prev[index] 
    sigma_t = sampler.ddim_sigmas[index]
    sqrt_one_minus_at = sampler.ddim_sqrt_one_minus_alphas[index]
    
    # Convert to proper tensor shapes
    b = x_t.shape[0]  # Use original x_t shape, not padded
    device = x_t.device
    a_t = torch.full((b, 1, 1, 1), a_t, device=device)
    a_prev = torch.full((b, 1, 1, 1), a_prev, device=device)
    sigma_t = torch.full((b, 1, 1, 1), sigma_t, device=device)
    sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_at, device=device)
    
    # Step 6: Generate ONE global stochastic noise tensor for original size
    global_stochastic_noise = sigma_t * torch.randn_like(x_t) if index > 0 else 0.0
    
    # Step 7: Apply DDIM step to ORIGINAL x_t using cropped noise prediction
    # DDIM formula: x_{t-1} = sqrt(α_{t-1}) * pred_x0 + sqrt(1 - α_{t-1} - σ_t²) * ε + σ_t * noise
    pred_x0 = (x_t - sqrt_one_minus_at * full_noise_pred_cropped) / a_t.sqrt()
    dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * full_noise_pred_cropped
    
    x_denoised = a_prev.sqrt() * pred_x0 + dir_xt + global_stochastic_noise
    
    return x_denoised

def spotdiffusion_sample(sampler, shape, conditioning, unconditional_conditioning,
                        unconditional_guidance_scale=3.0, eta=0.1, x_T=None, S=200, config=None):
    """
    SpotDiffusion sampling for AudioLDM
    
    Args:
        sampler: DDIM sampler
        shape: Latent shape [batch, channels, time, freq]
        conditioning: Text conditioning
        unconditional_conditioning: Unconditional conditioning
        unconditional_guidance_scale: CFG scale
        eta: DDIM eta parameter
        x_T: Initial noise (if None, random noise is used)
        S: Number of diffusion steps
        config: SpotDiffusionConfig object
    """
    
    if config is None:
        config = SpotDiffusionConfig()
    
    device = sampler.model.device
    
    # Initialize noise
    if x_T is None:
        x_T = torch.randn(shape).to(device)
    
    # Make schedule if not already done
    if not hasattr(sampler, 'ddim_timesteps'):
        sampler.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=False)
    
    # Get timesteps
    timesteps = sampler.ddim_timesteps
    
    print(f"SpotDiffusion: Processing {len(timesteps)} steps")
    print(f"Tile config: time={config.tile_size_time} (freq=full spectrum)")
    print(f"Latent shape: {shape}")
    
    x_t = x_T
    
    # Diffusion loop
    for i, t in enumerate(tqdm(timesteps, desc="SpotDiffusion Steps")):
        # Single denoising step using SpotDiffusion
        x_t = spotdiffusion_denoise_step(
            sampler, x_t, t.item(), conditioning, unconditional_conditioning,
            unconditional_guidance_scale, eta, config, step=i
        )
    
    return x_t

def should_use_spotdiffusion(latent_size, config):
    """Determine if SpotDiffusion should be used based on audio length"""
    # Use SpotDiffusion if audio is longer than a single tile
    return latent_size > config.tile_size_time

# Sliding window VAE decode (reuse from utils.py if available)
def sliding_window_vae_decode(model, latent, window_size=256, overlap_size=64):
    """
    Decode large latent tensors using sliding window approach to manage memory
    """
    device = latent.device
    latent_length = latent.shape[2]  # Time dimension
    
    if latent_length <= window_size:
        # Small enough to decode directly
        with torch.no_grad():
            mel = model.decode_first_stage(latent)
        return mel
    
    print(f"🪟 Using sliding window VAE decode: latent_length={latent_length}, window_size={window_size}")
    
    # Calculate windows
    advance_size = window_size - overlap_size
    num_windows = (latent_length - overlap_size + advance_size - 1) // advance_size
    
    decoded_segments = []
    
    for i in range(num_windows):
        start = i * advance_size
        end = min(start + window_size, latent_length)
        
        print(f"   Window {i+1}/{num_windows}: latent frames {start}-{end}")
        
        # Extract window
        window_latent = latent[:, :, start:end, :]
        
        # Decode window
        with torch.no_grad():
            window_mel = model.decode_first_stage(window_latent)
        
        # Handle overlap blending
        if i > 0 and start < end:
            # Calculate overlap region in mel space
            mel_overlap_size = overlap_size * (window_mel.shape[2] // window_latent.shape[2])
            
            if len(decoded_segments) > 0 and mel_overlap_size > 0:
                # Blend overlapping region
                prev_segment = decoded_segments[-1]
                prev_end = prev_segment.shape[2]
                curr_start = 0
                
                overlap_len = min(mel_overlap_size, prev_end, window_mel.shape[2])
                
                if overlap_len > 0:
                    # Linear blending weights
                    blend_weights = torch.linspace(0, 1, overlap_len).to(device)
                    blend_weights = blend_weights.view(1, 1, -1, 1)
                    
                    # Apply blending
                    prev_segment[:, :, -overlap_len:, :] = (
                        prev_segment[:, :, -overlap_len:, :] * (1 - blend_weights) +
                        window_mel[:, :, :overlap_len, :] * blend_weights
                    )
                    
                    # Add non-overlapping part
                    if window_mel.shape[2] > overlap_len:
                        decoded_segments.append(window_mel[:, :, overlap_len:, :])
                else:
                    decoded_segments.append(window_mel)
            else:
                decoded_segments.append(window_mel)
        else:
            decoded_segments.append(window_mel)
    
    # Concatenate all segments
    if len(decoded_segments) > 1:
        full_mel = torch.cat(decoded_segments, dim=2)
    else:
        full_mel = decoded_segments[0]
    
    print(f"✅ VAE decode complete: {latent.shape} -> {full_mel.shape}")
    return full_mel

def plot_latent_spectrogram(latent_tensor, title="SpotDiffusion Latent"):
    """Plot latent tensor as spectrogram for debugging"""
    import matplotlib.pyplot as plt
    
    if len(latent_tensor.shape) == 4:
        spec = latent_tensor[0, 0].cpu().numpy()
    else:
        spec = latent_tensor.cpu().numpy()
    
    plt.figure(figsize=(12, 4))
    plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()

# Main inference function
def spotdiffusion_text_to_audio(model, prompt, duration=10.0, ddim_steps=200, 
                               unconditional_guidance_scale=3.0, ddim_eta=0.1,
                               config=None):
    """
    Generate audio using SpotDiffusion
    
    Args:
        model: AudioLDM model
        prompt: Text prompt
        duration: Audio duration in seconds
        ddim_steps: Number of diffusion steps
        unconditional_guidance_scale: CFG scale
        ddim_eta: DDIM eta parameter
        config: SpotDiffusionConfig (if None, uses default)
    
    Returns:
        waveform: Generated audio waveform as numpy array
    """
    
    if config is None:
        config = SpotDiffusionConfig()
    
    # Create embeddings
    model.cond_stage_model.embed_mode = "text"
    conditioning = model.get_learned_conditioning([prompt])
    unconditional_conditioning = model.get_learned_conditioning([""])
    
    # Setup sampling
    sampler = DDIMSampler(model)
    latent_size = duration_to_latent_t_size(duration)
    shape = [1, model.channels, latent_size, model.latent_f_size]
    
    print(f"🎯 Generating '{prompt}' ({duration}s)")
    print(f"📐 Latent shape: {shape}")
    
    # Check if we should use SpotDiffusion
    if should_use_spotdiffusion(latent_size, config):
        print(f"🔄 Using SpotDiffusion (latent_size={latent_size} > tile_size={config.tile_size_time})")
        
        # Generate with SpotDiffusion
        start_time = time.time()
        with torch.no_grad():
            samples = spotdiffusion_sample(
                sampler=sampler,
                shape=shape,
                conditioning=conditioning,
                unconditional_conditioning=unconditional_conditioning,
                unconditional_guidance_scale=unconditional_guidance_scale,
                eta=ddim_eta,
                S=ddim_steps,
                config=config
            )
        diffusion_time = time.time() - start_time
    else:
        print(f"🔄 Using regular DDIM (latent_size={latent_size} <= tile_size={config.tile_size_time})")
        
        # Use regular DDIM for short audio
        start_time = time.time()
        with torch.no_grad():
            samples, _ = sampler.sample(
                S=ddim_steps,
                conditioning=conditioning,
                batch_size=1,
                shape=shape[1:],
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                eta=ddim_eta
            )
        diffusion_time = time.time() - start_time
    
    print(f"⏱️  Diffusion time: {diffusion_time:.1f}s")
    
    # Decode with sliding window VAE
    print(f"🪟 VAE decoding...")
    vae_start = time.time()
    mel_spectrogram = sliding_window_vae_decode(model, samples, window_size=256, overlap_size=64)
    vae_time = time.time() - vae_start
    print(f"⏱️  VAE time: {vae_time:.1f}s")
    
    # Generate waveform
    print(f"🔊 Vocoder...")
    vocoder_start = time.time()
    waveform = model.mel_spectrogram_to_waveform(mel_spectrogram)
    if isinstance(waveform, torch.Tensor):
        if waveform.dim() > 1:
            waveform = waveform[0]
        waveform = waveform.squeeze().cpu().numpy()
    else:
        if waveform.ndim > 1:
            waveform = waveform[0]
        waveform = waveform.squeeze()
    
    vocoder_time = time.time() - vocoder_start
    print(f"⏱️  Vocoder time: {vocoder_time:.1f}s")
    
    total_time = diffusion_time + vae_time + vocoder_time
    print(f"⏱️  Total time: {total_time:.1f}s")
    
    return waveform
