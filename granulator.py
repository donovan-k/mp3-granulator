import numpy as np
from pydub import AudioSegment
import random
import math

# Load audio file
def load_audio(file_path):
    return AudioSegment.from_file(file_path)

# Convert audio to numpy array
def audio_to_np(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    return samples, audio_segment.frame_rate, audio_segment.channels

# Convert numpy array back to audio segment
def np_to_audio(samples, frame_rate, channels):
    audio_segment = AudioSegment(
        samples.tobytes(), 
        frame_rate=frame_rate, 
        sample_width=samples.dtype.itemsize, 
        channels=channels
    )
    return audio_segment

# Apply crossfade to two overlapping grains of different sizes without trimming the larger one
def crossfade_grains(grain1, grain2, crossfade_duration):
    min_length = min(len(grain1), len(grain2))
    crossfade_samples = int(min_length * crossfade_duration)

    # Convert grains to float32 for processing
    grain1 = grain1.astype(np.float32)
    grain2 = grain2.astype(np.float32)

    # Apply linear crossfade on the overlapping portion
    fade_out = np.linspace(1, 0, crossfade_samples, dtype=np.float32)
    fade_in = np.linspace(0, 1, crossfade_samples, dtype=np.float32)

    # Crossfade only the overlapping part
    grain1[-crossfade_samples:] *= fade_out
    grain2[:crossfade_samples] *= fade_in

    # Sum the overlapping part
    crossfaded_part = grain1[:min_length] + grain2[:min_length]

    # Append the remaining part of the larger grain
    if len(grain1) > len(grain2):
        crossfaded_grain = np.concatenate((crossfaded_part, grain1[min_length:]))
    else:
        crossfaded_grain = np.concatenate((crossfaded_part, grain2[min_length:]))

    # Clip the result to avoid going beyond int16 limits
    crossfaded_grain = np.clip(crossfaded_grain, -32768, 32767)

    return crossfaded_grain.astype(np.int16)

# Function to calculate a sine-modulated grain size
def get_sine_wave_grain_size(min_size, max_size, t, frequency):
    sine_value = 0.5 * (1 + math.sin(2 * math.pi * frequency * t))
    return int(min_size + (max_size - min_size) * sine_value)

# Granular synthesis with random grain selection, dynamic grain size, and controlled overlap
def granular_synthesis(audio_segment, min_grain_size_ms=10, max_grain_size_ms=100, overlap=0.5, crossfade_duration=0.1, frequency=0.01):
    samples, frame_rate, channels = audio_to_np(audio_segment)

    min_grain_size = int(frame_rate * (min_grain_size_ms / 1000.0))  # Min grain size in samples
    max_grain_size = int(frame_rate * (max_grain_size_ms / 1000.0))  # Max grain size in samples

    output = np.zeros(len(samples), dtype=np.float32)  # Output buffer
    audio_length = len(samples)

    prev_grain = None  # Hold the previous grain for crossfading
    amplitude_reduction = 0.5  # Reduce amplitude to avoid clipping
    current_position = 0  # Track position for placing grains

    t = 0  # Time index for sine wave modulation

    while current_position + min_grain_size < audio_length:
        # Get dynamic grain size based on a sine wave
        grain_size = get_sine_wave_grain_size(min_grain_size, max_grain_size, t, frequency)

        # Calculate step size to ensure overlap
        step_size = int(grain_size * (1 - overlap))

        # Randomly select a start point within a sliding window to prevent silence
        window_size = grain_size * 10  # Restrict random grain selection within this range
        start = random.randint(max(0, current_position - window_size), min(audio_length - grain_size, current_position + window_size))
        grain = samples[start:start+grain_size].astype(np.float32)

        # Reverse grain for random pitch shifting (optional)
        if random.random() > 0.5:
            grain = grain[::-1]

        # Apply crossfade between the current and previous grain
        if prev_grain is not None:
            crossfaded_grain = crossfade_grains(prev_grain, grain, crossfade_duration)

            available_space = len(output) - current_position  # Remaining space in the output buffer
            if len(crossfaded_grain) > available_space:
                crossfaded_grain = crossfaded_grain[:available_space]

            output[current_position:current_position+len(crossfaded_grain)] += crossfaded_grain * amplitude_reduction
        else:
            available_space = len(output) - current_position
            if grain_size > available_space:
                grain = grain[:available_space]

            output[current_position:current_position+len(grain)] += grain * amplitude_reduction

        # Update position: ensure grains overlap with the previous one
        current_position += step_size

        prev_grain = grain
        t += 1  # Increment time step for sine modulation

    # Normalize output to avoid clipping
    output = np.clip(output, -32768, 32767)
    output = output.astype(np.int16)

    return np_to_audio(output, frame_rate, channels)

# Chunked processing for large files
def granular_synthesis_in_chunks(file_path, chunk_duration_ms=10000, *args, **kwargs):
    audio_segment = load_audio(file_path)
    total_duration_ms = len(audio_segment)
    output_audio = AudioSegment.silent(duration=total_duration_ms)  # Prepare an empty output buffer

    for chunk_start in range(0, total_duration_ms, chunk_duration_ms):
        chunk_end = min(chunk_start + chunk_duration_ms, total_duration_ms)
        chunk = audio_segment[chunk_start:chunk_end]

        granular_chunk = granular_synthesis(chunk, *args, **kwargs)
        output_audio = output_audio.overlay(granular_chunk, position=chunk_start)

    return output_audio

# Save output to file
def save_audio(audio_segment, file_path):
    audio_segment.export(file_path, format="mp3")

# Example usage
input_file = "ambientjazz.mp3"  # Your input MP3 file
output_file = "ambientjazzgran.mp3"  # Your output MP3 file

granular_audio = granular_synthesis_in_chunks(input_file, chunk_duration_ms=10000, min_grain_size_ms=20, max_grain_size_ms=5000, overlap=0.5, crossfade_duration=0.1, frequency=0.05)
save_audio(granular_audio, output_file)
