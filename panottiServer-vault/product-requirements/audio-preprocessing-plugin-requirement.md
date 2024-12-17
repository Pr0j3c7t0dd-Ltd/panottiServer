I want to create a plugin using the plugin architecture in the app that will pre-process audio files to remove the background noice from the microphone file.  The background noise is in the system audio file.  The incoming microphone audio file is in the event "microphone_audio_path" and the incoming system audio is in the "system_audio_path".

I want the plugin to listen for the 'recording_ended' even type (see @main.py for event info) to trigger the processing.  The plugin should create it's own table in the sqlite database to manage the processing and state if required.  The table should capture the recording_id from the events table, although the same recording_id may be used to process multiple files.

The plugin config should include a directory to store the cleaned up files in.  The format of the cleaned filename should be in the same format as the input files, e.g. <recording_id>microphone_cleaned.wav.  Example: 20241216140128_2096E65E_microphone_cleaned.wav

The plugin should emit it's own event when the process is complete.  The completed event should pass on the information from the original event, plus the file information for the cleaned file, including the path to the cleaned file.

Ideally the processing should use multi-threading so that processing can happen concurrently.

If either the "microphone_audio_path" and / or the "system_audio_path" is null or empty, then do not do any processing and emit the completion event.

The working code to be implemented is below, no need to change, apart to incorporate into the plugin structure and app logging.  Ensure you have the correct logging in place.

Ensure you add a detailed README.md to the plugin, with any information needed for additional python package requirements.txt

Look at the current example plugin in the /app/plugins/example directory to get context on how to create the plugin.

---
Working Function to impliment:

```
def reduce_noise(mic_file, noise_file, output_file, noise_reduce_factor=0.7):

"""

Reduce background noise from microphone audio using a noise profile.

Parameters:

mic_file (str): Path to microphone recording WAV file

noise_file (str): Path to system recording WAV file (noise profile)

output_file (str): Path to save cleaned audio

noise_reduce_factor (float): Amount of noise reduction (0 to 1)

"""

# Read both audio files

mic_rate, mic_data = wavfile.read(mic_file)

noise_rate, noise_data = wavfile.read(noise_file)

print(f"Mic data shape: {mic_data.shape}, Noise data shape: {noise_data.shape}")

print(f"Mic rate: {mic_rate}, Noise rate: {noise_rate}")

# Convert stereo to mono by averaging channels

if len(mic_data.shape) > 1:

mic_data = np.mean(mic_data, axis=1)

if len(noise_data.shape) > 1:

noise_data = np.mean(noise_data, axis=1)

print(f"After mono conversion - Mic data shape: {mic_data.shape}, Noise data shape: {noise_data.shape}")

# Convert to float32 for processing

mic_data = mic_data.astype(np.float32)

noise_data = noise_data.astype(np.float32)

# Normalize audio

mic_data = mic_data / np.max(np.abs(mic_data))

noise_data = noise_data / np.max(np.abs(noise_data))

# Calculate STFT parameters based on input length

nperseg = 2048 # Use fixed window size now that we have enough samples

noverlap = nperseg // 2 # 50% overlap

print(f"Using nperseg={nperseg}, noverlap={noverlap}")

# Compute noise profile using Short-time Fourier Transform (STFT)

_, _, noise_spectrogram = signal.stft(noise_data,

fs=noise_rate,

nperseg=nperseg,

noverlap=noverlap)

# Calculate noise profile magnitude

noise_profile = np.mean(np.abs(noise_spectrogram), axis=1)

# Compute STFT of microphone audio

f, t, mic_spectrogram = signal.stft(mic_data,

fs=mic_rate,

nperseg=nperseg,

noverlap=noverlap)

# Apply spectral subtraction

mic_mag = np.abs(mic_spectrogram)

mic_phase = np.angle(mic_spectrogram)

# Expand noise profile to match spectrogram shape

noise_profile = noise_profile.reshape(-1, 1)

# Subtract noise profile from magnitude spectrum

cleaned_mag = np.maximum(

mic_mag - noise_profile * noise_reduce_factor,

mic_mag * 0.1 # Spectral floor to prevent complete silence

)

# Reconstruct complex spectrogram

cleaned_spectrogram = cleaned_mag * np.exp(1j * mic_phase)

# Inverse STFT to get cleaned audio

_, cleaned_audio = signal.istft(cleaned_spectrogram,

fs=mic_rate,

nperseg=nperseg,

noverlap=noverlap)

# Normalize output

cleaned_audio = cleaned_audio / np.max(np.abs(cleaned_audio))

# Convert back to int16 for WAV file

cleaned_audio = (cleaned_audio * 32767).astype(np.int16)

# Save cleaned audio

wavfile.write(output_file, mic_rate, cleaned_audio)

return cleaned_audio

  

# Example usage

mic_file = '/Users/todd/Library/Containers/com.pr0j3c7t0dd.PanottiAudio/Data/Documents/20241216140128_2096E65E_microphone.wav'

noise_file = '/Users/todd/Library/Containers/com.pr0j3c7t0dd.PanottiAudio/Data/Documents/20241216140128_2096E65E_system_audio.wav'

output_file = 'cleaned.wav'

noise_reduce_factor=.7 # Adjust this value between 0 and 1 to control noise reduction strength

cleaned_audio = reduce_noise(mic_file, noise_file, output_file, noise_reduce_factor)
```


