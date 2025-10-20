# Voice Cleaner

A Python module for cleaning and preprocessing audio files for speech recognition and other audio processing tasks.

## Features

- **High-pass filtering**: Removes low-frequency noise and rumble
- **Noise reduction**: Stationary noise removal using spectral gating
- **De-essing**: Gentle reduction of harsh sibilant sounds
- **Hum removal**: Optional notch filtering for mains hum (50/60 Hz)
- **LUFS normalization**: Consistent loudness levels across files
- **Format conversion**: Outputs 16-bit PCM WAV files at 16 kHz mono (Whisper-optimized)

## Installation

This module uses its own isolated UV environment to avoid conflicts with the main project.

```bash
cd my_code/voice_cleaner
uv sync
```

## Usage

### Basic Usage

Process a single audio file:
```bash
uv run voice_cleaner.py input_audio.mp3
```

Process all audio files in a directory:
```bash
uv run voice_cleaner.py /path/to/audio/directory
```

### Advanced Options

```bash
uv run voice_cleaner.py input.wav --sr 22050 --hpf 100 --lufs -18 --hum
```

#### Command Line Options

- `input`: Path to an audio file or directory to process recursively
- `--sr`: Target sample rate (default: 16000 Hz)
- `--hpf`: High-pass filter cutoff frequency in Hz (default: 80 Hz)
- `--no-deess`: Disable the gentle de-esser
- `--hum`: Enable mains-hum notch filtering (disabled by default)
- `--hum-f0`: Hum base frequency, 50 or 60 Hz (default: 50 Hz)
- `--hum-harmonics`: Number of harmonics to also notch (default: 0, only base frequency)
- `--lufs`: Target LUFS for mono audio (default: -16 LUFS)

### Output

Processed files are saved with the suffix `_clean.wav` in the same directory as the input files.

Example:
- Input: `interview_raw.m4a`
- Output: `interview_raw_clean.wav`

## Supported Audio Formats

- WAV, MP3, M4A, AAC, FLAC, OGG, WMA
- WebM, MKV, MP4 (audio tracks)
- AIFF, AIF

## Processing Pipeline

1. **Load and convert to mono**: Any input format ’ mono audio
2. **Resample**: Convert to target sample rate (default 16 kHz)
3. **DC removal**: Remove any DC offset
4. **High-pass filter**: Remove low-frequency noise (default: 80 Hz cutoff)
5. **Optional hum removal**: Notch filter for mains hum (50/60 Hz)
6. **Noise reduction**: Spectral gating for stationary noise
7. **Optional de-essing**: Gentle reduction of harsh sibilants (6-10 kHz)
8. **LUFS normalization**: Consistent loudness (default: -16 LUFS)
9. **Save**: 16-bit PCM WAV format

## Examples

### Process interview recordings with hum removal:
```bash
uv run voice_cleaner.py interviews/ --hum --hum-f0 60 --hum-harmonics 2
```

### Higher quality output for music:
```bash
uv run voice_cleaner.py music.flac --sr 44100 --lufs -14 --no-deess
```

### Aggressive noise cleaning:
```bash
uv run voice_cleaner.py noisy_recording.wav --hpf 120 --lufs -18
```

## Dependencies

All dependencies are managed by UV and isolated from the main project:

- numpy: Numerical computing
- soundfile: Audio I/O
- librosa: Audio analysis and processing
- scipy: Signal processing algorithms
- noisereduce: Spectral noise reduction
- pyloudnorm: LUFS loudness normalization
- tqdm: Progress bars