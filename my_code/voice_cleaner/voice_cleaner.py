#!/usr/bin/env python3
# -------------------------------------------
# Required packages (add to your uv project):
#   numpy
#   soundfile
#   librosa
#   scipy
#   noisereduce
#   pyloudnorm
#   tqdm            (optional, for nice progress bars)
#
# With uv:
#   uv add numpy soundfile librosa scipy noisereduce pyloudnorm tqdm
# -------------------------------------------

import argparse
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfiltfilt, iirnotch, sosfilt
import noisereduce as nr
import pyloudnorm as pyln

try:
    from tqdm import tqdm
except Exception:
    # Fallback if tqdm isn't installed
    def tqdm(x, **kwargs):
        return x


# ---------------------------
# DSP helpers
# ---------------------------

def highpass(y: np.ndarray, sr: int, fc: float = 80.0, order: int = 2) -> np.ndarray:
    """Butterworth high-pass (zero-phase)."""
    sos = butter(order, fc / (sr / 2), btype="highpass", output="sos")
    return sosfiltfilt(sos, y)

def notch_hum(y: np.ndarray, sr: int, f0: float = 50.0, Q: float = 30.0, harmonics: int = 0) -> np.ndarray:
    """Notch out mains hum at f0 and up to N harmonics. Disabled by default."""
    y_out = y.copy()
    for k in range(1, max(1, harmonics) + 1):
        w0 = (f0 * k) / (sr / 2)
        b, a = iirnotch(w0, Q)
        # Convert to SOS-like single section for sosfilt
        sos = np.array([[b[0], b[1], b[2], 1.0, a[1], a[2]]], dtype=float)
        y_out = sosfilt(sos, y_out)
    return y_out

def gentle_deess(y: np.ndarray, sr: int, fmin: float = 6000.0, fmax: float = 10000.0,
                 max_reduction_db: float = 6.0) -> np.ndarray:
    """
    Very simple, robust de-esser:
    - STFT
    - detect frames with unusually high energy in [fmin, fmax]
    - attenuate those frames in that band up to max_reduction_db
    """
    n_fft = 1024
    hop = 256
    win = 1024

    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win, window="hann", center=True)
    mag, ph = np.abs(S), np.angle(S)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return y

    band_energy = mag[band, :].mean(axis=0)
    # Threshold: 95th percentile of high-band energy
    th = np.percentile(band_energy, 95)
    # Per-frame gain (cap reduction between 0 dB and -max_reduction_db)
    # If energy >> threshold, reduce more; if <= threshold, do nothing.
    # Gain = min(1.0, th / energy) but clipped not to exceed max_reduction_db
    safe_energy = band_energy + 1e-9
    raw_gain = np.clip(th / safe_energy, 10 ** (-max_reduction_db / 20.0), 1.0)

    # Apply only to the high band
    mag[band, :] *= raw_gain[np.newaxis, :]

    S_fixed = mag * np.exp(1j * ph)
    y_fixed = librosa.istft(S_fixed, hop_length=hop, win_length=win, window="hann", center=True)
    # match length
    if len(y_fixed) != len(y):
        y_fixed = librosa.util.fix_length(y_fixed, size=len(y))
    return y_fixed

def lufs_normalize_mono(y: np.ndarray, sr: int, target_lufs: float = -16.0) -> np.ndarray:
    """
    Normalize integrated loudness to target LUFS.
    For safety, apply a simple peak guard after normalization.
    """
    meter = pyln.Meter(sr)  # mono
    loudness = meter.integrated_loudness(y)
    y_norm = pyln.normalize.loudness(y, loudness, target_lufs)

    # Peak safety (simple limiter-ish clamp)
    peak = np.max(np.abs(y_norm)) + 1e-12
    if peak > 0.999:
        y_norm = y_norm / peak * 0.999
    return y_norm


# ---------------------------
# Core processing
# ---------------------------

def process_file(
    in_path: Path,
    target_sr: int = 16000,
    hpf_fc: float = 80.0,
    apply_deess: bool = True,
    apply_hum: bool = False,
    hum_f0: float = 50.0,
    hum_harmonics: int = 0,
    target_lufs: float = -16.0,
) -> Path:
    """
    Load -> mono -> resample -> DC remove -> HPF -> (optional hum notch)
         -> noise reduce -> (optional de-ess) -> LUFS normalize -> save WAV (PCM_16, 16k mono)
    """
    # Load any audio type; librosa falls back to audioread if needed
    y, sr = librosa.load(str(in_path), sr=None, mono=True)

    # Resample to target_sr (Whisper-friendly)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
        sr = target_sr

    # Remove DC offset & leave headroom
    y = y - np.mean(y)

    # High-pass: remove rumble/plosives
    y = highpass(y, sr, fc=hpf_fc, order=2)

    # Optional mains hum notch (off by default)
    if apply_hum:
        y = notch_hum(y, sr, f0=hum_f0, Q=30.0, harmonics=hum_harmonics)

    # Broadband denoise (stationary = True is robust for single-speaker rooms)
    y = nr.reduce_noise(y=y, sr=sr, stationary=True)

    # Gentle de-ess to tame harsh "Z/S" spikes
    if apply_deess:
        y = gentle_deess(y, sr, fmin=6000.0, fmax=10000.0, max_reduction_db=6.0)

    # Normalize to target LUFS (stabilizes level across near/far)
    y = lufs_normalize_mono(y, sr, target_lufs=target_lufs)

    # Write as 16-bit PCM WAV, 16 kHz mono (ideal for most ASR next-steps)
    out_path = in_path.with_name(f"{in_path.stem}_clean.wav")
    sf.write(str(out_path), y, sr, subtype="PCM_16")
    return out_path


# ---------------------------
# CLI
# ---------------------------

def find_audio_files(path: Path) -> list[Path]:
    exts = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".wma", ".webm", ".mkv", ".mp4", ".aiff", ".aif"}
    if path.is_file():
        return [path]
    files = []
    for p in path.rglob("*"):
        if p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)

def clean_audio_batch(
    input_path: str | Path, 
    output_dir: str | Path = None,
    target_sr: int = 16000,
    hpf_fc: float = 80.0,
    apply_deess: bool = True,
    apply_hum: bool = False,
    hum_f0: float = 50.0,
    hum_harmonics: int = 0,
    target_lufs: float = -16.0,
    verbose: bool = True
) -> list[Path]:
    """
    Clean audio files and optionally save to a different directory.
    
    Args:
        input_path: Path to file or directory
        output_dir: Output directory (default: same as input with '_clean' suffix)
        **kwargs: Processing parameters
        
    Returns:
        List of cleaned audio file paths
    """
    input_path = Path(input_path)
    audio_files = find_audio_files(input_path)
    
    if not audio_files:
        if verbose:
            print("No audio files found.", file=sys.stderr)
        return []
    
    # Determine output directory
    if output_dir is None:
        if input_path.is_file():
            output_dir = input_path.parent / f"{input_path.stem}_clean"
        else:
            output_dir = input_path.parent / f"{input_path.name}_clean"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    cleaned_files = []
    
    if verbose:
        print(f"Found {len(audio_files)} file(s). Processing to {output_dir}...")
        iterator = tqdm(audio_files, desc="Cleaning", unit="file")
    else:
        iterator = audio_files
    
    for audio_file in iterator:
        try:
            # Process the file
            y, sr = librosa.load(str(audio_file), sr=None, mono=True)
            
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
                sr = target_sr
            
            y = y - np.mean(y)
            y = highpass(y, sr, fc=hpf_fc, order=2)
            
            if apply_hum:
                y = notch_hum(y, sr, f0=hum_f0, Q=30.0, harmonics=hum_harmonics)
            
            y = nr.reduce_noise(y=y, sr=sr, stationary=True)
            
            if apply_deess:
                y = gentle_deess(y, sr, fmin=6000.0, fmax=10000.0, max_reduction_db=6.0)
            
            y = lufs_normalize_mono(y, sr, target_lufs=target_lufs)
            
            # Save to output directory with original name but .wav extension
            output_file = output_dir / f"{audio_file.stem}_clean.wav"
            sf.write(str(output_file), y, sr, subtype="PCM_16")
            cleaned_files.append(output_file)
            
            if verbose:
                print(f"OK  → {output_file}")
                
        except Exception as e:
            if verbose:
                print(f"ERR → {audio_file} : {e}", file=sys.stderr)
    
    return cleaned_files

def main():
    parser = argparse.ArgumentParser(
        description="Clean voice files (HPF → denoise → de-ess → LUFS) and save *_clean.wav (16 kHz mono)."
    )
    parser.add_argument("input", type=str, help="Path to an audio file OR a directory to process recursively.")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate (default: 16000).")
    parser.add_argument("--hpf", type=float, default=80.0, help="High-pass cutoff in Hz (default: 80).")
    parser.add_argument("--no-deess", action="store_true", help="Disable the gentle de-esser.")
    parser.add_argument("--hum", action="store_true", help="Enable mains-hum notch filtering (off by default).")
    parser.add_argument("--hum-f0", type=float, default=50.0, help="Hum base frequency 50 or 60 Hz.")
    parser.add_argument("--hum-harmonics", type=int, default=0, help="Number of harmonics to also notch (0=only base).")
    parser.add_argument("--lufs", type=float, default=-16.0, help="Target LUFS for mono (default: -16).")

    args = parser.parse_args()

    in_path = Path(args.input)
    inputs = find_audio_files(in_path)
    if not inputs:
        print("No audio files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(inputs)} file(s). Processing…")
    for f in tqdm(inputs, desc="Cleaning", unit="file"):
        try:
            out = process_file(
                f,
                target_sr=args.sr,
                hpf_fc=args.hpf,
                apply_deess=not args.no_deess,
                apply_hum=args.hum,
                hum_f0=args.hum_f0,
                hum_harmonics=args.hum_harmonics,
                target_lufs=args.lufs,
            )
            # Print per-file result for easy scripting
            print(f"OK  → {out}")
        except Exception as e:
            print(f"ERR → {f} : {e}", file=sys.stderr)

if __name__ == "__main__":
    main()