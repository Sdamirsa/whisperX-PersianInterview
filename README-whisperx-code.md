# WhisperX Code Architecture

## Core Pipeline
```
Audio → VAD → Whisper → Alignment → Diarization → Output
```

## Module Structure

### 1. `asr.py` - Transcription Engine
- **FasterWhisperModel**: Main ASR class wrapping faster-whisper
- **Key defaults**:
  - `beam_size`: 5
  - `batch_size`: 8
  - `chunk_size`: 30 seconds
  - `no_speech_threshold`: 0.6
  - `compression_ratio_threshold`: 2.4

### 2. `alignment.py` - Forced Alignment
- Uses Wav2Vec2 models for word-level timestamps
- **Language models**:
  - Torchaudio: en, fr, de, es, it
  - HuggingFace: ja, zh, nl, uk, pt, ar, cs, ru, pl, hu, fi, fa, el, tr, da, he, vi, ur, te, hi, ca, ml

### 3. `diarize.py` - Speaker Diarization
- PyAnnote-based speaker separation
- Default model: `pyannote/speaker-diarization-3.1`
- Assigns speaker labels to transcript segments

### 4. `audio.py` - Audio Processing
- **Constants**:
  - Sample rate: 16kHz
  - Chunk length: 30 seconds
  - FFT: 400, Hop: 160

### 5. `vads/` - Voice Activity Detection
- **pyannote.py**: Default VAD (onset=0.5, offset=0.363)
- **silero.py**: Alternative VAD option

## Default Parameters

### ASR Options
```python
{
    "beam_size": 5,
    "best_of": 5,
    "patience": 1,
    "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "condition_on_previous_text": False,
    "without_timestamps": True,
    "suppress_blank": True
}
```

### VAD Options
```python
{
    "chunk_size": 30,
    "vad_onset": 0.500,
    "vad_offset": 0.363
}
```

## Key Functions

### `load_model(model_name, device, compute_type)`
Loads Whisper model with specified precision (float16/int8/float32)

### `transcribe(audio, batch_size, language)`
Main transcription with VAD preprocessing and batched inference

### `align(segments, model, metadata, audio, device)`
Forced alignment for word-level timestamps using Wav2Vec2

### `DiarizationPipeline(use_auth_token, device)`
Speaker diarization pipeline initialization

### `assign_word_speakers(diarize_segments, result)`
Maps speaker labels to transcript segments

## Processing Flow

1. **Audio Loading**: Convert to 16kHz mono WAV
2. **VAD Segmentation**: Detect speech regions
3. **Batch Transcription**: Process VAD chunks through Whisper
4. **Alignment**: Map words to timestamps
5. **Diarization**: Identify speakers
6. **Output**: Generate formatted results

## Output Formats
- JSON: Complete metadata and segments
- SRT/VTT: Subtitles with timestamps
- TXT: Plain text transcript
- TSV: Tab-separated values
- AUD: Audacity labels

## Model Sizes
- tiny (39M), base (74M), small (244M)
- medium (769M), large-v1/v2/v3 (1550M)

## Performance Tips
- Use `int8` compute for CPU inference
- Adjust `batch_size` based on GPU memory
- Enable `model_flush` between stages
- Use smaller models for faster processing