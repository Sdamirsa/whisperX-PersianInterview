# Persian Interview Transcription Tool

An automated transcription system for Persian (Farsi) interview recordings with speaker diarization using WhisperX and OpenAI's Whisper large-v3 model.

## What's Enhanced in This WhisperX Implementation

This project extends the original WhisperX with several key improvements:

- 🎯 **Persian-Optimized Pipeline**: Pre-configured for Farsi language with large-v3 model
- 🐛 **Advanced Debug Mode**: Comprehensive logging, UTF-8 validation, and error tracking
- 📊 **Rich Metadata**: Complete processing information saved in all output files
- 🔄 **Resume Functionality**: Smart batch processing with automatic progress tracking  
- ⚡ **Performance Optimization**: ONNXRuntime-GPU setup for 3-5x faster diarization
- 🎨 **Rich Terminal UI**: Colored progress bars, status tables, and error visualization
- 🔐 **Smart Authentication**: HuggingFace token validation and model access testing
- 📁 **Organized Output**: Timestamped folders with stage-specific files (`_S1`, `_S2`, `_S3`)
- 🛡️ **Error Recovery**: Continues processing if individual stages fail
- 🔍 **Character Validation**: Detects and reports encoding artifacts in transcriptions

## Features

- **Persian Language Support**: Optimized for Farsi transcription with embedded English word handling
- **Speaker Diarization**: Automatically identifies and labels different speakers using pyannote Speaker-Diarization-3.1
- **High Accuracy**: Uses Whisper large-v3 model for improved transcription quality
- **Robust Processing**: Saves intermediate outputs with graceful error handling
- **macOS Optimized**: Configured for Apple Silicon (M3 Max) performance

## Quick Start Guide for Non-Technical Users

**Choose your operating system:**

<details>
<summary><strong>🍎 macOS Setup (Click to expand)</strong></summary>

### Step 1: Install Prerequisites
1. **Open Terminal** (search "Terminal" in Spotlight or find in Applications → Utilities)
2. **Install Homebrew** (package manager):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
3. **Install uv** (Python package manager):
   ```bash
   brew install uv
   ```

### Step 2: Get the Code
1. **Download the project**:
   ```bash
   git clone https://github.com/yourusername/whisperX-PersianInterview.git
   cd whisperX-PersianInterview
   ```

### Step 3: Set Up HuggingFace Account
1. **Create HuggingFace account** at [huggingface.co](https://huggingface.co/join)
2. **Get your access token**:
   - Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
   - Click "New token"
   - Name it "Persian Interview Tool"
   - Select "Read" permissions
   - Copy the token (starts with `hf_`)

### Step 4: Accept Model Permissions
**IMPORTANT**: You must accept these model agreements or the tool won't work:
1. Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. Click "Agree and access repository"
3. Fill out the form with your information
4. Wait for approval (usually instant)

### Step 5: Configure the Tool
1. **Run the tool once** (it will create the `.env` file automatically):
   ```bash
   uv run python Run_whisperx_PersianInterview.py --help
   ```
2. **Add your HuggingFace token**:
   - Open the `.env` file that was created
   - Replace `your_hf_token_here` with your actual token
   - Save the file

### Step 6: Optimize Performance (Optional but Highly Recommended)
**Run this once for significantly faster diarization:**
```bash
uv run python optimize_onnx_for_diarization.py
```
⚡ This can improve diarization speed by 3-5x by optimizing ONNXRuntime for GPU acceleration.
💡 **Note**: After running this optimization, diarization performance will be significantly improved.

### Step 7: Test with Sample Audio
```bash
uv run python Run_whisperx_PersianInterview.py data/test-voice/
```

</details>

<details>
<summary><strong>🪟 Windows Setup (Click to expand)</strong></summary>

### Step 1: Install Prerequisites
1. **Install Python**:
   - Download Python 3.10+ from [python.org](https://www.python.org/downloads/)
   - ⚠️ **IMPORTANT**: Check "Add Python to PATH" during installation
2. **Install Git**:
   - Download from [git-scm.com](https://git-scm.com/download/win)
   - Use default settings during installation
3. **Install uv**:
   - Open **Command Prompt** (search "cmd" in Start menu)
   - Run: `pip install uv`

### Step 2: Get the Code
1. **Open Command Prompt**
2. **Download the project**:
   ```cmd
   git clone https://github.com/yourusername/whisperX-PersianInterview.git
   cd whisperX-PersianInterview
   ```

### Step 3: Set Up HuggingFace Account
1. **Create HuggingFace account** at [huggingface.co](https://huggingface.co/join)
2. **Get your access token**:
   - Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
   - Click "New token"
   - Name it "Persian Interview Tool"
   - Select "Read" permissions
   - Copy the token (starts with `hf_`)

### Step 4: Accept Model Permissions
**IMPORTANT**: You must accept these model agreements or the tool won't work:
1. Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. Click "Agree and access repository"
3. Fill out the form with your information
4. Wait for approval (usually instant)

### Step 5: Configure the Tool
1. **Run the tool once** (it will create the `.env` file automatically):
   ```cmd
   uv run python Run_whisperx_PersianInterview.py --help
   ```
2. **Add your HuggingFace token**:
   - Open the `.env` file with Notepad
   - Replace `your_hf_token_here` with your actual token
   - Save the file

### Step 6: Optimize Performance (Optional but Highly Recommended)
**Run this once for significantly faster diarization:**
```cmd
uv run python optimize_onnx_for_diarization.py
```
⚡ This can improve diarization speed by 3-5x by optimizing ONNXRuntime for GPU acceleration.
💡 **Note**: After running this optimization, diarization performance will be significantly improved.

### Step 7: Test with Sample Audio
```cmd
uv run python Run_whisperx_PersianInterview.py data\test-voice\
```

</details>

## Prerequisites

- macOS (tested on M3 Max) or Windows 10+
- HuggingFace account with model access
- Internet connection for initial setup

## Usage

### Basic Commands

**Single file:**
```bash
uv run python Run_whisperx_PersianInterview.py path/to/interview.mp3
```

**Batch process folder:**
```bash
uv run python Run_whisperx_PersianInterview.py path/to/folder/
```

**Resume interrupted processing:**
```bash
uv run python Run_whisperx_PersianInterview.py path/to/folder/ --resume
```

### Advanced Options

```bash
uv run python Run_whisperx_PersianInterview.py path/to/folder/ \
    --min_speakers 2 \
    --max_speakers 4 \
    --batch_size 2 \
    --no_align
```

### Command Line Parameters

- `--min_speakers`: Minimum number of expected speakers (default: 2)
- `--max_speakers`: Maximum number of expected speakers (default: 5)
- `--output_dir`: Directory for saving output files (default: ./output)
- `--batch_size`: Batch size for processing (default: 4, reduce if low on memory)

## Output Files

The tool generates multiple output files for each processed interview:

1. **transcription_[timestamp].json**: Initial transcription without speaker labels
2. **diarization_[timestamp].json**: Speaker diarization segments
3. **wav2vec2_[timestamp].json**: Word-level alignment data
4. **final_[timestamp].json**: Complete transcription with speaker labels
5. **interview_[timestamp].txt**: Human-readable transcript with speaker labels
6. **interview_[timestamp].srt**: Subtitle file with timestamps

## Project Structure

```
whisperX-PersianInterview/
├── README.md                         # This file
├── README-whisperx.md               # Original WhisperX documentation  
├── README-whisperx-code.md          # WhisperX code architecture guide
├── .claude                          # Strategic guide for development
├── .env                             # HuggingFace token (auto-created)
├── Run_whisperx_PersianInterview.py # Main transcription script
├── optimize_onnx_for_diarization.py # Performance optimization script
├── pyproject.toml                   # UV project configuration
├── data/
│   ├── test-voice/                  # Sample audio files
│   └── outputs/                     # Generated transcriptions
└── whisperx/                        # WhisperX source code
```

## Technical Details

### Model Configuration
- **ASR Model**: Whisper large-v3 (latest version)
- **Diarization**: pyannote Speaker-Diarization-3.1
- **Alignment**: wav2vec2 (auto-selected for Persian)
- **Compute Type**: int8 (optimized for Apple Silicon)

### Processing Pipeline
1. Audio preprocessing with VAD (Voice Activity Detection)
2. Transcription using Whisper large-v3
3. Speaker diarization with pyannote
4. Word-level alignment with wav2vec2
5. Output generation with error handling

## Troubleshooting

### Common Issues

1. **"Repository not found" or "Access denied"**:
   - ⚠️ **Most common issue**: You haven't accepted the model agreement
   - Go to [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - Click "Agree and access repository" and fill out the form
   - Wait a few minutes for approval

2. **"No HuggingFace token found"**:
   - Run the script once to auto-create the `.env` file
   - Edit the `.env` file and replace `your_hf_token_here` with your actual token
   - Make sure your token starts with `hf_`

3. **Out of Memory**: Reduce batch size using `--batch_size 2`

4. **Slow Diarization**: Run the optimization script first:
   ```bash
   uv run python optimize_onnx_for_diarization.py
   ```

6. **Missing Speakers**: Adjust `--min_speakers` and `--max_speakers` parameters

7. **Python/uv not found**: 
   - **macOS**: Install Homebrew first, then `brew install uv`
   - **Windows**: Install Python with "Add to PATH" checked, then `pip install uv`

### Performance Tips
- Use shorter audio segments (< 30 minutes) for better performance
- Pre-process audio to remove silence and noise
- Ensure clear audio quality for best results

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project uses WhisperX and is subject to its licensing terms. See [WhisperX License](https://github.com/m-bain/whisperX/blob/master/LICENSE) for details.

## Acknowledgments

- Built on [WhisperX](https://github.com/m-bain/whisperX) by Max Bain
- Uses [OpenAI Whisper](https://github.com/openai/whisper) models
- Speaker diarization by [pyannote-audio](https://github.com/pyannote/pyannote-audio)

## Contact

For questions or support, please open an issue on GitHub.