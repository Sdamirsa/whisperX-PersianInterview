#!/usr/bin/env python3
"""
Persian Interview Transcription Tool
Transcribes Persian interview recordings with speaker diarization using WhisperX
"""

import os
import sys
import json
import argparse
import gc
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

import whisperx
from whisperx.diarize import DiarizationPipeline

# Initialize Rich console
console = Console()

# Load environment variables
load_dotenv()

def check_hf_token():
    """Check if HF_TOKEN exists in environment, create .env template if not."""
    hf_token = os.getenv('HF_TOKEN') or os.getenv('hf_token')
    env_file = Path('.env')
    
    if not hf_token:
        # Create or update .env file with template
        if not env_file.exists():
            with open(env_file, 'w') as f:
                f.write('# HuggingFace Access Token\n')
                f.write('# Generate your token at: https://huggingface.co/settings/tokens\n')
                f.write('# Accept model agreements at:\n')
                f.write('# - https://huggingface.co/pyannote/segmentation-3.0\n')
                f.write('# - https://huggingface.co/pyannote/speaker-diarization-3.1\n')
                f.write('HF_TOKEN=your_hf_token_here\n')
            
            console.print(Panel.fit(
                "[yellow]⚠️  No HuggingFace token found![/yellow]\n\n"
                f"Created .env file at: [cyan]{env_file.absolute()}[/cyan]\n\n"
                "[bold]Steps to enable speaker diarization:[/bold]\n"
                "1. Edit the .env file and add your HuggingFace token\n"
                "2. Generate token at: [link]https://huggingface.co/settings/tokens[/link]\n"
                "3. Accept required model agreements (see links in .env)",
                title="[red]HuggingFace Token Required[/red]",
                border_style="yellow"
            ))
        else:
            # Check if HF_TOKEN line exists in .env
            with open(env_file, 'r') as f:
                content = f.read()
            
            if 'HF_TOKEN' not in content and 'hf_token' not in content:
                # Append HF_TOKEN template to existing .env
                with open(env_file, 'a') as f:
                    f.write('\n# HuggingFace Access Token\n')
                    f.write('# Generate your token at: https://huggingface.co/settings/tokens\n')
                    f.write('# Accept model agreements at:\n')
                    f.write('# - https://huggingface.co/pyannote/segmentation-3.0\n')
                    f.write('# - https://huggingface.co/pyannote/speaker-diarization-3.1\n')
                    f.write('HF_TOKEN=your_hf_token_here\n')
                
                console.print("[yellow]⚠️  Added HF_TOKEN template to existing .env file[/yellow]")
                console.print("Please edit the .env file and add your token")
            else:
                console.print("[yellow]⚠️  HF_TOKEN found in .env but not loaded properly[/yellow]")
        return None
    
    return hf_token

# Check HF token on module load
HF_TOKEN_STATUS = check_hf_token()

def extract_function_defaults(func):
    """Extract default parameter values from a function."""
    import inspect
    sig = inspect.signature(func)
    defaults = {}
    for param_name, param in sig.parameters.items():
        if param.default is not inspect.Parameter.empty:
            defaults[param_name] = param.default
    return defaults

def get_whisperx_defaults():
    """Dynamically extract WhisperX default parameters from actual implementations."""
    import inspect
    from whisperx import asr, alignment, diarize, audio, load_model, load_align_model
    from whisperx.diarize import DiarizationPipeline
    
    defaults = {}
    
    # Extract load_model defaults
    try:
        defaults['load_model'] = extract_function_defaults(load_model)
    except:
        defaults['load_model'] = {}
    
    # Extract FasterWhisperPipeline transcribe method defaults
    try:
        # Get a dummy model to inspect transcribe defaults
        import torch
        if torch.cuda.is_available():
            dummy_device = "cuda"
        else:
            dummy_device = "cpu"
        
        # We'll inspect the transcribe method signature
        from whisperx.asr import FasterWhisperPipeline
        defaults['transcribe'] = extract_function_defaults(FasterWhisperPipeline.transcribe)
        
        # Also get default_asr_options from the module
        if hasattr(asr, 'default_asr_options'):
            defaults['asr_options'] = asr.default_asr_options
        
        # Get VAD defaults
        if hasattr(asr, 'default_vad_options'):
            defaults['vad_options'] = asr.default_vad_options
            
    except Exception as e:
        console.print(f"[yellow]Note: Could not extract some ASR defaults: {e}[/yellow]")
        defaults['transcribe'] = {}
    
    # Extract alignment defaults
    try:
        defaults['load_align_model'] = extract_function_defaults(load_align_model)
        defaults['align'] = extract_function_defaults(alignment.align)
    except:
        defaults['align'] = {}
    
    # Extract diarization defaults
    try:
        defaults['diarization'] = extract_function_defaults(DiarizationPipeline.__init__)
        defaults['assign_word_speakers'] = extract_function_defaults(diarize.assign_word_speakers)
    except:
        defaults['diarization'] = {}
    
    # Extract audio processing defaults
    try:
        defaults['load_audio'] = extract_function_defaults(audio.load_audio)
        # Get audio constants
        defaults['audio_constants'] = {
            'SAMPLE_RATE': getattr(audio, 'SAMPLE_RATE', 16000),
            'N_FFT': getattr(audio, 'N_FFT', 400),
            'HOP_LENGTH': getattr(audio, 'HOP_LENGTH', 160),
            'CHUNK_LENGTH': getattr(audio, 'CHUNK_LENGTH', 30),
            'N_SAMPLES': getattr(audio, 'N_SAMPLES', 480000),
            'N_FRAMES': getattr(audio, 'N_FRAMES', 3000)
        }
    except:
        defaults['audio_constants'] = {}
    
    return defaults

def get_runtime_config(args, model_name="large-v3"):
    """Get the actual runtime configuration being used."""
    import platform
    import torch
    import whisperx
    import faster_whisper
    import pyannote.audio
    
    # Get all defaults
    whisperx_defaults = get_whisperx_defaults()
    
    config = {
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "machine": platform.machine(),
        },
        "package_versions": {
            "torch": torch.__version__,
            "whisperx": getattr(whisperx, "__version__", "3.7.4"),
            "faster_whisper": faster_whisper.__version__,
            "pyannote.audio": pyannote.audio.__version__,
        },
        "cuda_info": {
            "available": torch.cuda.is_available(),
            "version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "model_config": {
            "model_name": model_name,
            "device": args.device,
            "compute_type": args.compute_type,
            "batch_size": args.batch_size,
            "language": "fa",
        },
        "processing_config": {
            "diarize": args.diarize,
            "min_speakers": args.min_speakers,
            "max_speakers": args.max_speakers,
            "no_align": args.no_align,
            "model_flush": args.model_flush,
        },
        "defaults_used": whisperx_defaults,
        "user_overrides": {
            "batch_size": args.batch_size if args.batch_size != 4 else None,
            "compute_type": args.compute_type if args.compute_type != "int8" else None,
            "device": args.device if args.device != "cpu" else None,
        }
    }
    
    # Remove None values from user_overrides
    config["user_overrides"] = {k: v for k, v in config["user_overrides"].items() if v is not None}
    
    return config

def transcribe_audio(audio_path, args, output_dir):
    """
    Transcribe a single audio file with speaker diarization.
    
    Args:
        audio_path: Path to audio file
        args: Command line arguments
        output_dir: Specific output directory for this session
    
    Returns:
        dict: Results from each stage or None if critical error
    """
    device = args.device
    compute_type = args.compute_type
    batch_size = args.batch_size
    hf_token = os.getenv('HF_TOKEN') or os.getenv('hf_token')
    
    if not hf_token and args.diarize:
        print("Warning: HF_TOKEN not found in .env file. Diarization will be skipped.")
        
    audio_name = Path(audio_path).stem
    output_base = Path(output_dir) / audio_name
    
    # Get comprehensive metadata
    runtime_config = get_runtime_config(args, model_name="large-v3")
    
    results = {
        'metadata': runtime_config,
        'audio_file': {
            'path': str(audio_path),
            'name': Path(audio_path).name,
            'size_bytes': Path(audio_path).stat().st_size,
        },
        'processing': {
            'started_at': datetime.now().isoformat(),
            'stages_completed': [],
            'errors': []
        },
        'stages': {}
    }
    
    console.print(f"\n[bold cyan]📂 Processing:[/bold cyan] {Path(audio_path).name}")
    console.rule(style="cyan")
    
    try:
        # Stage 1: Load and transcribe with Whisper
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            task = progress.add_task("[yellow]🎯 Stage 1: Transcription[/yellow]", total=3)
            
            # Load model
            progress.update(task, description="[yellow]Loading Whisper large-v3...[/yellow]", advance=1)
            model = whisperx.load_model(
                "large-v3", 
                device, 
                compute_type=compute_type,
                language="fa"  # Set Persian as default
            )
            
            # Load audio
            progress.update(task, description="[yellow]Loading audio file...[/yellow]", advance=1)
            audio = whisperx.load_audio(audio_path)
            
            # Transcribe
            progress.update(task, description="[yellow]Transcribing Persian audio...[/yellow]", advance=1)
            result = model.transcribe(
                audio, 
                batch_size=batch_size,
                language="fa"  # Persian
            )
        
        # Save Stage 1 results with metadata
        stage1_data = {
            'metadata': runtime_config,
            'stage': 'transcription',
            'completed_at': datetime.now().isoformat(),
            'model_used': 'large-v3',
            'language_detected': result.get('language', 'fa'),
            'segments': result.get('segments', []),
            'text': result.get('text', '')
        }
        
        stage1_file = f"{output_base}_S1transcription.json"
        with open(stage1_file, 'w', encoding='utf-8') as f:
            json.dump(stage1_data, f, ensure_ascii=False, indent=2)
        
        console.print(f"[green]✓[/green] Saved transcription: [cyan]{Path(stage1_file).name}[/cyan]")
        results['stages']['transcription'] = result
        results['processing']['stages_completed'].append('transcription')
        
        # Clean up model if needed
        if args.model_flush:
            gc.collect()
            if device == "cuda":
                import torch
                torch.cuda.empty_cache()
            del model
            
    except Exception as e:
        console.print(f"[red]✗ Transcription failed: {e}[/red]")
        results['stages']['transcription'] = {'error': str(e)}
        results['processing']['errors'].append({'stage': 'transcription', 'error': str(e)})
        return results
    
    try:
        # Stage 2: Speaker Diarization
        if args.diarize and hf_token:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[magenta]👥 Stage 2: Speaker Diarization[/magenta]", total=None)
                
                diarize_model = DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=device
                )
                
                # Run diarization
                diarize_segments = diarize_model(
                    audio,
                    min_speakers=args.min_speakers,
                    max_speakers=args.max_speakers
                )
                
                # Assign speakers to words
                result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # Save Stage 2 results with metadata
            speakers_found = list(set(
                seg.get('speaker', 'UNKNOWN') 
                for seg in result.get('segments', [])
                if seg.get('speaker')
            ))
            
            stage2_data = {
                'metadata': runtime_config,
                'stage': 'diarization',
                'completed_at': datetime.now().isoformat(),
                'diarization_model': 'pyannote/speaker-diarization-3.1',
                'speakers_detected': len(speakers_found),
                'speaker_labels': speakers_found,
                'min_speakers_config': args.min_speakers,
                'max_speakers_config': args.max_speakers,
                'segments': result.get('segments', [])
            }
            
            stage2_file = f"{output_base}_S2diarization.json"
            with open(stage2_file, 'w', encoding='utf-8') as f:
                json.dump(stage2_data, f, ensure_ascii=False, indent=2)
            
            num_speakers = len(speakers_found)
            console.print(f"[green]✓[/green] Saved diarization: [cyan]{Path(stage2_file).name}[/cyan] ([bold]{num_speakers} speakers[/bold])")
            results['stages']['diarization'] = stage2_data
            results['processing']['stages_completed'].append('diarization')
            
            # Clean up
            if args.model_flush:
                gc.collect()
                if device == "cuda":
                    import torch
                    torch.cuda.empty_cache()
                del diarize_model
                
        else:
            console.print("[dim]⏭️  Stage 2: Skipping diarization (no HF token or disabled)[/dim]")
            results['stages']['diarization'] = {'skipped': True}
            
    except Exception as e:
        console.print(f"[red]✗ Diarization failed: {e}[/red]")
        results['stages']['diarization'] = {'error': str(e)}
        results['processing']['errors'].append({'stage': 'diarization', 'error': str(e)})
    
    try:
        # Stage 3: Word-level alignment with wav2vec2
        if not args.no_align:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[blue]🎵 Stage 3: Word Alignment[/blue]", total=None)
                
                # Load alignment model
                model_a, metadata = whisperx.load_align_model(
                    language_code="fa",
                    device=device
                )
                
                # Perform alignment
                result = whisperx.align(
                    result["segments"], 
                    model_a, 
                    metadata, 
                    audio, 
                    device
                )
            
            # Save Stage 3 results with metadata
            stage3_data = {
                'metadata': runtime_config,
                'stage': 'alignment',
                'completed_at': datetime.now().isoformat(),
                'alignment_model': 'wav2vec2 (auto-selected for Persian)',
                'word_alignments': True,
                'segments': result.get('segments', []),
                'word_segments': result.get('word_segments', [])
            }
            
            stage3_file = f"{output_base}_S3wav2vec.json"
            with open(stage3_file, 'w', encoding='utf-8') as f:
                json.dump(stage3_data, f, ensure_ascii=False, indent=2)
            console.print(f"[green]✓[/green] Saved alignment: [cyan]{Path(stage3_file).name}[/cyan]")
            results['stages']['wav2vec'] = result
            results['processing']['stages_completed'].append('alignment')
            
            # Clean up
            if args.model_flush:
                gc.collect()
                if device == "cuda":
                    import torch
                    torch.cuda.empty_cache()
                del model_a
                
        else:
            console.print("[dim]⏭️  Stage 3: Skipping alignment[/dim]")
            results['stages']['wav2vec'] = {'skipped': True}
            
    except Exception as e:
        console.print(f"[red]✗ Alignment failed: {e}[/red]")
        results['stages']['wav2vec'] = {'error': str(e)}
        results['processing']['errors'].append({'stage': 'alignment', 'error': str(e)})
    
    # Generate final readable transcript
    try:
        console.print("\n[bold green]📝 Generating final transcript...[/bold green]")
        
        # Use the most complete result available
        final_segments = results['stages'].get('wav2vec', {}).get('segments') or \
                        results['stages'].get('diarization', {}).get('segments') or \
                        results['stages'].get('transcription', {}).get('segments', [])
        
        # Create readable text file
        txt_file = f"{output_base}_final.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            current_speaker = None
            
            for segment in final_segments:
                speaker = segment.get('speaker', 'SPEAKER_UNKNOWN')
                text = segment.get('text', '').strip()
                
                if text:
                    if speaker != current_speaker:
                        f.write(f"\n[{speaker}]:\n")
                        current_speaker = speaker
                    f.write(f"{text}\n")
                    
        console.print(f"[green]✓[/green] Saved transcript: [cyan]{Path(txt_file).name}[/cyan]")
        
        # Update processing completion time
        results['processing']['completed_at'] = datetime.now().isoformat()
        results['processing']['total_duration'] = (
            datetime.fromisoformat(results['processing']['completed_at']) - 
            datetime.fromisoformat(results['processing']['started_at'])
        ).total_seconds()
        
        # Also save complete JSON with all metadata
        complete_file = f"{output_base}_complete.json"
        with open(complete_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        console.print(f"[green]✓[/green] Saved complete: [cyan]{Path(complete_file).name}[/cyan]")
        
    except Exception as e:
        console.print(f"[red]✗ Final output generation failed: {e}[/red]")
        results['processing']['errors'].append({'stage': 'final_output', 'error': str(e)})
    
    # Final processing update
    results['processing']['completed_at'] = datetime.now().isoformat()
    
    console.rule(style="green")
    return results


def get_processed_files(output_dir):
    """Get list of already processed files in the output directory."""
    processed = set()
    output_path = Path(output_dir)
    
    if output_path.exists():
        # Check for completed transcription files
        for file in output_path.glob("*_S1transcription.json"):
            # Extract the base name (remove _S1transcription.json)
            base_name = file.stem.replace('_S1transcription', '')
            processed.add(base_name)
    
    return processed


def get_output_directory(input_folder, resume=False):
    """
    Create or select output directory based on input folder name and timestamp.
    
    Args:
        input_folder: Path to input folder
        resume: Whether to resume from existing folder
    
    Returns:
        Path object for output directory
    """
    folder_name = Path(input_folder).name
    base_output = Path("data/outputs")
    base_output.mkdir(parents=True, exist_ok=True)
    
    # Find existing folders for this input
    existing_folders = sorted([
        d for d in base_output.glob(f"{folder_name}_*")
        if d.is_dir()
    ])
    
    if existing_folders and resume:
        # Create table for existing folders
        table = Table(title="📁 Existing Output Folders", show_header=True)
        table.add_column("#", style="cyan", justify="center")
        table.add_column("Folder Name", style="white")
        table.add_column("Files Processed", style="green", justify="center")
        
        for i, folder in enumerate(existing_folders, 1):
            processed_count = len(list(folder.glob("*_S1transcription.json")))
            table.add_row(str(i), folder.name, str(processed_count))
        
        table.add_row("0", "[bold yellow]Create new folder[/bold yellow]", "-")
        console.print(table)
        
        while True:
            try:
                choice = console.input("\n[bold cyan]Select folder number:[/bold cyan] ")
                choice = int(choice)
                
                if choice == 0:
                    # Create new folder
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = base_output / f"{folder_name}_{timestamp}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    console.print(f"\n[green]✓[/green] Created new folder: [cyan]{output_dir}[/cyan]")
                    return output_dir
                elif 1 <= choice <= len(existing_folders):
                    output_dir = existing_folders[choice - 1]
                    console.print(f"\n[green]✓[/green] Resuming in: [cyan]{output_dir}[/cyan]")
                    return output_dir
                else:
                    console.print("[red]Invalid choice. Please try again.[/red]")
            except (ValueError, KeyboardInterrupt):
                console.print("\n[yellow]Creating new folder...[/yellow]")
                break
    
    # Create new folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output / f"{folder_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"\n[green]✓[/green] Created folder: [cyan]{output_dir}[/cyan]")
    
    return output_dir


def process_folder(input_folder, args):
    """Process all audio files in a folder with resume capability."""
    
    # Get or create output directory
    output_dir = get_output_directory(input_folder, resume=args.resume)
    
    # Get already processed files if resuming
    processed_files = get_processed_files(output_dir) if args.resume else set()
    
    # Supported audio formats
    audio_extensions = {'.m4a', '.mp3', '.wav', '.mp4', '.aac', '.flac', '.ogg'}
    
    # Find all audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(Path(input_folder).glob(f"*{ext}"))
    
    if not audio_files:
        console.print(f"[red]No audio files found in {input_folder}[/red]")
        return
    
    # Filter out already processed files
    files_to_process = []
    for audio_file in audio_files:
        if audio_file.stem not in processed_files:
            files_to_process.append(audio_file)
        else:
            console.print(f"[dim]⏭️  Skipping: {audio_file.name}[/dim]")
    
    if not files_to_process:
        console.print(Panel.fit(
            "[bold green]✅ All files have been processed![/bold green]",
            border_style="green"
        ))
        return
    
    # Status summary
    status_table = Table(show_header=False, box=None)
    status_table.add_column(style="bold cyan")
    status_table.add_column(style="white")
    status_table.add_row("Total files:", str(len(audio_files)))
    status_table.add_row("Already processed:", f"[green]{len(processed_files)}[/green]")
    status_table.add_row("To process:", f"[yellow]{len(files_to_process)}[/yellow]")
    
    console.print(Panel(
        status_table,
        title="[bold]📊 Processing Status[/bold]",
        border_style="cyan"
    ))
    
    # Process each file
    all_results = []
    with Progress(
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        BarColumn(bar_width=30),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        TimeRemainingColumn(),
        console=console
    ) as overall_progress:
        overall_task = overall_progress.add_task(
            "Overall Progress",
            total=len(files_to_process),
            filename="Overall"
        )
        
        for i, audio_file in enumerate(files_to_process, 1):
            console.print(f"\n[bold cyan][🎧 {i}/{len(files_to_process)}][/bold cyan]")
            result = transcribe_audio(str(audio_file), args, output_dir)
            all_results.append(result)
            overall_progress.update(overall_task, advance=1)
    
    # Save or update summary
    summary_file = output_dir / "processing_summary.json"
    
    # Load existing summary if resuming
    existing_results = []
    if summary_file.exists() and args.resume:
        with open(summary_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            existing_results = existing_data.get('results', [])
    
    # Combine results
    all_results = existing_results + all_results
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'processed_files': len(all_results),
            'timestamp': datetime.now().isoformat(),
            'settings': vars(args),
            'results': all_results
        }, f, ensure_ascii=False, indent=2)
    
    console.print(Panel.fit(
        f"[bold green]✓ Processing Complete![/bold green]\n\n"
        f"Summary saved to: [cyan]{summary_file}[/cyan]",
        border_style="green"
    ))


def main():
    parser = argparse.ArgumentParser(
        description='Transcribe Persian interviews with speaker diarization'
    )
    
    parser.add_argument(
        'input_path',
        help='Path to audio file or folder containing audio files'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume processing from existing output folder'
    )
    
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use (default: cpu for macOS compatibility)'
    )
    
    parser.add_argument(
        '--compute_type',
        default='int8',
        choices=['int8', 'float16', 'float32'],
        help='Compute type (default: int8 for efficiency)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for processing (default: 4, reduce if OOM)'
    )
    
    parser.add_argument(
        '--diarize',
        action='store_true',
        default=True,
        help='Enable speaker diarization (default: True)'
    )
    
    parser.add_argument(
        '--min_speakers',
        type=int,
        default=2,
        help='Minimum number of speakers (default: 2)'
    )
    
    parser.add_argument(
        '--max_speakers',
        type=int,
        default=5,
        help='Maximum number of speakers (default: 5)'
    )
    
    parser.add_argument(
        '--no_align',
        action='store_true',
        help='Skip wav2vec2 alignment stage'
    )
    
    parser.add_argument(
        '--model_flush',
        action='store_true',
        default=True,
        help='Flush models between stages to save memory (default: True)'
    )
    
    args = parser.parse_args()
    
    # Display welcome message
    console.print(Panel.fit(
        "[bold cyan]🎙️  Persian Interview Transcription Tool[/bold cyan]\n"
        "[white]Using WhisperX with large-v3 model[/white]",
        border_style="cyan"
    ))
    
    # Check if input is file or directory
    input_path = Path(args.input_path)
    
    if input_path.is_file():
        # For single file, create output directory based on parent folder
        parent_folder = input_path.parent.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/outputs") / f"{parent_folder}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n✓ Created output folder: {output_dir}")
        transcribe_audio(str(input_path), args, output_dir)
    elif input_path.is_dir():
        process_folder(str(input_path), args)
    else:
        print(f"Error: {input_path} is neither a file nor a directory")
        sys.exit(1)


if __name__ == '__main__':
    main()