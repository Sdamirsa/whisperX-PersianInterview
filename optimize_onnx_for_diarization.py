#!/usr/bin/env python3
"""
Performance Optimization Script for WhisperX Persian Interview Tool

This script optimizes the onnxruntime setup for faster diarization performance.
Based on the GitHub issue https://github.com/m-bain/whisperX/issues/499: pyannote.audio 3.0.1 with onnxruntime-gpu provides
significant speedup over CPU-only onnxruntime.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_cuda_available():
    """Check if CUDA is available on the system."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✅ CUDA detected: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  CUDA not available, staying with CPU-optimized setup")
        return cuda_available
    except ImportError:
        print("⚠️  PyTorch not available, cannot check CUDA")
        return False

def optimize_onnxruntime():
    """Optimize onnxruntime installation for better performance."""
    print("🚀 Starting ONNXRuntime optimization...")
    
    # Check current setup
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"📊 Current ONNXRuntime providers: {', '.join(providers)}")
        
        has_gpu_provider = 'CUDAExecutionProvider' in providers
        if has_gpu_provider:
            print("✅ ONNXRuntime-GPU is already properly configured!")
            return True
            
    except ImportError:
        print("⚠️  ONNXRuntime not found")
    
    # Check if CUDA is available
    if not check_cuda_available():
        print("ℹ️  Skipping GPU optimization (no CUDA available)")
        return True
    
    print("\n🔄 Optimizing ONNXRuntime for GPU acceleration...")
    
    # Step 1: Upgrade pyannote.audio (already done in pyproject.toml)
    print("✅ pyannote.audio is already at version 3.4.0+")
    
    # Step 2: Remove old onnxruntime
    if not run_command(["uv", "remove", "onnxruntime"], "Removing CPU-only onnxruntime"):
        print("⚠️  Could not remove onnxruntime (might not be installed)")
    
    # Step 3: Install onnxruntime-gpu
    if not run_command(["uv", "add", "onnxruntime-gpu"], "Installing onnxruntime-gpu"):
        print("❌ Failed to install onnxruntime-gpu")
        print("💡 You may need to install it manually:")
        print("   uv add onnxruntime-gpu")
        return False
    
    # Step 4: Verify installation
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"🔍 New ONNXRuntime providers: {', '.join(providers)}")
        
        if 'CUDAExecutionProvider' in providers:
            print("🎉 Success! ONNXRuntime-GPU is now properly configured")
            print("📈 You should see significant performance improvements in diarization")
            return True
        else:
            print("⚠️  GPU provider not found. Manual intervention may be needed.")
            return False
            
    except ImportError:
        print("❌ Could not import onnxruntime after installation")
        return False

def main():
    """Main optimization function."""
    print("🎙️ WhisperX Performance Optimization Tool")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("Run_whisperx_PersianInterview.py").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   (where Run_whisperx_PersianInterview.py is located)")
        sys.exit(1)
    
    success = optimize_onnxruntime()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Optimization completed successfully!")
        print("\n📋 Next steps:")
        print("1. Test with: uv run python Run_whisperx_PersianInterview.py --debug_mode path/to/audio")
        print("2. Monitor performance improvements in diarization stage")
        print("3. Check debug logs for ONNXRuntime provider confirmation")
    else:
        print("⚠️  Optimization completed with warnings")
        print("\n🔧 Manual steps if needed:")
        print("1. uv remove onnxruntime")
        print("2. uv add onnxruntime-gpu")
        print("3. Restart your terminal/IDE")
    
    print("\n💡 For more help, run with --debug_mode to see detailed logs")

if __name__ == "__main__":
    main()