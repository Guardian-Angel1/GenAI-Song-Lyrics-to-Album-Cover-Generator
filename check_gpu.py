#!/usr/bin/env python
"""
GPU & CUDA Configuration Checker
Run this to verify your GPU is properly configured
"""

import sys
import torch
import platform

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_section(text):
    print(f"\n{'─'*70}")
    print(f"  {text}")
    print(f"{'─'*70}")

def main():
    print_header("🎮 GPU & CUDA Configuration Checker")
    
    # System Info
    print_section("System Information")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Python Version: {sys.version}")
    print(f"  PyTorch Version: {torch.__version__}")
    
    # CUDA Info
    print_section("CUDA Configuration")
    print(f"  CUDA Available: {'✅ YES' if torch.cuda.is_available() else '❌ NO'}")
    print(f"  CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"  cuDNN Version: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}")
    print(f"  cuDNN Enabled: {torch.backends.cudnn.enabled}")
    
    if not torch.cuda.is_available():
        print_section("⚠️  GPU NOT DETECTED")
        print("""
  If you have an NVIDIA GPU, please:
  
  1. Install NVIDIA Driver:
     https://www.nvidia.com/Download/driverDetails.aspx
     
  2. Install CUDA Toolkit (11.8 or 12.1):
     https://developer.nvidia.com/cuda-downloads
     
  3. Reinstall PyTorch with GPU support:
     pip uninstall torch torchvision torchaudio
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  
  4. Verify installation:
     python check_gpu.py
        """)
        return
    
    # GPU Devices
    print_section("GPU Devices")
    num_devices = torch.cuda.device_count()
    print(f"  Number of GPUs: {num_devices}")
    
    for i in range(num_devices):
        print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    ├─ Compute Capability: {props.major}.{props.minor}")
        print(f"    ├─ Total Memory: {props.total_memory / 1024 / 1024 / 1024:.2f} GB")
        print(f"    ├─ Max Threads per Block: {props.max_threads_per_block}")
        print(f"    └─ Multi-Processor Count: {props.multi_processor_count}")
    
    # Current Device
    print_section("Current Device")
    current = torch.cuda.current_device()
    print(f"  Active GPU: {current}")
    print(f"  Active GPU Name: {torch.cuda.get_device_name(current)}")
    
    # Memory Info
    print_section("Memory Information")
    if num_devices > 0:
        for i in range(num_devices):
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / 1024 / 1024 / 1024
            reserved = torch.cuda.memory_reserved(i) / 1024 / 1024 / 1024
            total = torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024
            free = total - reserved
            print(f"\n  GPU {i} Memory:")
            print(f"    ├─ Total:     {total:.2f} GB")
            print(f"    ├─ Allocated: {allocated:.2f} GB")
            print(f"    ├─ Reserved:  {reserved:.2f} GB")
            print(f"    └─ Free:      {free:.2f} GB")
    
    # Test Tensor
    print_section("GPU Computation Test")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print("  ✅ GPU computation test: PASSED")
        print(f"  ✅ Matrix multiplication result shape: {z.shape}")
    except Exception as e:
        print(f"  ❌ GPU computation test: FAILED")
        print(f"  Error: {e}")
    
    # Recommendations
    print_section("✅ Status & Recommendations")
    if torch.cuda.is_available():
        print("  ✅ GPU is properly configured!")
        print("\n  Your app will use GPU for fast image generation:")
        print("    • Expected speed: 5-15 seconds per image")
        print("    • Memory efficient: Uses ~3-4GB VRAM")
        print("    • Optimized with xFormers if available\n")
        
        # Check VRAM for model size
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
        print(f"  Your GPU has {total_vram:.1f}GB VRAM - suitable for:")
        if total_vram >= 6:
            print("    ✅ Multiple variations (up to 4)")
            print("    ✅ Higher guidance scale (7-15)")
            print("    ✅ Larger batch sizes")
        elif total_vram >= 4:
            print("    ✅ 1-2 variations")
            print("    ✅ Standard guidance scale (7-10)")
            print("    ✅ Single image generation")
        else:
            print("    ⚠️  Limited VRAM - use 1 variation, guidance 7.0")
    
    print_header("🚀 Ready to Generate!")
    print("""
  Run the app:
    streamlit run app.py --server.fileWatcherType none
  
  Recommended settings for GPU:
    • Art Style: Any style
    • Mood Intensity: 0.5-1.0
    • Guidance Scale: 7.5-12
    • Variations: 2-4 (if 6GB+ VRAM)
    • Color Temperature: Any
    • Lighting: Any
    """)

if __name__ == "__main__":
    main()
