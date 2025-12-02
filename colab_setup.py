#!/usr/bin/env python3
"""
Colab Setup Script for Geometry Tools + Qwen 3VL Integration

This script handles all the setup needed for Google Colab environment.
Run this in a Colab cell to prepare the environment.

Usage:
    !python colab_setup.py
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🎯 GPU detected: {gpu_name}")
            print(f"💾 GPU memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("⚠️  No GPU detected. Using CPU (slower performance)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet")
        return False

def main():
    """Main setup function"""
    print("🚀 Geometry Tools + Qwen 3VL Colab Setup")
    print("=" * 50)
    
    # Check environment
    print("🔍 Checking Colab environment...")
    colab_env = 'google.colab' in sys.modules
    if colab_env:
        print("✅ Running in Google Colab")
    else:
        print("⚠️  Not in Colab - some features may not work")
    
    # Install geometry tools from GitHub
    success = run_command(
        "pip install git+https://github.com/Ayingxizhao/geometry-tools.git[colab]",
        "Installing Geometry Tools from GitHub"
    )
    
    if not success:
        print("❌ Failed to install geometry tools")
        return False
    
    # Install Qwen 3VL dependencies
    dependencies = [
        "transformers>=4.20.0",
        "torch>=1.12.0",
        "accelerate",
        "qwen-vl-utils",
        "Pillow>=8.3.0"
    ]
    
    for dep in dependencies:
        success = run_command(
            f"pip install {dep} --quiet",
            f"Installing {dep}"
        )
        if not success:
            print(f"⚠️  Failed to install {dep}")
    
    # Check GPU availability
    print("\n🎯 Checking GPU availability...")
    check_gpu()
    
    # Test imports
    print("\n🧪 Testing imports...")
    try:
        import torch
        import transformers
        from geometry_tools import is_line_longer, measure_line_length
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Load Qwen 3VL model")
    print("2. Create test images")
    print("3. Run visual reasoning tasks")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
