#!/usr/bin/env python3
"""
Install missing dependencies for the three methods.
Run this before trying the fixed extractor.
"""

import subprocess
import sys
import os

def run_pip_install(package, description=""):
    """Install a package with pip."""
    print(f"📦 Installing {package} {description}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def install_all_dependencies():
    """Install all required dependencies."""
    print("🚀 Installing dependencies for fixed three methods...")
    
    dependencies = [
        ("timm", "(needed for DETR)"),
        ("transformers", "(Hugging Face transformers)"),
        ("layoutparser", "(document layout analysis)"),
        ("torch", "(PyTorch)"),
        ("torchvision", "(PyTorch vision)"),
        ("pillow", "(PIL image processing)"),
    ]
    
    success_count = 0
    
    for package, description in dependencies:
        if run_pip_install(package, description):
            success_count += 1
    
    # Try Detectron2 with proper CUDA version
    print(f"\n📦 Installing detectron2 (this might take a while)...")
    
    # Detect CUDA version
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            print(f"🔍 Detected CUDA version: {cuda_version}")
            
            if cuda_version.startswith("12.1"):
                detectron2_url = "detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.5/index.html"
            elif cuda_version.startswith("11.8"):
                detectron2_url = "detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.5/index.html"
            else:
                detectron2_url = "detectron2"
        else:
            print("🔍 No CUDA detected, installing CPU version")
            detectron2_url = "detectron2"
    except:
        detectron2_url = "detectron2"
    
    if run_pip_install(detectron2_url, "(object detection framework)"):
        success_count += 1
    
    print(f"\n📊 Installation summary:")
    print(f"✅ Successfully installed: {success_count}/{len(dependencies) + 1}")
    
    if success_count == len(dependencies) + 1:
        print(f"🎉 All dependencies installed successfully!")
        return True
    else:
        print(f"⚠️  Some installations failed. You may need to install them manually.")
        return False

def check_current_installations():
    """Check what's currently installed."""
    print("🔍 Checking current installations...")
    
    packages_to_check = [
        ("timm", "import timm"),
        ("transformers", "from transformers import DetrImageProcessor"),
        ("layoutparser", "import layoutparser"),
        ("detectron2", "import detectron2"),
        ("torch", "import torch"),
        ("torchvision", "import torchvision"),
        ("PIL", "from PIL import Image"),
    ]
    
    installed = []
    missing = []
    
    for package_name, import_statement in packages_to_check:
        try:
            exec(import_statement)
            print(f"✅ {package_name} - installed")
            installed.append(package_name)
        except ImportError:
            print(f"❌ {package_name} - missing")
            missing.append(package_name)
    
    print(f"\n📊 Status: {len(installed)}/{len(packages_to_check)} packages available")
    
    if missing:
        print(f"💡 Missing packages: {', '.join(missing)}")
        return False
    else:
        print(f"🎉 All packages are installed!")
        return True

def main():
    """Main installation function."""
    print("🔧 Dependency Installation Tool for Fixed Three Methods")
    print("=" * 60)
    
    # Check current status
    all_installed = check_current_installations()
    
    if all_installed:
        print(f"\n✅ All dependencies are already installed!")
        print(f"🚀 You can now run: python fixed_three_methods_extractor.py paper.pdf")
        return
    
    print(f"\n💡 Some dependencies are missing. Installing them now...")
    
    # Install missing dependencies
    install_success = install_all_dependencies()
    
    if install_success:
        print(f"\n🎉 Installation complete!")
        print(f"🚀 You can now test the fixed three methods:")
        print(f"   python fixed_three_methods_extractor.py paper.pdf")
        print(f"   python fixed_three_methods_extractor.py paper.pdf --method layoutparser")
        print(f"   python fixed_three_methods_extractor.py paper.pdf --method detr")
        print(f"   python fixed_three_methods_extractor.py paper.pdf --method detectron2")
    else:
        print(f"\n⚠️  Some installations failed.")
        print(f"💡 Manual installation commands:")
        print(f"   pip install timm transformers layoutparser")
        print(f"   pip install torch torchvision")
        print(f"   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.5/index.html")

if __name__ == "__main__":
    main()
    