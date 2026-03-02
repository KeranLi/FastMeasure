#!/usr/bin/env python3
"""
Check and manage model files for FastMeasure

This script checks if required model files are present.
Model files must be downloaded manually from the links provided below.
"""

import os
import sys
from pathlib import Path

# Model file information
MODELS = {
    "yolo": {
        "name": "YOLO Detection Model",
        "filename": "best_yolo_20260107.pt",
        "size_mb": 100,
        "required": True,
    },
    "fastsam": {
        "name": "FastSAM Model", 
        "filename": "FastSAM-s.pt",
        "size_mb": 150,
        "required": True,
    },
    "mobilesam": {
        "name": "MobileSAM Model",
        "filename": "mobile_sam.pt", 
        "size_mb": 450,
        "required": False,  # Optional
    }
}

# Download links
DOWNLOAD_URLS = {
    "google_drive": "https://drive.google.com/drive/folders/1SPah9woaytIeinkLzQgGiXyj_SCJ3v1q?usp=drive_link",
    "github": "https://github.com/KeranLi/FastMeasure/releases",
}


def check_model(model_key):
    """Check if model file exists"""
    if model_key not in MODELS:
        return False
    model_info = MODELS[model_key]
    model_path = Path("models") / model_info["filename"]
    return model_path.exists() and model_path.stat().st_size > 0


def get_model_status():
    """Get status of all models"""
    status = {}
    for key, info in MODELS.items():
        exists = check_model(key)
        status[key] = {
            **info,
            "exists": exists
        }
    return status


def print_download_instructions():
    """Print download instructions"""
    print("\n" + "="*60)
    print("Download Instructions")
    print("="*60)
    print("\nModel files are not included in the repository due to size.")
    print("Please download from Google Drive:")
    print(f"\n   {DOWNLOAD_URLS['google_drive']}")
    print("\nFiles to download:")
    print("   - best_yolo_20260107.pt (required, ~100 MB)")
    print("   - FastSAM-s.pt (required, ~150 MB)")
    print("   - mobile_sam.pt (optional, ~450 MB)")
    print("\nAfter downloading, place the files in the 'models/' folder.")


def main():
    print("="*60)
    print("FastMeasure Model Checker")
    print("="*60)
    
    # Create models directory if not exists
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Check all models
    status = get_model_status()
    
    print("\nModel Status:")
    print("-"*60)
    
    all_required_present = True
    
    for key, info in status.items():
        symbol = "✓" if info["exists"] else "✗"
        required = "(required)" if info["required"] else "(optional)"
        print(f"  {symbol} {info['name']:<25} {info['filename']:<25} {required}")
        
        if info["required"] and not info["exists"]:
            all_required_present = False
    
    print("-"*60)
    
    if all_required_present:
        print("\n✓ All required models are present!")
        print("\nYou can now run FastMeasure:")
        print("  python run_fastsam.py --input your_image.jpg")
        return 0
    else:
        print("\n✗ Some required models are missing!")
        print_download_instructions()
        
        print("\nExpected files in models/ folder:")
        for key, info in MODELS.items():
            status = "✓ Present" if check_model(key) else "✗ Missing"
            print(f"  - {info['filename']:<30} (~{info['size_mb']} MB) {status}")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
