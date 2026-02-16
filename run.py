#!/usr/bin/env python3
"""
FastMeasure Unified Startup Script
File: run.py
Function: Unified entry point supporting both FastSAM and MobileSAM modes

Usage:
  python run.py fastsam          # Start FastSAM
  python run.py mobilesam        # Start MobileSAM
  python run.py --help           # Show help
"""

import os
import sys
import subprocess
from pathlib import Path


def print_usage():
    """Print usage instructions"""
    print("""
FastMeasure - Rock Grain Auto Segmentation System

Usage:
  python run.py fastsam [options]    Start FastSAM mode
  python run.py mobilesam [options]  Start MobileSAM mode

Options:
  --input, -i <path>        Input image or folder path
  --batch, -b              Batch processing mode
  --interactive, -t        GUI interactive mode
  --config, -c <path>      Config file path
  --conf <float>           YOLO confidence threshold
  --min-area <int>         Minimum grain area
  --output, -o <path>      Output directory
  --help                   Show detailed help

Examples:
  # FastSAM process single image
  python run.py fastsam --input image.tif

  # MobileSAM batch processing
  python run.py mobilesam --input ./images --batch

  # Interactive mode
  python run.py mobilesam --interactive
""")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print_usage()
        return 1
    
    mode = sys.argv[1].lower()
    
    if mode in ['--help', '-h', 'help']:
        print_usage()
        return 0
    
    if mode not in ['fastsam', 'mobilesam']:
        print(f"Error: Unknown mode '{mode}'")
        print("Supported modes: fastsam, mobilesam")
        print("\nUse 'python run.py --help' for detailed help")
        return 1
    
    # Determine script to run
    script_name = f"run_{mode}.py"
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"Error: Cannot find startup script {script_name}")
        return 1
    
    # Build command line arguments
    # Remove first argument (run.py), keep remaining
    remaining_args = sys.argv[2:]
    
    # Execute corresponding startup script
    cmd = [sys.executable, str(script_path)] + remaining_args
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        return 1
    except Exception as e:
        print(f"Startup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
