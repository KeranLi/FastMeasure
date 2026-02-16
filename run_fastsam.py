#!/usr/bin/env python3
"""
FastSAM Startup Script - Simplified Version
File: run_fastsam.py
Function: Provide command line interface to start FastSAM system
"""

import os
import sys
import time
import traceback
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import core CLI modules
from core import (
    create_base_parser,
    terminal_interactive_wizard,
    update_config_from_args,
    print_summary,
    print_welcome
)

# Import FastSAM system
try:
    from fastsam.rock_fastsam_system import RockUltraSystem
    SYSTEM_AVAILABLE = True
except ImportError as e:
    SYSTEM_AVAILABLE = False
    print(f"Failed to import FastSAM system: {e}")


# Constants
SYSTEM_NAME = "FastSAM"
VERSION = "FastSAM Rock Grain Auto Segmentation System v2.0.0"
DEFAULT_CONF = 0.25
DEFAULT_OUTPUT = "results_fastsam"
DEFAULT_CONFIG = "config.yaml"


EPILOG = """
Usage Examples:
  # Terminal interactive wizard mode
  python run_fastsam.py
  
  # Process single image
  python run_fastsam.py --input path/to/image.tif
  
  # Batch process folder
  python run_fastsam.py --input path/to/folder --batch
  
  # GUI interactive mode
  python run_fastsam.py --interactive [--input image_path]
  
  # Use custom config
  python run_fastsam.py --config config.yaml --input image.tif
  
  # Adjust processing parameters
  python run_fastsam.py --input image.tif --conf 0.3 --min-area 50
"""


def parse_arguments():
    """Parse command line arguments"""
    parser = create_base_parser(
        description=f"{SYSTEM_NAME} Rock Grain Auto Segmentation System",
        epilog=EPILOG,
        version=VERSION
    )
    
    # Update default config for FastSAM
    parser.set_defaults(config=DEFAULT_CONFIG)
    
    # FastSAM specific parameters
    parser.add_argument(
        "--gui-backend",
        type=str,
        choices=['auto', 'agg', 'tkagg', 'qt5agg'],
        default='auto',
        help="Specify matplotlib backend (default: auto)"
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    if not SYSTEM_AVAILABLE:
        print("FastSAM system unavailable, cannot start")
        return 1
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if entering terminal interactive wizard mode
    is_interactive_mode = len(sys.argv) == 1
    
    if is_interactive_mode:
        # Terminal interactive wizard mode
        print_welcome(SYSTEM_NAME, [
            "Auto segmentation | Batch processing | GUI interactive segmentation",
            f"Version: {VERSION.split()[-1]}"
        ])
        args = terminal_interactive_wizard(SYSTEM_NAME, DEFAULT_CONF, DEFAULT_OUTPUT)
    elif not args.quiet:
        print_welcome(SYSTEM_NAME, [
            "Auto segmentation | Batch processing | GUI interactive segmentation",
            f"Version: {VERSION.split()[-1]}"
        ])
        print(f"\nStart time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Interactive mode
    if args.interactive:
        try:
            system = RockUltraSystem(args.config)
            
            if not args.quiet:
                system.show_system_info()
            
            print("\nStarting GUI interactive mode...")
            system.run_interactive_mode(args.input)
            
            print("\n" + "=" * 60)
            print("GUI interaction ended")
            print("=" * 60)
            
            # Save interactive results
            if system.interactive_system and hasattr(system.interactive_system, 'grains'):
                grains_count = len(system.interactive_system.grains)
                if grains_count > 0:
                    print(f"Marked {grains_count} grains during interaction")
                    save_choice = input("Save interactive results? (y/n): ").strip().lower()
                    if save_choice in ['y', 'yes'] and hasattr(system.interactive_system, '_generate_complete_outputs'):
                        output_dir = system.interactive_system._generate_complete_outputs()
                        if output_dir:
                            print(f"Results saved to: {output_dir}")
            
            return 0
        except Exception as e:
            print(f"Interactive mode execution failed: {e}")
            traceback.print_exc()
            return 1
    
    # Check input path
    if not args.input:
        print("No input path specified")
        print("\nPlease use one of the following:")
        print(f"  1. Direct run: python run_fastsam.py")
        print(f"  2. Specify input: python run_fastsam.py --input image_path")
        print(f"  3. Batch processing: python run_fastsam.py --input folder --batch")
        return 1
    
    if not os.path.exists(args.input):
        print(f"Input path does not exist: {args.input}")
        return 1
    
    # Create system instance
    try:
        system = RockUltraSystem(args.config)
    except Exception as e:
        print(f"System initialization failed: {e}")
        return 1
    
    # Update config
    system = update_config_from_args(system, args)
    
    # Show system info
    if not args.quiet:
        system.show_system_info()
    
    # Initialize models
    print("\nInitializing AI models...")
    if not system.initialize_models():
        print("Model initialization failed")
        return 1
    print("AI models initialized successfully")
    
    # Process
    results = None
    
    if os.path.isfile(args.input):
        print(f"\nProcessing single image: {args.input}")
        results = system.process_single_image(args.input)
    elif os.path.isdir(args.input):
        if args.batch:
            print(f"\nBatch processing folder: {args.input}")
            results = system.batch_process(args.input)
        else:
            print("Input path is folder, please use --batch for batch processing")
            return 1
    
    # Show results
    if results and not args.quiet:
        print_summary(results, SYSTEM_NAME)
    
    if not args.quiet:
        print(f"\nAll results saved to: {system.output_root}")
        print("\n" + "=" * 70)
        print(f"{SYSTEM_NAME} processing complete!")
        print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
