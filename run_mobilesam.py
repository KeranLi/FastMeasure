#!/usr/bin/env python3
"""
MobileSAM Startup Script - Simplified Version
File: run_mobilesam.py
Function: Provide command line interface to start MobileSAM system
"""

import os
import sys
import time
import traceback
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Smart matplotlib backend setup
import matplotlib

def setup_matplotlib_backend():
    """Smart matplotlib backend setup based on running mode"""
    is_interactive = '--interactive' in sys.argv or '-t' in sys.argv
    
    if is_interactive:
        for backend in ['TkAgg', 'Qt5Agg', 'WXAgg']:
            try:
                matplotlib.use(backend)
                return
            except Exception:
                continue
        matplotlib.use('Agg')
    else:
        matplotlib.use('Agg')

setup_matplotlib_backend()

# Import core CLI modules
from core import (
    create_base_parser,
    terminal_interactive_wizard,
    update_config_from_args,
    print_summary,
    print_welcome
)

# Import MobileSAM system
try:
    from mobilesam.rock_mobilesam_system import RockMobileSystem
    SYSTEM_AVAILABLE = True
except ImportError as e:
    SYSTEM_AVAILABLE = False
    print(f"Failed to import MobileSAM system: {e}")


# Constants
SYSTEM_NAME = "MobileSAM"
VERSION = "MobileSAM Rock Grain Auto Segmentation System v3.0.0"
DEFAULT_CONF = 0.15
DEFAULT_OUTPUT = "results_mobilesam"
DEFAULT_CONFIG = "configs/mobilesam.yaml"


EPILOG = """
Usage Examples:
  # Terminal interactive mode
  python run_mobilesam.py
  
  # Single image auto processing
  python run_mobilesam.py --input path/to/image.tif
  
  # Batch process folder
  python run_mobilesam.py --input path/to/folder --batch
  
  # GUI interactive segmentation
  python run_mobilesam.py --interactive
"""


def parse_arguments():
    """Parse command line arguments"""
    parser = create_base_parser(
        description=f"{SYSTEM_NAME} Rock Grain Auto Segmentation System",
        epilog=EPILOG,
        version=VERSION
    )
    
    # Update default config for MobileSAM
    parser.set_defaults(config=DEFAULT_CONFIG)
    
    # MobileSAM specific parameters
    parser.add_argument(
        "--gui-backend",
        type=str,
        choices=['auto', 'agg', 'tkagg', 'qt5agg', 'webagg'],
        default='auto',
        help="Specify matplotlib backend (default: auto)"
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    if not SYSTEM_AVAILABLE:
        print("MobileSAM system unavailable, cannot start")
        return 1
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if entering terminal interactive mode
    is_interactive_mode = len(sys.argv) == 1
    
    if is_interactive_mode:
        print_welcome(SYSTEM_NAME, [
            "Smart environment detection",
            "Terminal interactive mode",
            "Multi-mode support"
        ])
        args = terminal_interactive_wizard(SYSTEM_NAME, DEFAULT_CONF, DEFAULT_OUTPUT)
    elif not args.quiet:
        print_welcome(SYSTEM_NAME, [
            "Smart environment detection",
            "Terminal interactive mode",
            "Multi-mode support"
        ])
    
    try:
        # Create system instance
        print(f"\nInitializing {SYSTEM_NAME} main system...")
        system = RockMobileSystem(args.config)
        
        # Initialize models
        print("Initializing AI models...")
        if not system.initialize_models():
            print("Model initialization failed")
            return 1
        print("AI models initialized successfully")
        
        # Update config
        system = update_config_from_args(system, args)
        
        # Show system info
        if not args.quiet:
            system.show_system_info()
        
        # Interactive mode
        if args.interactive:
            print(f"\nStarting GUI interactive mode...")
            try:
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
        
        # Check input
        if not args.input:
            print("\nNo input path specified")
            print("  1. python run_mobilesam.py (terminal interactive mode)")
            print("  2. python run_mobilesam.py --input image_path")
            print("  3. python run_mobilesam.py --input folder --batch")
            return 1
        
        input_path = Path(args.input).absolute()
        
        if not input_path.exists():
            print(f"Input path does not exist: {input_path}")
            return 1
        
        # Process
        if input_path.is_file():
            print(f"\nProcessing single image: {input_path}")
            results = system.process_single_image(str(input_path))
            if results and not args.quiet:
                print_summary(results, SYSTEM_NAME)
        elif input_path.is_dir():
            if args.batch:
                print(f"\nBatch processing folder: {input_path}")
                results = system.batch_process(str(input_path))
                if results and not args.quiet:
                    print_summary(results, SYSTEM_NAME)
            else:
                print("Input path is folder, please use --batch parameter")
                return 1
        
        if not args.quiet:
            print(f"\nAll results saved to: {system.output_root}")
            print("\n" + "=" * 70)
            print(f"{SYSTEM_NAME} processing complete!")
            print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"Program error: {e}")
        if args.debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
        sys.exit(1)
