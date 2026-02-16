"""
Command Line Interface Base Module - Provides shared CLI functionality
File: core/cli_base.py
Function: Provides shared CLI logic for FastSAM and MobileSAM startup scripts
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any


class SimpleArgs:
    """Simple argument namespace class for terminal interactive mode"""
    
    def __init__(self):
        self.input: Optional[str] = None
        self.batch: bool = False
        self.interactive: bool = False
        self.config: str = "config.yaml"
        self.conf: Optional[float] = None
        self.min_area: Optional[int] = None
        self.min_bbox_area: Optional[int] = None
        self.remove_edge: bool = False
        self.output: Optional[str] = None
        self.performance: bool = False
        self.debug: bool = False
        self.quiet: bool = False


def create_base_parser(description: str, epilog: str, version: str) -> argparse.ArgumentParser:
    """
    Create base argument parser
    
    Args:
        description: Program description
        epilog: Help epilog
        version: Version string
        
    Returns:
        Argument parser
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog
    )
    
    # Input arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input image or folder path"
    )
    
    # Processing mode
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Batch processing mode (when input is folder)"
    )
    
    parser.add_argument(
        "--interactive", "-t",
        action="store_true",
        help="GUI interactive mode (manual grain selection, requires GUI)"
    )
    
    # Config file
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Config file path"
    )
    
    # Processing parameters
    parser.add_argument(
        "--conf",
        type=float,
        help="YOLO detection confidence threshold (0-1)"
    )
    
    parser.add_argument(
        "--min-area",
        type=int,
        help="Minimum grain area (pixels)"
    )
    
    parser.add_argument(
        "--min-bbox-area",
        type=int,
        help="Minimum bounding box area (pixels)"
    )
    
    parser.add_argument(
        "--remove-edge",
        action="store_true",
        help="Remove edge grains"
    )
    
    # Output parameters
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory path"
    )
    
    # Performance parameters
    parser.add_argument(
        "--performance", "-p",
        action="store_true",
        help="Enable performance monitoring mode"
    )
    
    # Debug parameters
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode"
    )
    
    # Other options
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode, reduce output"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=version
    )
    
    return parser


def terminal_interactive_wizard(system_name: str, default_conf: float, 
                                default_output: str) -> SimpleArgs:
    """
    Terminal interactive wizard mode
    
    Args:
        system_name: System name
        default_conf: Default confidence
        default_output: Default output directory
        
    Returns:
        Argument object
    """
    print("\n" + "=" * 70)
    print(f"    {system_name} Terminal Interactive Wizard")
    print("=" * 70)
    print(f"Welcome to {system_name}! I will guide you through the setup.")
    print("=" * 70)
    
    args = SimpleArgs()
    
    # 1. Select processing mode
    print("\nPlease select processing mode:")
    print("  1.  Automatic processing mode (YOLO+SAM automatic segmentation)")
    print("  2.  Batch processing mode (process entire folder)")
    print("  3.  GUI interactive segmentation (manual grain selection, requires GUI)")
    print("  4.  Exit program")
    
    while True:
        try:
            choice = input("\nPlease enter option number (1-4): ").strip()
            if choice == '1':
                print("Selected: Automatic processing mode")
                break
            elif choice == '2':
                print("Selected: Batch processing mode")
                args.batch = True
                break
            elif choice == '3':
                print("Selected: GUI interactive segmentation mode")
                args.interactive = True
                break
            elif choice == '4':
                print("Exiting program")
                sys.exit(0)
            else:
                print("Invalid option, please try again")
        except KeyboardInterrupt:
            print("\nUser interrupted, exiting program")
            sys.exit(0)
    
    # 2. Get input path
    if not args.interactive:
        print("\nPlease enter image or folder path:")
        print("Tip: You can drag and drop files/folders to the terminal")
        
        while True:
            try:
                prompt = "Please enter folder path: " if args.batch else "Please enter image path: "
                user_input = input(prompt).strip()
                
                if user_input:
                    input_path = Path(user_input)
                    
                    # Support relative and absolute paths
                    if not input_path.exists():
                        current_dir = Path.cwd() / user_input
                        if current_dir.exists():
                            input_path = current_dir
                    
                    if input_path.exists():
                        args.input = str(input_path.absolute())
                        
                        if input_path.is_file():
                            print(f"Found file: {input_path.name}")
                        elif input_path.is_dir():
                            print(f"Found folder")
                        break
                    else:
                        print(f"Path does not exist: {user_input}")
                else:
                    print("Path cannot be empty")
            
            except KeyboardInterrupt:
                print("\nUser interrupted, exiting program")
                sys.exit(0)
    
    # 3. Set parameters (only for automatic processing mode)
    if not args.interactive:
        print("\n  Parameter settings (press Enter to use defaults):")
        
        # Confidence threshold
        print(f"\nYOLO detection confidence threshold (0.0-1.0)")
        print(f"Default: {default_conf}")
        
        while True:
            try:
                conf_input = input(f"Please enter confidence threshold (default {default_conf}): ").strip()
                
                if conf_input == "":
                    args.conf = default_conf
                    print(f"Using default confidence: {default_conf}")
                    break
                else:
                    conf_value = float(conf_input)
                    if 0.0 <= conf_value <= 1.0:
                        args.conf = conf_value
                        print(f"Set confidence: {conf_value}")
                        break
                    else:
                        print("Confidence must be between 0.0 and 1.0")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nUser interrupted, exiting program")
                sys.exit(0)
        
        # Output directory
        print("\nOutput directory settings")
        print(f"Default: {default_output}")
        
        while True:
            try:
                output_input = input(f"Please enter output directory (default {default_output}): ").strip()
                
                if output_input == "":
                    args.output = default_output
                    print(f"Using default output directory: {default_output}")
                    break
                else:
                    output_path = Path(output_input)
                    output_path.mkdir(parents=True, exist_ok=True)
                    args.output = output_input
                    print(f"Set output directory: {output_path}")
                    break
            except KeyboardInterrupt:
                print("\nUser interrupted, exiting program")
                sys.exit(0)
        
        # Advanced settings
        print("\nAdvanced settings (optional)")
        
        # Minimum area
        while True:
            try:
                min_area_input = input("Minimum grain area in pixels (default 30, press Enter to skip): ").strip()
                
                if min_area_input == "":
                    break
                else:
                    min_area = int(min_area_input)
                    if min_area > 0:
                        args.min_area = min_area
                        print(f"Set minimum grain area: {min_area} pixels")
                        break
                    else:
                        print("Minimum area must be greater than 0")
            except ValueError:
                print("Please enter a valid integer")
            except KeyboardInterrupt:
                print("\nUser interrupted, exiting program")
                sys.exit(0)
        
        # Remove edge grains
        while True:
            try:
                remove_edge_input = input("Remove edge grains? (y/n, default n): ").strip().lower()
                
                if remove_edge_input in ['', 'n', 'no']:
                    print("Keep edge grains")
                    break
                elif remove_edge_input in ['y', 'yes']:
                    args.remove_edge = True
                    print("Will remove edge grains")
                    break
                else:
                    print("Please enter y or n")
            except KeyboardInterrupt:
                print("\nUser interrupted, exiting program")
                sys.exit(0)
    
    # 4. Confirm settings
    print("\n" + "=" * 70)
    print("   Configuration Confirmation")
    print("=" * 70)
    
    if args.interactive:
        print("   Mode: GUI interactive segmentation")
        if args.input:
            print(f"   Input: {args.input}")
        else:
            print("   Input: Use GUI to select file")
    elif args.batch:
        print(f"   Mode: Batch processing")
        print(f"   Input: {args.input}")
    else:
        print(f"   Mode: Automatic processing")
        print(f"   Input: {args.input}")
    
    if args.conf:
        print(f"   Confidence: {args.conf}")
    
    print(f"   Output: {args.output or default_output}")
    
    if args.min_area:
        print(f"   Minimum area: {args.min_area} pixels")
    
    if args.remove_edge:
        print("   Edge processing: Remove edge grains")
    
    print("=" * 70)
    
    # Confirm start processing
    while True:
        try:
            confirm = input("\nStart processing? (y/n): ").strip().lower()
            
            if confirm in ['y', 'yes']:
                print("Starting processing...")
                return args
            elif confirm in ['n', 'no']:
                print("User cancelled, exiting program")
                sys.exit(0)
            else:
                print("Please enter y or n")
        except KeyboardInterrupt:
            print("\nUser interrupted, exiting program")
            sys.exit(0)


def update_config_from_args(system, args) -> Any:
    """
    Update config based on command line arguments
    
    Args:
        system: System instance
        args: Argument object
        
    Returns:
        Updated system instance
    """
    config_updated = False
    
    if hasattr(args, 'conf') and args.conf is not None:
        system.config['processing']['yolo_confidence'] = args.conf
        config_updated = True
    
    if hasattr(args, 'min_area') and args.min_area is not None:
        system.config['processing']['min_area'] = args.min_area
        config_updated = True
    
    if hasattr(args, 'min_bbox_area') and args.min_bbox_area is not None:
        system.config['processing']['min_bbox_area'] = args.min_bbox_area
        config_updated = True
    
    if hasattr(args, 'remove_edge') and args.remove_edge:
        system.config['processing']['remove_edge_grains'] = True
        config_updated = True
    
    if hasattr(args, 'output') and args.output is not None:
        system.config['output']['root_dir'] = args.output
        system.output_root = Path(args.output)
        system.output_root.mkdir(parents=True, exist_ok=True)
        config_updated = True
    
    if hasattr(args, 'performance') and args.performance:
        system.config['processing']['performance_monitoring'] = True
        system.config['output']['save_performance'] = True
        config_updated = True
    
    if hasattr(args, 'debug') and args.debug:
        system.config['output']['save_debug_info'] = True
        system.config['logging']['level'] = 'DEBUG'
        config_updated = True
    
    if hasattr(args, 'quiet') and args.quiet:
        system.config['logging']['show_in_console'] = False
        config_updated = True
    
    if config_updated:
        print("Config updated based on command line arguments")
    
    return system


def print_summary(results: Dict[str, Any], system_name: str = "System"):
    """
    Display processing result summary
    
    Args:
        results: Result dictionary
        system_name: System name
    """
    if not results:
        return
    
    print("\n" + "=" * 60)
    print(f"{system_name} Processing Result Summary")
    print("=" * 60)
    
    if 'total' in results:  # Batch processing results
        print(f"Total images: {results['total']}")
        print(f"Successfully processed: {results['success']}")
        print(f"Failed: {results['failed']}")
        print(f"Total grains: {results['total_grains']}")
        
        if results.get('failed_images'):
            print(f"\nFailed image list saved to report file")
    else:  # Single image result
        print(f"Image: {results.get('image_name', 'Unknown')}")
        print(f"Status: {'Success' if results.get('success') else 'Failed'}")
        
        if results.get('success'):
            print(f"Grain count: {results.get('grains_count', 0)}")
            print(f"Processing time: {results.get('processing_time', 0):.2f}s")
            
            if results.get('scale_detection_success'):
                print(f"Scale factor: {results.get('scale_factor', 'N/A')} um/px")
            
            output_files = results.get('output_files', [])
            print(f"Output files: {len(output_files)}")
            
            if output_files:
                print(f"Generated files:")
                for i, file in enumerate(output_files[:5], 1):
                    file_name = Path(file).name
                    print(f"  {i}. {file_name}")
                
                if len(output_files) > 5:
                    print(f"  ... and {len(output_files)-5} more files")
    
    if results.get('error_message'):
        print(f"Error message: {results.get('error_message')}")
    
    print("=" * 60)


def print_welcome(system_name: str, features: list):
    """
    Display welcome message
    
    Args:
        system_name: System name
        features: Feature list
    """
    print("\n" + "=" * 70)
    print(f"     {system_name} Rock Grain Auto Segmentation System")
    print("=" * 70)
    print("Function: Rock microscopic image grain segmentation")
    
    for feature in features:
        print(f"  * {feature}")
    
    print("=" * 70)
