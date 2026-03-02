#!/usr/bin/env python
"""
YOLO Model Training/Fine-tuning Script for FastMeasure

This script provides an easy way to fine-tune YOLO models on your rock grain data,
similar to segmenteverygrain's U-Net fine-tuning capability.

Quick Start:
    # Fine-tune from interactive segmentation results
    python train_yolo.py --mode quick --input results/mobilesam/interactive/
    
    # Train with existing YOLO-format dataset
    python train_yolo.py --mode train --data path/to/dataset.yaml
    
    # Fine-tune from existing model with custom settings
    python train_yolo.py --mode quick --input results/interactive/ \\
                         --base ./models/best_yolo_20260107.pt \\
                         --epochs 50 --imgsz 1024

The fine-tuned model can then be used in FastMeasure for improved detection.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.yolo_trainer import YOLOFineTuner


def main():
    parser = argparse.ArgumentParser(
        description="Train or Fine-tune YOLO for Rock Grain Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick fine-tune from interactive results (recommended)
  python train_yolo.py --mode quick --input results/mobilesam/interactive/
  
  # Train with more epochs
  python train_yolo.py --mode quick --input results/interactive/ --epochs 100
  
  # Use larger model for better accuracy
  python train_yolo.py --mode quick --input results/interactive/ --base yolov8m.pt
  
  # Fine-tune from your existing model
  python train_yolo.py --mode quick --input results/interactive/ \\
                       --base ./models/best_yolo_20260107.pt --epochs 50
  
  # Train with custom dataset (YOLO format)
  python train_yolo.py --mode train --data ./my_dataset/dataset.yaml
        """
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["quick", "train"],
        default="quick",
        help="Training mode: 'quick' auto-creates dataset from interactive results, 'train' uses existing dataset"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input directory containing interactive mode results (for 'quick' mode)"
    )
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        help="Path to dataset YAML file (for 'train' mode)"
    )
    
    parser.add_argument(
        "--base", "-b",
        type=str,
        default="yolov8n.pt",
        help="Base model: yolov8n/s/m/l/x.pt or path to existing .pt file (default: yolov8n.pt)"
    )
    
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Input image size (default: 1024)"
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8, reduce if OOM)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda, mps (default: auto)"
    )
    
    parser.add_argument(
        "--project", "-p",
        type=str,
        default="training_outputs",
        help="Project directory for training outputs (default: training_outputs)"
    )
    
    parser.add_argument(
        "--name", "-n",
        type=str,
        help="Run name (default: auto-generated with timestamp)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  FastMeasure YOLO Training/Fine-tuning")
    print("=" * 70)
    
    # Validate arguments
    if args.mode == "quick" and not args.input:
        print("\nError: --input is required for 'quick' mode")
        print("Example: python train_yolo.py --mode quick --input results/mobilesam/interactive/")
        return 1
    
    if args.mode == "train" and not args.data:
        print("\nError: --data is required for 'train' mode")
        print("Example: python train_yolo.py --mode train --data ./my_dataset/dataset.yaml")
        return 1
    
    # Initialize trainer
    print(f"\n[Config] Base model: {args.base}")
    print(f"[Config] Device: {args.device}")
    print(f"[Config] Epochs: {args.epochs}")
    print(f"[Config] Image size: {args.imgsz}")
    print(f"[Config] Batch size: {args.batch}")
    
    trainer = YOLOFineTuner(base_model=args.base, device=args.device)
    
    # Run training
    if args.mode == "quick":
        print(f"\n[Mode] Quick fine-tuning from: {args.input}")
        results = trainer.quick_finetune(
            interactive_results_dir=args.input,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.lr,
            project=args.project,
            name=args.name
        )
    else:
        print(f"\n[Mode] Training with dataset: {args.data}")
        results = trainer.train(
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.lr,
            project=args.project,
            name=args.name
        )
    
    # Report results
    print("\n" + "=" * 70)
    if results.get("success"):
        print("  Training Completed Successfully!")
        print("=" * 70)
        print(f"\nBest model saved to: {results['best_model']}")
        print(f"Results directory: {results['results_dir']}")
        
        # Print usage instructions
        print("\n" + "-" * 70)
        print("  Next Steps:")
        print("-" * 70)
        print("1. Copy the best model to your models directory:")
        print(f"   cp {results['best_model']} ./models/my_finetuned_model.pt")
        print("\n2. Update config.yaml to use your fine-tuned model:")
        print("   model_paths:")
        print("     yolo: \"./models/my_finetuned_model.pt\"")
        print("\n3. Run FastMeasure with your new model:")
        print("   python run.py fastsam --input your_image.tif")
        
        return 0
    else:
        print("  Training Failed!")
        print("=" * 70)
        print(f"\nError: {results.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
