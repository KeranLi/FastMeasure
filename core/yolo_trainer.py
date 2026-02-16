"""
YOLO Model Fine-tuning Module for FastMeasure

This module provides easy fine-tuning capabilities for YOLO detection models,
similar to segmenteverygrain's U-Net fine-tuning but adapted for YOLO architecture.

Usage:
    from core.yolo_trainer import YOLOFineTuner
    
    # Quick start with auto-generated dataset from interactive annotations
    trainer = YOLOFineTuner()
    trainer.prepare_dataset_from_interactive_results("results/mobilesam/interactive/")
    trainer.train(epochs=50, imgsz=1024)
    
    # Or use existing YOLO-format dataset
    trainer = YOLOFineTuner()
    trainer.train(data_yaml="path/to/dataset.yaml", epochs=100, imgsz=1024)
"""

import os
import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import warnings

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Ultralytics is required for YOLO training. Install with: pip install ultralytics")

import numpy as np
from PIL import Image
import cv2


class YOLOFineTuner:
    """
    YOLO model fine-tuning utility for rock grain detection.
    
    This class provides an easy-to-use interface for fine-tuning YOLO models
    on custom rock grain datasets. It supports:
    - Training from scratch
    - Fine-tuning from existing model
    - Auto-conversion from interactive segmentation results
    - Built-in data augmentation
    - Automatic dataset splitting (train/val/test)
    """
    
    def __init__(self, base_model: str = "yolov8n.pt", device: str = "auto"):
        """
        Initialize YOLO fine-tuner.
        
        Args:
            base_model: Base YOLO model to start from. Options:
                       - "yolov8n.pt" (nano, fastest, default)
                       - "yolov8s.pt" (small)
                       - "yolov8m.pt" (medium)
                       - "yolov8l.pt" (large)
                       - "yolov8x.pt" (extra large, most accurate)
                       - Path to existing .pt file for fine-tuning
            device: Device to use for training ("auto", "cpu", "cuda", or "mps")
        """
        self.base_model = base_model
        self.device = device
        self.model = None
        self.training_results = None
        
        # Create training outputs directory
        self.training_dir = Path("training_outputs")
        self.training_dir.mkdir(exist_ok=True)
        
        print(f"[YOLO Trainer] Initialized with base model: {base_model}")
        print(f"[YOLO Trainer] Device: {device}")
    
    def prepare_dataset_from_interactive_results(
        self,
        interactive_results_dir: str,
        output_dir: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        min_box_size: int = 10
    ) -> str:
        """
        Convert interactive segmentation results to YOLO training dataset.
        
        This function automatically extracts bounding boxes from saved segmentation
        masks and creates a properly formatted YOLO dataset.
        
        Args:
            interactive_results_dir: Directory containing interactive mode results
                                    (e.g., "results/mobilesam/interactive/")
            output_dir: Output directory for the dataset (default: auto-generated)
            train_ratio: Ratio of training data (default: 0.7)
            val_ratio: Ratio of validation data (default: 0.2)
            test_ratio: Ratio of test data (default: 0.1)
            min_box_size: Minimum bounding box size in pixels
            
        Returns:
            Path to the dataset.yaml file
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.training_dir / f"dataset_{timestamp}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[YOLO Trainer] Preparing dataset from: {interactive_results_dir}")
        print(f"[YOLO Trainer] Output directory: {output_dir}")
        
        # Create YOLO directory structure
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        
        for split in ["train", "val", "test"]:
            (images_dir / split).mkdir(parents=True, exist_ok=True)
            (labels_dir / split).mkdir(parents=True, exist_ok=True)
        
        # Process interactive results
        results_dir = Path(interactive_results_dir)
        all_samples = []
        
        for session_dir in results_dir.glob("*/"):
            if not session_dir.is_dir():
                continue
            
            # Find image and mask files
            image_file = None
            mask_file = None
            metadata_file = None
            
            for f in session_dir.glob("*"):
                if f.suffix in [".png", ".jpg", ".tif", ".tiff"]:
                    if "mask" in f.name.lower():
                        mask_file = f
                    elif "original" in f.name.lower() or "image" in f.name.lower():
                        image_file = f
                elif f.suffix == ".json":
                    metadata_file = f
            
            if image_file and mask_file:
                all_samples.append({
                    "image": image_file,
                    "mask": mask_file,
                    "metadata": metadata_file
                })
        
        if not all_samples:
            raise ValueError(f"No valid samples found in {interactive_results_dir}")
        
        print(f"[YOLO Trainer] Found {len(all_samples)} samples")
        
        # Shuffle and split
        np.random.shuffle(all_samples)
        n_total = len(all_samples)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits = {
            "train": all_samples[:n_train],
            "val": all_samples[n_train:n_train + n_val],
            "test": all_samples[n_train + n_val:]
        }
        
        # Process each split
        for split_name, samples in splits.items():
            print(f"[YOLO Trainer] Processing {split_name}: {len(samples)} samples")
            for idx, sample in enumerate(samples):
                self._convert_sample(
                    sample=sample,
                    output_images_dir=images_dir / split_name,
                    output_labels_dir=labels_dir / split_name,
                    sample_idx=f"{split_name}_{idx:04d}",
                    min_box_size=min_box_size
                )
        
        # Create dataset.yaml
        dataset_yaml = output_dir / "dataset.yaml"
        yaml_content = {
            "path": str(output_dir.absolute()),
            "train": str(images_dir / "train"),
            "val": str(images_dir / "val"),
            "test": str(images_dir / "test"),
            "nc": 1,  # number of classes
            "names": ["grain"]  # class names
        }
        
        with open(dataset_yaml, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"[YOLO Trainer] Dataset created successfully!")
        print(f"[YOLO Trainer] Dataset YAML: {dataset_yaml}")
        print(f"[YOLO Trainer] Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
        
        return str(dataset_yaml)
    
    def _convert_sample(
        self,
        sample: Dict,
        output_images_dir: Path,
        output_labels_dir: Path,
        sample_idx: str,
        min_box_size: int
    ):
        """Convert a single sample to YOLO format."""
        # Load image and mask
        image = cv2.imread(str(sample["image"]))
        if image is None:
            return
        
        mask = cv2.imread(str(sample["mask"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return
        
        h, w = image.shape[:2]
        
        # Save image
        img_ext = sample["image"].suffix
        output_image_path = output_images_dir / f"{sample_idx}{img_ext}"
        cv2.imwrite(str(output_image_path), image)
        
        # Extract bounding boxes from mask
        # Each connected component is a separate grain
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        boxes = []
        for i in range(1, num_labels):  # Skip background (0)
            x, y, bw, bh, area = stats[i]
            
            # Filter small boxes
            if bw < min_box_size or bh < min_box_size:
                continue
            
            # Convert to YOLO format (normalized center x, center y, width, height)
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            
            boxes.append((0, cx, cy, nw, nh))  # class_id=0 for grain
        
        # Save labels
        output_label_path = output_labels_dir / f"{sample_idx}.txt"
        with open(output_label_path, "w") as f:
            for box in boxes:
                f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
    
    def train(
        self,
        data_yaml: Optional[str] = None,
        epochs: int = 100,
        imgsz: int = 1024,
        batch: int = 8,
        lr0: float = 0.01,
        lrf: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        warmup_epochs: float = 3.0,
        augment: bool = True,
        patience: int = 50,
        save: bool = True,
        project: Optional[str] = None,
        name: Optional[str] = None,
        exist_ok: bool = False,
        **kwargs
    ) -> Dict:
        """
        Train or fine-tune YOLO model.
        
        Args:
            data_yaml: Path to dataset YAML file. If None, uses COCO pretraining only
            epochs: Number of training epochs
            imgsz: Input image size
            batch: Batch size
            lr0: Initial learning rate
            lrf: Final learning rate factor (lr0 * lrf)
            momentum: SGD momentum / Adam beta1
            weight_decay: Optimizer weight decay
            warmup_epochs: Warmup epochs
            augment: Use data augmentation
            patience: Early stopping patience (epochs without improvement)
            save: Save best checkpoints
            project: Project directory for results
            name: Run name
            exist_ok: Overwrite existing project
            **kwargs: Additional arguments passed to YOLO trainer
            
        Returns:
            Training results dictionary
        """
        # Load model
        print(f"[YOLO Trainer] Loading base model: {self.base_model}")
        self.model = YOLO(self.base_model)
        
        # Default project directory
        if project is None:
            project = str(self.training_dir / "runs")
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"train_{timestamp}"
        
        # Training arguments
        train_args = {
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "lr0": lr0,
            "lrf": lrf,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "augment": augment,
            "patience": patience,
            "save": save,
            "project": project,
            "name": name,
            "exist_ok": exist_ok,
            "device": self.device,
            "verbose": True,
        }
        
        # Add data if provided
        if data_yaml:
            train_args["data"] = data_yaml
        else:
            # If no data provided, use COCO (for pre-training/fine-tuning on generic objects)
            print("[YOLO Trainer] No dataset provided, using COCO8 for demonstration")
            train_args["data"] = "coco8.yaml"
        
        # Add any extra arguments
        train_args.update(kwargs)
        
        print(f"[YOLO Trainer] Starting training...")
        print(f"[YOLO Trainer] Epochs: {epochs}, Image size: {imgsz}, Batch: {batch}")
        if data_yaml:
            print(f"[YOLO Trainer] Dataset: {data_yaml}")
        
        # Run training
        try:
            self.training_results = self.model.train(**train_args)
            
            print(f"[YOLO Trainer] Training completed!")
            print(f"[YOLO Trainer] Best model: {self.training_results.best}")
            
            return {
                "success": True,
                "best_model": str(self.training_results.best),
                "results_dir": str(self.training_results.save_dir),
                "metrics": self.training_results.results_dict if hasattr(self.training_results, 'results_dict') else {}
            }
            
        except Exception as e:
            print(f"[YOLO Trainer] Training failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def quick_finetune(
        self,
        interactive_results_dir: str,
        epochs: int = 50,
        imgsz: int = 1024,
        **kwargs
    ) -> Dict:
        """
        One-step fine-tuning from interactive segmentation results.
        
        This is the simplest way to fine-tune: just point to your interactive
        segmentation results and let it handle the rest.
        
        Args:
            interactive_results_dir: Directory with interactive mode results
            epochs: Training epochs (default: 50, usually enough for fine-tuning)
            imgsz: Image size
            **kwargs: Additional training arguments
            
        Returns:
            Training results
            
        Example:
            >>> trainer = YOLOFineTuner(base_model="./models/best_yolo_20260107.pt")
            >>> results = trainer.quick_finetune("results/mobilesam/interactive/", epochs=50)
            >>> print(f"Fine-tuned model saved to: {results['best_model']}")
        """
        print("=" * 60)
        print("[YOLO Trainer] Quick Fine-tune Mode")
        print("=" * 60)
        
        # Step 1: Prepare dataset
        print("\n[Step 1/2] Preparing dataset from interactive results...")
        dataset_yaml = self.prepare_dataset_from_interactive_results(
            interactive_results_dir=interactive_results_dir
        )
        
        # Step 2: Train
        print("\n[Step 2/2] Starting fine-tuning...")
        results = self.train(
            data_yaml=dataset_yaml,
            epochs=epochs,
            imgsz=imgsz,
            **kwargs
        )
        
        if results["success"]:
            print("\n" + "=" * 60)
            print("[YOLO Trainer] Fine-tuning completed successfully!")
            print(f"[YOLO Trainer] Best model: {results['best_model']}")
            print("=" * 60)
        
        return results
    
    def export_model(self, format: str = "onnx", output_path: Optional[str] = None) -> str:
        """
        Export trained model to different formats.
        
        Args:
            format: Export format ("onnx", "torchscript", "engine", etc.)
            output_path: Output path (default: auto-generated)
            
        Returns:
            Path to exported model
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        print(f"[YOLO Trainer] Exporting model to {format} format...")
        
        exported = self.model.export(format=format)
        
        if output_path:
            shutil.move(exported, output_path)
            exported = output_path
        
        print(f"[YOLO Trainer] Model exported to: {exported}")
        return str(exported)
    
    def validate(self, data_yaml: str) -> Dict:
        """
        Validate model on a dataset.
        
        Args:
            data_yaml: Path to dataset YAML
            
        Returns:
            Validation metrics
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        print(f"[YOLO Trainer] Validating on: {data_yaml}")
        metrics = self.model.val(data=data_yaml)
        
        return {
            "mAP50": metrics.box.map50,
            "mAP50-95": metrics.box.map,
            "precision": metrics.box.mp,
            "recall": metrics.box.mr
        }


def main():
    """CLI entry point for YOLO training."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="YOLO Fine-tuning for Rock Grain Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick fine-tune from interactive results
  python -m core.yolo_trainer --mode quick --input results/mobilesam/interactive/ --epochs 50
  
  # Train with existing dataset
  python -m core.yolo_trainer --mode train --data path/to/dataset.yaml --epochs 100
  
  # Fine-tune from existing model
  python -m core.yolo_trainer --mode quick --input results/interactive/ --base ./models/best.pt --epochs 30
        """
    )
    
    parser.add_argument("--mode", choices=["quick", "train"], default="quick",
                       help="Training mode: 'quick' for auto-dataset creation, 'train' for existing dataset")
    parser.add_argument("--input", "-i", type=str,
                       help="Input directory (interactive results for 'quick' mode)")
    parser.add_argument("--data", "-d", type=str,
                       help="Dataset YAML path (for 'train' mode)")
    parser.add_argument("--base", "-b", type=str, default="yolov8n.pt",
                       help="Base model (yolov8n/s/m/l/x.pt or path to .pt file)")
    parser.add_argument("--epochs", "-e", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=1024,
                       help="Input image size")
    parser.add_argument("--batch", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: auto, cpu, cuda, mps")
    parser.add_argument("--project", "-p", type=str,
                       help="Project directory for results")
    parser.add_argument("--name", "-n", type=str,
                       help="Run name")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = YOLOFineTuner(base_model=args.base, device=args.device)
    
    # Run training
    if args.mode == "quick":
        if not args.input:
            print("Error: --input is required for quick mode")
            return 1
        
        results = trainer.quick_finetune(
            interactive_results_dir=args.input,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name
        )
    else:
        if not args.data:
            print("Error: --data is required for train mode")
            return 1
        
        results = trainer.train(
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name
        )
    
    if results.get("success"):
        print(f"\nTraining successful! Best model: {results['best_model']}")
        return 0
    else:
        print(f"\nTraining failed: {results.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    exit(main())
