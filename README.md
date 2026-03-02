
<!-- 顶部徽章区域 -->
<div align="center">

##  FastMeasure: A cross-platform workflow and software to fast measure the geomtric parameters via deep learning 
</div>

Keran Li<sup>a</sup>, Wen Lai<sup>b,*</sup>

<sup>a</sup>State Key Laboratory of Critical Earth Material Cycling and Mineral Deposits, Frontiers Science Center for Critical Earth Material Cycling, School of Earth Sciences and Engineering, Nanjing University, Nanjing, 210023, China

<sup>b</sup>Gannan Normal University

<sup>*</sup>Corresponding authors

---

## Project Overview

FastMeasure is a professional tool for processing rock microscopic images, automatically detecting and segmenting grains. This project is inspired by and builds upon [segmenteverygrain](https://github.com/zsylvester/segmenteverygrain) by Zoltán Sylvester. FastMeasure introduces YOLO-based detection, multiple SAM variants, automatic scale detection, and enhanced geometric analysis.. Based on deep learning technology, the system supports two model combinations: **YOLO+FastSAM** and **YOLO+MobileSAM**, combined with intelligent scale bar detection and rich geometric parameter calculation, enabling precise extraction of grain information from rock microscopic images and generation of complete statistical analysis reports.

### Inspiration

This project is inspired by and builds upon **[segmenteverygrain](https://github.com/zsylvester/segmenteverygrain)** by Zoltán Sylvester. We appreciate the excellent work done by the segmenteverygrain team in developing a U-Net + SAM based grain segmentation solution for geomorphology and sedimentary geology research.

While segmenteverygrain pioneered the use of SAM for grain segmentation, FastMeasure takes a different approach and introduces several enhancements:

| Feature | segmenteverygrain | FastMeasure |
|---------|-------------------|-------------|
| Detection Model | U-Net (patch-based CNN) | **YOLO (real-time object detection)** |
| SAM Variants | SAM 2.1 only | **FastSAM + MobileSAM** |
| Processing Speed | ~2.5 min for 3MP image | **~0.3s for FastSAM (GPU)** |
| Scale Calibration | Manual (Shift+drag) | **Automatic + Manual calibration** |
| Geometric Parameters | Basic shape metrics | **10+ parameters including fractal dimension, angularity** |
| Interactive Mode | Jupyter notebook based | **Standalone GUI with unified key controls** |
| Batch Processing | Notebook-based | **Command-line batch processing** |
| Model Fine-tuning | U-Net (TensorFlow) | **YOLO (Ultralytics, easier)** |
| Training Data | Manual annotation | **Auto from interactive results** |
| Code Structure | Notebook + modules | **Modular core library with CLI** |

The system supports three usage modes:
- **Auto Processing Mode**: YOLO detection + SAM auto segmentation
- **Batch Processing Mode**: Batch processing of all images in a folder
- **Interactive Mode**: Manual point selection for fine segmentation via GUI

### Model Fine-tuning

Similar to segmenteverygrain's U-Net fine-tuning, FastMeasure supports **YOLO model fine-tuning** to improve detection accuracy on your specific rock types:

```bash
# Quick fine-tune from interactive segmentation results
python utils/train_yolo.py --mode quick --input results/mobilesam/interactive/

# The fine-tuned model can then be used for better detection
```

See [Model Training Guide](#model-training) below for detailed instructions.

## Core Features

### 1. Dual Model Support
| Model Combination | Features | Applicable Scenarios |
|------------------|----------|---------------------|
| YOLO + FastSAM | Fast, lightweight | Large batch quick processing |
| YOLO + MobileSAM | High precision, supports interaction | High precision requirements, interactive annotation |

### 2. Scale Bar Detection
- Automatically recognize red scale bar at bottom-right corner of images
- Calculate conversion factor from pixels to actual microns
- Support custom scale bar length configuration

### 3. Grain Segmentation and Labeling
- Automatic grain detection and segmentation
- Intelligent grain numbering and area labeling
- Support custom labeling styles (font, color, outline, etc.)

### 4. Geometric Parameter Calculation
The system can calculate various grain geometric parameters:
- **Basic Parameters**: Area, perimeter, centroid coordinates, bounding rectangle
- **Shape Parameters**: Circularity, Aspect Ratio, Rectangularity
- **Structural Parameters**: Compactness, Roundness, Convexity
- **Advanced Parameters**: Fractal Dimension, Angularity

### 5. Flexible Configuration System
- `configs/fastsam.yaml` / `configs/mobilesam.yaml`: Main configuration files (model paths, processing parameters, output settings)
- `configs/geometry.yaml`: Geometric parameter configuration file (custom CSV export fields)

## Installation

### Environment Requirements
- Python 3.8+
- PyTorch
- CUDA (recommended for GPU acceleration)

### Installation Steps

1. Clone this repository to local:
    ```bash
    git clone https://github.com/KeranLi/FastMeasure.git
    cd FastMeasure
    ```

2. Create and activate virtual environment (recommend using `conda`):
    ```bash
    conda create -n rockseg python=3.8
    conda activate rockseg
    ```

3. Install dependencies:
    ```bash
    pip install torch torchvision opencv-python pandas matplotlib numpy pyyaml ultralytics shapely scikit-image pillow
    
    # MobileSAM interactive mode requires additional installation
    pip install mobile_sam
    ```

4. Prepare model files:
    - YOLO model: `./models/best_yolo_20260107.pt` (FastSAM workflow) or `./models/best.pt` (MobileSAM workflow)
    - FastSAM model: `./models/FastSAM-s.pt`
    - MobileSAM model: `./models/mobile_sam.pt`

## Usage Guide

### Unified Entry Point (Recommended)

The project provides a unified entry script `run.py` to start FastSAM or MobileSAM:

```bash
# FastSAM processing
python run.py fastsam --input path/to/image.tif

# MobileSAM batch processing
python run.py mobilesam --input path/to/folder --batch

# Interactive mode
python run.py mobilesam --interactive
```

### FastSAM Processing Workflow

#### 1. Process Single Image
```bash
python run_fastsam.py --input path/to/image.tif
# Or use unified entry
python run.py fastsam --input path/to/image.tif
```

#### 2. Batch Process Folder
```bash
python run_fastsam.py --input path/to/folder --batch
# Or use unified entry
python run.py fastsam --input path/to/folder --batch
```

#### 3. Use Custom Configuration
```bash
python run_fastsam.py --config configs/fastsam.yaml --input image.tif
```

#### 4. Adjust Processing Parameters
```bash
python run_fastsam.py --input image.tif --conf 0.3 --min-area 50 --output my_results
```

### MobileSAM Processing Workflow

#### 1. Terminal Interactive Mode (Recommended for Beginners)
```bash
python run_mobilesam.py
# Or use unified entry
python run.py mobilesam
```
Follow prompts to select processing mode and input parameters.

#### 2. Process Single Image
```bash
python run_mobilesam.py --input path/to/image.tif
# Or use unified entry
python run.py mobilesam --input path/to/image.tif
```

#### 3. Batch Process Folder
```bash
python run_mobilesam.py --input path/to/folder --batch
# Or use unified entry
python run.py mobilesam --input path/to/folder --batch
```

#### 4. GUI Interactive Segmentation
```bash
python run_mobilesam.py --interactive
# Or use unified entry
python run.py mobilesam --interactive
```

### Interactive Mode Operation Guide

| Key/Operation | Function |
|--------------|----------|
| Left click | Add foreground point (segmentation target) |
| Right click | Add background point (exclusion area) |
| `X` | Delete last grain |
| `D` | Delete all grains |
| `S` | Save results |
| `Shift+S` | Quick save complete results |
| `C` | Clear all point marks |
| `R` | Reset interface |
| `M` | Manual scale calibration (measure known length) |
| `H` | Show help |
| `Q` | Quit |

#### Manual Scale Calibration

When automatic scale bar detection fails, you can manually calibrate the scale:

1. Press `M` key to enter scale calibration mode
2. Click the **start point** of a known-length line (e.g., scale bar, ruler)
3. Click the **end point** of the line
4. Enter the **actual length in microns** when prompted
5. The system will calculate and store the scale factor (um/px)

This allows you to use any known-length feature in the image for calibration.

### Platform Notes

#### macOS Compatibility
The system is fully compatible with macOS. However, please note:
- First run may be slower due to model loading
- Interactive mode requires a display (not supported on remote SSH without X11)
- File dialogs run on main thread to ensure macOS compatibility

## Model Training

FastMeasure includes a **YOLO Fine-tuning Module** (`utils/train_yolo.py`) that allows you to improve detection accuracy on your specific rock types, similar to segmenteverygrain's U-Net fine-tuning capability.

### Why Fine-tune?

- **Better Accuracy**: YOLO models trained on generic datasets may miss specific grain types in your samples
- **Adapt to New Rock Types**: Fine-tune on your own thin-section images for best results
- **Iterative Improvement**: Use interactive mode results as training data

### Quick Start

The easiest way to fine-tune is using your interactive segmentation results:

```bash
# Step 1: Generate some training data using interactive mode
python run.py mobilesam --interactive
# Segment several images and save the results

# Step 2: Fine-tune YOLO using those results
python utils/train_yolo.py --mode quick --input results/mobilesam/interactive/ --epochs 50

# Step 3: Use the fine-tuned model
cp training_outputs/runs/train_*/weights/best.pt ./models/my_finetuned_yolo.pt
# Update configs/fastsam.yaml: yolo: "./models/my_finetuned_yolo.pt"
```

### Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Training mode: `quick` (from interactive results) or `train` (from dataset) | `quick` |
| `--input` | Directory containing interactive mode results | Required for `quick` |
| `--data` | Path to YOLO-format dataset YAML | Required for `train` |
| `--base` | Base model: `yolov8n/s/m/l/x.pt` or path to `.pt` file | `yolov8n.pt` |
| `--epochs` | Number of training epochs | 50 |
| `--imgsz` | Input image size | 1024 |
| `--batch` | Batch size (reduce if out of memory) | 8 |
| `--device` | Device: `auto`, `cpu`, `cuda`, `mps` | `auto` |

### Advanced Usage

```bash
# Fine-tune from existing model with more epochs
python utils/train_yolo.py --mode quick \
                     --input results/mobilesam/interactive/ \
                     --base ./models/best_yolo_20260107.pt \
                     --epochs 100 \
                     --imgsz 1024

# Use larger model for better accuracy (slower)
python utils/train_yolo.py --mode quick \
                     --input results/interactive/ \
                     --base yolov8m.pt \
                     --epochs 50

# Train with custom YOLO-format dataset
python utils/train_yolo.py --mode train \
                     --data ./my_grain_dataset/dataset.yaml \
                     --epochs 200
```

### Dataset Format (for Custom Training)

If you have existing annotations, you can create a YOLO-format dataset:

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

`dataset.yaml` format:
```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test
nc: 1
names: ['grain']
```

## Configuration File Guide

### Main Configuration File (`configs/fastsam.yaml` / `configs/mobilesam.yaml`)

```yaml
# Model path configuration
model_paths:
  yolo: "./models/best_yolo_20260107.pt"    # YOLO model path
  fastsam: "./models/FastSAM-s.pt"          # FastSAM model path
  device: "cpu"                              # Running device: cpu or cuda

# Scale bar detection configuration
scale_detection:
  enabled: true
  known_length_um: 1000.0                    # Scale bar actual length (microns)

# Processing parameter configuration
processing:
  yolo_confidence: 0.25                      # YOLO detection confidence threshold
  min_area: 30                               # Minimum grain area (pixels)
  remove_edge_grains: false                  # Whether to remove edge grains

# Output configuration
output:
  root_dir: "results"                        # Result output directory
  save_visualization: true                   # Save visualization results
  save_statistics: true                      # Save CSV statistics file
  save_summary: true                         # Save JSON summary
```

**Note**: 
- `configs/fastsam.yaml` is used for FastSAM mode (default: CPU)
- `configs/mobilesam.yaml` is used for MobileSAM mode (default: CPU)
- Change `device` to `cuda` if you have NVIDIA GPU and CUDA installed

### Geometric Parameter Configuration File (`configs/geometry.yaml`)

```yaml
grain_statistics_csv:
  enabled: true
  # Columns finally written to CSV (output in this order)
  keep_columns:
    - label
    - area
    - perimeter
    - circularity
    - aspect_ratio
    - compactness
    - roundness
    - area_um2
    - diameter_um
```

## Output File Guide

After processing is complete, the system generates the following files in the output directory:

| File Name | Description |
|-----------|-------------|
| `segmentation_result.png` | Segmentation result visualization (with grain contours) |
| `segmentation_labeled.png` | Labeled result image (with grain numbers and areas) |
| `segmentation_mask.png` | Binary segmentation mask image |
| `grain_statistics.csv` | Grain statistics data table |
| `summary.json` | Processing summary information (JSON format) |
| `performance.json` | Performance statistics information |

### Unified Output Directory Structure

All results are now organized under a unified `results/` directory:

```
results/
├── fastsam/                    # FastSAM results
│   ├── auto/                   # Automatic processing results
│   │   └── [image_name]/
│   │       ├── segmentation_result.png
│   │       ├── segmentation_labeled.png
│   │       ├── segmentation_mask.png
│   │       ├── grain_statistics.csv
│   │       └── summary.json
│   └── interactive/            # Interactive processing results
│       └── [timestamp]/
│           └── ...
├── mobilesam/                  # MobileSAM results
│   ├── auto/                   # Automatic processing results
│   └── interactive/            # Interactive processing results
├── logs/                       # Unified log directory
│   ├── fastsam/                # FastSAM logs
│   └── mobilesam/              # MobileSAM logs
└── temp/                       # Temporary files and cache
```

**Benefits:**
- All results in one place - easy to find and manage
- Clear separation between modes (FastSAM/MobileSAM) and types (auto/interactive)
- Unified logs for easier debugging
- No scattered result folders in project root

## Project Structure

```
.
├── run.py                      # Unified entry script (new)
├── run_fastsam.py              # FastSAM startup script
├── run_mobilesam.py            # MobileSAM startup script (supports interactive mode)
├── utils/                      # Utility scripts folder
│   ├── train_yolo.py           # YOLO model training/fine-tuning script
│   ├── gui_launcher.py         # GUI launcher for desktop app
│   ├── file_dialog.py          # Cross-platform file dialog utilities
│   ├── grain_marker.py         # Grain labeling module
│   └── scale_detector.py       # Scale bar detection module
├── mobilesam_interactive.py    # MobileSAM standalone interactive tool
├── configs/fastsam.yaml            # FastSAM configuration file
├── configs/mobilesam.yaml      # MobileSAM configuration file
├── configs/geometry.yaml       # Geometric parameter configuration file
│
├── core/                       # Core module (new)
│   ├── __init__.py             # Core module initialization
│   ├── seg_tools.py            # Shared tool functions
│   ├── seg_optimize.py         # Shared segmentation optimization
│   ├── cli_base.py             # Shared CLI functions
│   ├── scale_calibration.py    # Manual scale calibration (new)
│   └── yolo_trainer.py         # YOLO fine-tuning module (new)
│
├── fastsam/                    # FastSAM module
│   ├── rock_fastsam_system.py  # FastSAM main system
│   ├── yolo_fastsam.py         # YOLO+FastSAM pipeline
│   ├── seg_engine.py           # Segmentation engine
│   ├── seg_optimize.py         # Segmentation optimization (compatibility wrapper)
│   └── seg_tools.py            # Tool functions (compatibility wrapper)
│
├── mobilesam/                  # MobileSAM module
│   ├── rock_mobilesam_system.py  # MobileSAM main system
│   ├── yolo_mobilesam.py         # YOLO+MobileSAM pipeline
│   ├── mobile_sam_engine.py      # MobileSAM engine
│   ├── seg_optimize.py           # Segmentation optimization (compatibility wrapper)
│   └── seg_tools.py              # Tool functions (compatibility wrapper)
│
├── geometry/                   # Geometric parameter calculation module
│   ├── grain_metric.py         # Grain shape parameter calculation
│   ├── config_loader.py        # Config loader
│   └── export_csv.py           # CSV export utility
│
├── configs/                    # Configuration files folder
│   ├── fastsam.yaml            # FastSAM configuration
│   └── geometry.yaml           # Geometric parameters configuration
├── models/                     # Model files directory
├── results/                    # Default output directory
└── Boulder_20260107/           # Test data example
```

## Performance Reference

Performance tests based on RTX 3060 graphics card:

| Model | GPU Inference | CPU Inference | Speed Comparison |
|-------|--------------|---------------|------------------|
| FastSAM | ~77ms | ~294ms | CPU is ~4x GPU |
| MobileSAM | ~3.7s | ~101s | CPU is ~26x GPU |

**Recommendation**: For large batch processing, GPU acceleration is recommended; for small batches or testing, CPU mode can be used.

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework |
| `ultralytics` | YOLOv8 and SAM models |
| `opencv-python` | Image processing and scale bar detection |
| `pandas` | Data processing and statistics |
| `matplotlib` | Result visualization |
| `numpy` | Numerical computation |
| `pyyaml` | Configuration file parsing |
| `shapely` | Geometric calculation |
| `scikit-image` | Image processing tools |
| `mobile_sam` | MobileSAM library (required for interactive mode) |

## FAQ

**Q: What to do if scale bar detection fails?**  
A: Check if there is a clear red scale bar at the bottom-right corner of the image, or adjust color threshold parameters like `red_lower1/red_upper1` in the configuration file.

**Q: How to adjust detection sensitivity?**  
A: Modify the `yolo_confidence` parameter in the configuration file (smaller values mean more sensitive detection but may introduce noise).

**Q: Interactive mode cannot start GUI?**  
A: Ensure the system has GUI support, or try setting the environment variable `MPLBACKEND=TkAgg`.

## Change Log

See [CHANGELOG.md](CHANGELOG.md) for detailed update content of each project version.

## Contributing

Contributions are welcome! If you have improvement suggestions or find issues, you can contribute code by submitting an `issue` or `pull request`.

## Acknowledgments

This project builds upon the excellent work of **[segmenteverygrain](https://github.com/zsylvester/segmenteverygrain)** by Zoltán Sylvester and colleagues. We thank them for pioneering the application of SAM in sedimentary grain segmentation and for making their work open-source.

### Code Migration Notice

FastMeasure has migrated core segmentation functionality from segmenteverygrain to `core/segment_core.py`, making the project fully independent. The segmenteverygrain source code has been removed from this repository but remains available in Git history.

**Migrated functions** (now in `core/segment_core.py`):

- `create_labeled_image()` - Create labeled grain masks
- `plot_image_w_colorful_grains()` - Visualize grains with colors
- `plot_grain_axes_and_centroids()` - Plot grain orientation axes
- `find_connected_components()` - Detect overlapping grains
- `merge_overlapping_polygons()` - Merge overlapping segmentations
- `collect_polygon_from_mask()` - Extract polygons from masks
- `load_image()` - Image loading utilities
- `polygons_to_grains()` - Convert polygons to grain objects
- `save_grains()` - Save grain data

**To view original segmenteverygrain code:**
```bash
git show HEAD~1:segmenteverygrain/
# or restore temporarily:
git checkout HEAD~1 -- segmenteverygrain/
```

Key improvements in FastMeasure:
- **YOLO-based Detection**: Replaced patch-based U-Net with YOLO for real-time grain detection
- **Multiple SAM Backends**: Support for both FastSAM (speed) and MobileSAM (precision)
- **Automatic Scale Detection**: Intelligent red scale bar recognition at image corners
- **Enhanced Geometric Analysis**: 10+ grain shape parameters including fractal dimension
- **Unified Architecture**: Modular core library with command-line interface
- **Cross-Platform**: Full macOS and Linux/Windows support

## Building Standalone Executable

FastMeasure can be packaged as a standalone executable for Windows, allowing users to run it without installing Python.

### Prerequisites

```bash
# Install PyInstaller
pip install pyinstaller
```

### Build Instructions

#### Method 1: Using Build Script (Recommended)

```bash
# Run the build script
python build_exe.py
```

This will:
1. Clean previous builds
2. Package all Python dependencies
3. Include model configs and core modules
4. Create `dist/FastMeasure/` folder with executable

#### Method 2: Manual Build

```bash
# Build one-directory (recommended, faster startup)
pyinstaller --name FastMeasure \
            --windowed \
            --onedir \
            --add-data "core;core" \
            --add-data "fastsam;fastsam" \
            --add-data "mobilesam;mobilesam" \
            --add-data "geometry;geometry" \
            --add-data "configs;configs" \
            --hidden-import ultralytics \
            --hidden-import torch \
            gui_launcher.py
```

### Distribution

After building:

```
dist/
└── FastMeasure/
    ├── FastMeasure.exe      # Main executable
    ├── models/              # Place model files here
    ├── results/             # Output directory
    └── _internal/           # Python libraries
```

**Before distributing:**
1. Download model files (see [Model Files](#model-files))
2. Place in `dist/FastMeasure/models/`
3. Zip the entire `FastMeasure/` folder
4. Share the zip file with users

### Creating Windows Installer (Optional)

1. Install [Inno Setup](https://jrsoftware.org/isinfo.php)
2. Open `installer.iss` in Inno Setup Compiler
3. Build to create `FastMeasure_Setup.exe`

### Notes

- **Executable size**: ~500MB-1GB (includes Python + PyTorch)
- **Startup time**: First launch may take 10-30 seconds (model loading)
- **CPU mode**: The executable defaults to CPU mode for compatibility
- **Model files**: Not included in build (too large), must be downloaded separately

## License

[LICENSE](LICENSE)
