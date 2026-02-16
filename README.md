# FastMeasure - Rock Grain Auto Segmentation System

## Project Overview

FastMeasure is a professional tool for processing rock microscopic images, automatically detecting and segmenting grains. Based on deep learning technology, the system supports two model combinations: **YOLO+FastSAM** and **YOLO+MobileSAM**, combined with intelligent scale bar detection and rich geometric parameter calculation, enabling precise extraction of grain information from rock microscopic images and generation of complete statistical analysis reports.

The system supports three usage modes:
- **Auto Processing Mode**: YOLO detection + SAM auto segmentation
- **Batch Processing Mode**: Batch processing of all images in a folder
- **Interactive Mode**: Manual point selection for fine segmentation via GUI

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
- `config.yaml` / `config_mobilesam.yaml`: Main configuration files (model paths, processing parameters, output settings)
- `geometry_config.yaml`: Geometric parameter configuration file (custom CSV export fields)

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
python run_fastsam.py --config custom_config.yaml --input image.tif
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

## Configuration File Guide

### Main Configuration File (`config.yaml` / `config_mobilesam.yaml`)

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
- `config.yaml` is used for FastSAM mode (default: CPU)
- `config_mobilesam.yaml` is used for MobileSAM mode (default: CPU)
- Change `device` to `cuda` if you have NVIDIA GPU and CUDA installed

### Geometric Parameter Configuration File (`geometry_config.yaml`)

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

## Project Structure

```
.
├── run.py                      # Unified entry script (new)
├── run_fastsam.py              # FastSAM startup script
├── run_mobilesam.py            # MobileSAM startup script (supports interactive mode)
├── mobilesam_interactive.py    # MobileSAM standalone interactive tool
├── config.yaml                 # FastSAM configuration file
├── config_mobilesam.yaml       # MobileSAM configuration file
├── geometry_config.yaml        # Geometric parameter configuration file
│
├── core/                       # Core module (new)
│   ├── __init__.py             # Core module initialization
│   ├── seg_tools.py            # Shared tool functions
│   ├── seg_optimize.py         # Shared segmentation optimization
│   ├── cli_base.py             # Shared CLI functions
│   └── scale_calibration.py    # Manual scale calibration (new)
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
├── scale_detector.py           # Scale bar detection module
├── grain_marker.py             # Grain labeling module
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

## License

[LICENSE](LICENSE)
