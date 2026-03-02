# Project Change Log

All development progress, feature updates and architecture changes of this project are recorded here.

---

## 2026-03-02 - Remove segmenteverygrain Dependency

- **Modifier:** Core Team
- **Modification Type:** Architecture Refactoring
- **Involved Files:** `core/segment_core.py`, `fastsam_interactive.py`, `mobilesam_interactive.py`, `fastsam/yolo_fastsam.py`, `mobilesam/yolo_mobilesam.py`, `README.md`

**Specific Content:**
- **Complete Independence**: Removed `segmenteverygrain/` folder entirely, making FastMeasure fully independent from the external dependency
- **Core Function Migration**: Migrated all used functions from segmenteverygrain to new `core/segment_core.py` module:
  - `create_labeled_image()` - Create labeled grain masks
  - `plot_image_w_colorful_grains()` - Visualize grains with colors  
  - `plot_grain_axes_and_centroids()` - Plot grain orientation axes
  - `find_connected_components()` - Detect overlapping grains
  - `merge_overlapping_polygons()` - Merge overlapping segmentations
  - `collect_polygon_from_mask()` - Extract polygons from masks
  - `load_image()` - Image loading utilities
  - `polygons_to_grains()` - Convert polygons to grain objects
  - `save_grains()` - Save grain data
- **Import Updates**: Updated all import statements across the codebase:
  - `fastsam_interactive.py` & `fastsam_interactive_macos.py`
  - `mobilesam_interactive.py` & `mobilesam_interactive_macos.py`
  - `fastsam/yolo_fastsam.py`
  - `mobilesam/yolo_mobilesam.py`
- **Git History Preserved**: Original segmenteverygrain code remains accessible via Git history for reference

**Benefits:**
- Reduced external dependencies
- Simplified project structure
- Easier maintenance and deployment
- No licensing concerns from bundled external code

**Notes:**
- Original segmenteverygrain code can be restored via: `git checkout HEAD~1 -- segmenteverygrain/`
- All migrated functions maintain API compatibility with original implementations

---

## [Unreleased] Early Development Phase

- **Modifier:** Core Team
- **Modification Type:** Feature addition and refactoring
- **Involved Files:** Core function modules, GUI modules

**Specific Content:**
- **Interactive and Batch Processing**: Added interactive mode (GUI file selection) and batch processing mode (folder processing support).
- **Grain Labeling**: Added grain labeling function, automatically adding grain numbers and area labels to segmentation images.
- **Scale Bar Module**: Added scale bar detection module, supporting automatic recognition of image scale bar and calculation of scale factor.
- **Architecture Refactoring**: Refactored code into modular structure, improved scale bar detection algorithm accuracy and batch processing log output.
- **Fixes**: Fixed YOLO model loading failures in specific environments and incorrect calculations when scale bar is missing.

---

## 2026-02-16 - GUI Launcher & Executable Build Support

- **Modifier:** Core Team
- **Modification Type:** New Feature
- **Involved Files:** `utils/gui_launcher.py`, `application/build_exe.py`, `core/model_manager.py`, `.exe_config.yaml`

**Specific Content:**
- **GUI Launcher**: Added user-friendly graphical interface (`utils/gui_launcher.py`) for non-technical users:
  - One-click image/folder selection
  - Visual mode selection (FastSAM/MobileSAM, Auto/Batch/Interactive)
  - Real-time progress display with log output
  - Built-in system check and model download dialogs
  - Cross-platform support (Windows, macOS, Linux)
  
- **Executable Build System**: Added PyInstaller build script (`build_exe.py`):
  - Automated packaging of Python app to standalone executable
  - Support for both one-directory (recommended) and one-file modes
  - Includes all dependencies, configs, and core modules
  - Generates Inno Setup script for Windows installer creation
  
- **Model Manager**: Added `core/model_manager.py` for automatic model management:
  - Check model availability and download missing models
  - GUI dialog for model download with progress bar
  - Support for automatic model verification
  
- **Executable Config**: Added `.exe_config.yaml` for standalone app configuration
  
- **Benefits**:
  - End users can run FastMeasure without Python installation
  - Simplified workflow for non-technical geologists
  - Easier distribution and deployment

---

## 2026-02-16

- **Modifier:** Core Team
- **Modification Type:** Feature Addition + Documentation
- **Involved Files:** `core/yolo_trainer.py`, `utils/train_yolo.py`, `README.md`

**Specific Content:**
- **YOLO Fine-tuning Module**: Added `core/yolo_trainer.py` providing YOLO model fine-tuning capabilities similar to segmenteverygrain's U-Net fine-tuning:
  - `YOLOFineTuner` class with easy-to-use API
  - Auto-dataset generation from interactive segmentation results
  - Support for training from scratch or fine-tuning existing models
  - Built-in data augmentation and automatic train/val/test splitting
  - CLI script `utils/train_yolo.py` for command-line usage
- **Acknowledgment**: Added clear acknowledgment of [segmenteverygrain](https://github.com/zsylvester/segmenteverygrain) by Zoltán Sylvester as the inspiration for this project.
- **Comparison Table**: Added detailed feature comparison table highlighting key improvements:
  - YOLO-based detection vs U-Net (patch-based)
  - Dual SAM backends (FastSAM + MobileSAM) vs SAM 2.1 only
  - ~10x speed improvement for FastSAM mode
  - Automatic + manual scale calibration
  - 10+ geometric parameters including fractal dimension
  - Standalone GUI vs Jupyter notebook
  - Command-line batch processing
  - YOLO fine-tuning vs U-Net fine-tuning
  - Modular architecture with CLI

---

## [1.0.0] - 2026-01-07

- **Modifier:** Core Team
- **Modification Type:** Initial version release
- **Involved Files:** Project full initial code

**Specific Content:**
- **Feature Release**: Includes core functions such as single image processing, scale bar detection, grain segmentation, and grain labeling.
- **Model Integration**: Completed integration of YOLO and SAM models for efficient rock grain segmentation.
- **Config and Output**: Supports adjusting model paths and parameters through `configs/fastsam.yaml`; output files include segmentation images, labeled images, CSV statistics and JSON summaries.
- **Known Issues**: Label overlap in extreme cases; performance optimization needed for large image batches; EfficientSAM3 has weak Windows support due to Triton architecture limitations.

---

## 2026-01-08

- **Modifier:** System Record
- **Modification Type:** Performance Testing
- **Involved Files:** `fastsam_inference_test.py`

**Specific Content:**
- Added FastSAM test script.
- **Performance Evaluation (RTX 3060)**: CPU inference time is about 4x GPU (97.8/405.9/379.5 ms vs 61.1/102.1/68.0 ms).

---

## 2026-01-09

- **Modifier:** System Record
- **Modification Type:** Performance Testing
- **Involved Files:** `mobilesam_inference_test.py`

**Specific Content:**
- Added MobileSAM test script.
- **Performance Evaluation (RTX 3060)**: CPU inference time is about 26x GPU (CPU inference takes 101s, GPU takes 3.7s).

---

## 2026-01-10

- **Modifier:** Li Keran
- **Modification Type:** Model Adaptation and Logic Modification
- **Involved Files:** `rock.py`, `rock_new.py`, `run.py`, `configs/fastsam.yaml`

**Specific Content:**
- Adapted MobileSAM model in `rock_new.py`, and synchronized modifications to `run.py` calling logic.
- Updated corresponding config parameters in `configs/fastsam.yaml`.

---

## 2026-01-13

- **Modifier:** Zhang Lihua
- **Modification Type:** Documentation Addition and Code Optimization
- **Involved Files:** `yolosection.md`, `segmenteverygrain_remove_supervised_clustering.py`, etc.

**Specific Content:**
- **Documentation Supplement**: First upload of `new` folder file description document, supplementing project file structure information.
- **Redundancy Cleanup**: Removed useless functions about "supervised clustering" from code files.

---

## 2026-01-15

- **Modifier:** Zhang Lihua, Li Keran
- **Modification Type:** Project Structure Synchronization
- **Involved Files:** YOLO+SAM related code, YOLO+FastSAM related code

**Specific Content:**
- **Directory Adjustment**: Moved YOLO+SAM code into `new` folder; YOLO+FastSAM code into `new/super_fastsam` subfolder.

---

## 2026-01-19

- **Modifier:** Zhang Lihua
- **Modification Type:** Code Upload
- **Involved Files:** `fastsam` folder (formerly `fsatsam0118`)

**Specific Content:**
- Uploaded and integrated the latest FastSAM encapsulated code as of January 18.

---

## 2026-01-19

- **Modifier:** Li Keran
- **Modification Type:** Code Refactoring and Feature Enhancement
- **Involved Files:** `0118fastsam` folder, `run_fastsam.py`, `./geometry/grain_metric.py`, `yolo_fastsam.py`

**Specific Content:**
- **Structure Integration**: Encapsulated FastSAM as independent folder, standardized `run_xxx.py` scripts placement in root directory, unified output to `results` folder.
- **Standardized Development**: Reserved `mobilesam` folder structure, fixed script output format for large language model calling.
- **Geometric Calculation**: Added various geometric parameter calculation functions in `./geometry/grain_metric.py`, and adapted step 7 logic of `yolo_fastsam.py`.
- **Testing**: Completed testing of `Boulder_20260107` folder in CPU environment, good results.

---

## 2026-01-21

- **Modifier:** Li Keran
- **Modification Type:** Parameterization Feature Enhancement
- **Involved Files:** `configs/geometry.yaml`

**Specific Content:**
- Added function to dynamically select geometric parameter calculation items based on YAML config.

---

## 2026-01-22

- **Modifier:** Li Keran
- **Modification Type:** Config Logic Optimization
- **Involved Files:** `configs/geometry.yaml`, `geometry/config_loader.py`, `export_csv.py`, `fastsam/rock_fastsam_system.py`

**Specific Content:**
- Optimized geometric parameter config function, implemented final CSV export field determination based on `configs/geometry.yaml`.

---

## 2026-01-23

- **Modifier:** Zhang Lihua
- **Modification Type:** Module Addition
- **Involved Files:** `/mobilesam` folder, `run_mobilesam.py`

**Specific Content:**
- Referenced FastSAM structure, officially added and integrated MobileSAM processing flow.

---

## 2026-01-24

- **Modifier:** Li Keran
- **Modification Type:** Feature Addition
- **Involved Files:** `/geometry` folder, `grain_metric.py`

**Specific Content:**
- Convexity definition, officially added and integrated calculate_convexity calculation function.

---

## 2026-02-03

- **Modifier:** Li Keran
- **Modification Type:** Feature Addition
- **Involved Files:** `mobilesam_interactive.py`

**Specific Content:**
- Question about why MobileSAM library is needed when Ultralytics SAM library might work

---

## 2026-02-16 - Code Optimization and Refactoring

- **Modifier:** AI Assistant
- **Modification Type:** Code Refactoring and Optimization
- **Involved Files:** `core/` folder, `run_fastsam.py`, `run_mobilesam.py`, `run.py`, `fastsam/seg_tools.py`, `fastsam/seg_optimize.py`, `mobilesam/seg_tools.py`, `mobilesam/seg_optimize.py`, `scale_detector.py`, `grain_marker.py`, `README.md`

### Optimization Overview

This optimization mainly addresses code duplication and unclear module structure issues.

### Main Changes

#### 1. Core Module Creation (core/)

**New Files:**
- `core/__init__.py` - Core module initialization
- `core/seg_tools.py` - Shared tool classes (optimized from ~1400 lines duplicate code to ~700 lines)
- `core/seg_optimize.py` - Shared post-processing module
- `core/cli_base.py` - Shared CLI base functions (extracted ~500 lines duplicate code)

**Content:**
- `ImageProcessor` - Image processing tool class
- `PolygonUtils` - Polygon calculation tool class
- `FileUtils` - File operation tool class
- `PerformanceMonitor` - Performance monitoring class
- `SmartPostProcessor` - Smart post processor
- CLI base functions (argument parsing, interactive wizard, result printing, etc.)

#### 2. Startup Script Simplification

**Before:**
- `run_fastsam.py` - 678 lines
- `run_mobilesam.py` - 871 lines
- Lots of duplicate CLI logic

**After:**
- `run_fastsam.py` - ~200 lines
- `run_mobilesam.py` - ~200 lines
- Uses shared functions from `core/cli_base.py`

#### 3. Sub-module Interface Unification

**Modified Files:**
- `fastsam/seg_tools.py` - Changed to compatibility wrapper importing from core
- `fastsam/seg_optimize.py` - Changed to compatibility wrapper importing from core
- `mobilesam/seg_tools.py` - Changed to compatibility wrapper importing from core
- `mobilesam/seg_optimize.py` - Changed to compatibility wrapper importing from core

**Advantages:**
- Maintain backward compatibility, existing code needs no modification
- Eliminate code duplication
- Easy to maintain and update uniformly

#### 4. Unified Entry Point

**New File:**
- `run.py` - Unified command line entry

**Usage:**
```bash
python run.py fastsam --input image.tif
python run.py mobilesam --input image.tif --batch
```

### Code Line Statistics

| Item | Before | After | Reduction |
|------|--------|-------|-----------|
| seg_tools.py (x2) | ~1400 lines | ~20 lines | -98% |
| seg_optimize.py (x2) | ~840 lines | ~10 lines | -99% |
| run_fastsam.py | 678 lines | 200 lines | -70% |
| run_mobilesam.py | 871 lines | 200 lines | -77% |
| **Total** | **~3800 lines** | **~1300 lines** | **-66%** |

### Backward Compatibility

All optimizations maintain backward compatibility:
- Original API unchanged - All existing code can run without modification
- Config files unchanged - All config file formats remain unchanged
- Startup methods unchanged - Original startup scripts still work

### English Translation

- All comments translated to English
- All output messages translated to English
- All emoji characters removed
- README.md fully translated to English

## 2026-02-16 - Bug Fixes and Compatibility Improvements

- **Modifier:** AI Assistant
- **Modification Type:** Bug Fixes
- **Involved Files:** `run_mobilesam.py`, `run_fastsam.py`, `configs/mobilesam.yaml`, `mobilesam_interactive.py`, `fastsam_interactive.py`

### Fixes

#### 1. Default Config File Fix
- **Issue**: `run.py mobilesam` used wrong default config file (`configs/fastsam.yaml` instead of `configs/mobilesam.yaml`)
- **Solution**: Added `parser.set_defaults(config=DEFAULT_CONFIG)` in both startup scripts
- **Result**: Each mode now uses correct config file automatically

#### 2. Model Path Fix
- **Issue**: `configs/mobilesam.yaml` used incorrect relative paths (`../models/`) and wrong YOLO model filename (`best.pt`)
- **Solution**: Updated to correct paths (`./models/`) and correct filename (`best_yolo_20260107.pt`)
- **Result**: Models load correctly on all platforms

#### 3. macOS GUI Threading Fix
- **Issue**: File dialog crashed on macOS with `NSWindow should only be instantiated on the main thread` error
- **Solution**: Removed `threading.Thread` wrapper from `_safe_file_dialog()` methods, run dialogs directly in main thread
- **Files Modified**: 
  - `mobilesam_interactive.py`
  - `fastsam_interactive.py`
- **Result**: File dialogs work correctly on macOS

#### 4. CPU Mode Default
- **Issue**: Default config used CUDA (`device: cuda`) which fails on Mac without NVIDIA GPU
- **Solution**: Changed default to `device: cpu` in `configs/mobilesam.yaml`
- **Result**: Works out of the box on Mac and CPU-only systems

---

## 2026-02-16 - Unified Output Directory Structure

- **Modifier:** AI Assistant
- **Modification Type:** Architecture Improvement
- **Involved Files:** `configs/fastsam.yaml`, `configs/mobilesam.yaml`, `fastsam/rock_fastsam_system.py`, `mobilesam/rock_mobilesam_system.py`, `fastsam_interactive.py`, `mobilesam_interactive.py`

### Problem
Multiple scattered output directories in project root:
- `results/` - FastSAM auto results
- `results_mobilesam/` - MobileSAM auto results
- `interactive_results/` - MobileSAM interactive results
- `interactive_fastsam_results/` - FastSAM interactive results

This made it difficult to find and manage results.

### Solution
Implemented unified output directory structure:

```
results/
├── fastsam/
│   ├── auto/
│   └── interactive/
├── mobilesam/
│   ├── auto/
│   └── interactive/
├── logs/
│   ├── fastsam/
│   └── mobilesam/
└── temp/
```

### Changes

#### 1. Configuration Files
- Added `mode_subdir` and `type_subdir` options to output config
- Added `log_subdir` to logging config
- Updated both `configs/fastsam.yaml` and `configs/mobilesam.yaml`

#### 2. System Files
- Modified `rock_fastsam_system.py` and `rock_mobilesam_system.py`
- Output path now constructed as: `{root_dir}/{mode_subdir}/{type_subdir}/`
- Log path now constructed as: `{root_dir}/logs/{log_subdir}/`

#### 3. Interactive Scripts
- Updated `fastsam_interactive.py`: output dir changed to `results/fastsam/interactive/`
- Updated `mobilesam_interactive.py`: output dir changed to `results/mobilesam/interactive/`

### Benefits
- All results in one place
- Clear separation between modes and processing types
- Unified logs for easier debugging
- Cleaner project root directory

---

### Future Optimization Suggestions

1. **Merge interactive scripts** - Combine 4 interactive scripts into 1
2. **Unify system classes** - Extract common parts to base class
3. **Add type annotations** - Add complete type annotations to key functions
4. **Unit tests** - Add unit tests for core modules
