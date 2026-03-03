# FastMeasure Installation Guide

## Quick Start (Recommended)

### Method 1: Using Conda (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/KeranLi/FastMeasure.git
cd FastMeasure

# 2. Create conda environment (CPU version)
conda env create -f envs/environment.yml

# 3. Activate environment
conda activate fastmeasure

# 4. Download model files
# Visit: https://drive.google.com/drive/folders/1SPah9woaytIeinkLzQgGiXyj_SCJ3v1q?usp=drive_link
# Download and place in models/ folder

# 5. Verify and run
python utils/download_models.py
python run_fastsam.py --input your_image.jpg
```

### Method 2: Using pip

```bash
# 1. Create environment
conda create -n fastmeasure python=3.10 -y
conda activate fastmeasure

# 2. Install PyTorch 2.3+ (supports NumPy 2.x)
pip install torch>=2.3.0 torchvision>=0.18.0

# 3. Install other dependencies
pip install -r envs/requirements.txt

# 4. Download models and run (same as above)
```

## GPU Support

### Option 1: Edit environment.yml
Remove or comment out the `- cpuonly` line in `envs/environment.yml`:

```yaml
dependencies:
  - pytorch>=2.3.0
  - torchvision>=0.18.0
  # - cpuonly  # Remove this line for GPU
```

Then create environment:
```bash
conda env create -f envs/environment.yml
```

### Option 2: Manual GPU installation

```bash
# For CUDA 11.8
pip install torch>=2.3.0 torchvision>=0.18.0 --index-url https://download.pytorch.org/whl/cu118

# Then install other dependencies
pip install -r envs/requirements.txt
```

## Optional: MobileSAM

For MobileSAM support (higher precision, slower):

```bash
# MobileSAM requires timm (will be installed automatically with conda env)
# If using pip:
pip install timm>=0.9.0

# Then install MobileSAM
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

## Troubleshooting

### NumPy Compatibility

If you encounter NumPy version errors:
- **Solution**: Ensure PyTorch >= 2.3.0 (supports NumPy 2.x)
- **Old PyTorch** (< 2.3) requires NumPy < 2.0

### MobileSAM Import Error

If `import mobile_sam` fails with `ModuleNotFoundError: No module named 'timm'`:
```bash
pip install timm>=0.9.0
```

### Model Files

Model files (~700 MB) are not included in the repository. Download from:
- Google Drive: https://drive.google.com/drive/folders/1SPah9woaytIeinkLzQgGiXyj_SCJ3v1q?usp=drive_link

Place files in `models/` folder:
- `best_yolo_20260107.pt` (required)
- `FastSAM-s.pt` (required for FastSAM)
- `mobile_sam.pt` (optional, for MobileSAM)

## Verification

Check installation:
```bash
python utils/check_environment.py
```

This will verify all dependencies are correctly installed.
