#!/usr/bin/env python
"""
PyInstaller Build Script for FastMeasure

This script packages FastMeasure into a standalone executable for Windows.
Users can run the application without installing Python or dependencies.

Requirements:
    pip install pyinstaller

Usage:
    python build_exe.py

Output:
    dist/FastMeasure/        - Portable folder (recommended)
    dist/FastMeasure.exe     - Single executable (slower startup)
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def clean_build():
    """Clean previous build files."""
    print("[Build] Cleaning previous build files...")
    dirs_to_remove = ["build", "dist"]
    for dir_name in dirs_to_remove:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)
            print(f"[Build] Removed {dir_name}/")
    
    # Clean spec files
    for spec_file in Path(".").glob("*.spec"):
        spec_file.unlink()
        print(f"[Build] Removed {spec_file.name}")
    
    # Remove installer.iss if exists
    if Path("installer.iss").exists():
        Path("installer.iss").unlink()
        print("[Build] Removed installer.iss")


def check_requirements():
    """Check if required tools are installed."""
    print("[Build] Checking requirements...")
    
    try:
        import PyInstaller
        print(f"[Build] PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("[Build] Error: PyInstaller not installed!")
        print("[Build] Install with: pip install pyinstaller")
        return False
    
    # Check for required directories
    required_dirs = ["core", "fastsam", "mobilesam", "geometry"]
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            print(f"[Build] Error: Required directory '{dir_name}' not found!")
            return False
    
    # Create hooks directory if not exists
    hooks_dir = Path("hooks")
    if not hooks_dir.exists():
        hooks_dir.mkdir()
        print(f"[Build] Created hooks/ directory")
    
    # Create hook file if not exists
    hook_file = hooks_dir / "hook-pyparsing.py"
    if not hook_file.exists():
        hook_content = '''"""PyInstaller hook for pyparsing"""
hiddenimports = ['unittest']
'''
        with open(hook_file, "w") as f:
            f.write(hook_content)
        print(f"[Build] Created hooks/hook-pyparsing.py")
    
    print("[Build] All requirements satisfied")
    return True


def build_executable():
    """Build the executable using PyInstaller."""
    print("\n[Build] Starting PyInstaller build...")
    print("=" * 60)
    
    # Determine separator for --add-data based on platform
    # Windows uses ';', Linux/macOS uses ':'
    sep = ";" if sys.platform == "win32" else ":"
    
    # PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        # Main script
        "utils/gui_launcher.py",
        # Output name
        "--name", "FastMeasure",
        # Windowed mode (no console)
        "--windowed",
        # Icon (if available)
        # "--icon=assets/icon.ico",
        # Additional hooks directory (for custom hooks)
        "--additional-hooks-dir", "hooks",
        # Add data files (platform-specific separator)
        # Format: "source_path{sep}destination_folder"
        "--add-data", f"core{sep}core",
        "--add-data", f"fastsam{sep}fastsam",
        "--add-data", f"mobilesam{sep}mobilesam",
        "--add-data", f"geometry{sep}geometry",
        "--add-data", f"configs{sep}configs",
        "--add-data", f"utils{sep}utils",
        # Entry point scripts
        "--add-data", f"run.py{sep}.",
        "--add-data", f"run_fastsam.py{sep}.",
        "--add-data", f"run_mobilesam.py{sep}.",
        # Create hooks directory if not exists
        # (hook files should be in hooks/ directory)
        # Hidden imports for core modules
        "--hidden-import", "core",
        "--hidden-import", "core.seg_tools",
        "--hidden-import", "core.cli_base",
        "--hidden-import", "core.model_manager",
        "--hidden-import", "core.scale_calibration",
        "--hidden-import", "core.seg_optimize",
        "--hidden-import", "core.yolo_trainer",
        "--hidden-import", "core.segment_core",
        # Hidden imports for geometry modules
        "--hidden-import", "geometry",
        "--hidden-import", "geometry.grain_metric",
        "--hidden-import", "geometry.config_loader",
        "--hidden-import", "geometry.export_csv",
        # Hidden imports for fastsam modules
        "--hidden-import", "fastsam",
        "--hidden-import", "fastsam.rock_fastsam_system",
        "--hidden-import", "fastsam.yolo_fastsam",
        "--hidden-import", "fastsam.seg_engine",
        # Hidden imports for mobilesam modules
        "--hidden-import", "mobilesam",
        "--hidden-import", "mobilesam.rock_mobilesam_system",
        "--hidden-import", "mobilesam.yolo_mobilesam",
        "--hidden-import", "mobilesam.mobile_sam_engine",
        # Hidden imports for third-party packages
        "--hidden-import", "ultralytics",
        "--hidden-import", "ultralytics.nn.modules",
        "--hidden-import", "torch",
        "--hidden-import", "torchvision",
        "--hidden-import", "cv2",
        "--hidden-import", "cv2.cv2",
        "--hidden-import", "numpy",
        "--hidden-import", "pandas",
        "--hidden-import", "PIL",
        "--hidden-import", "yaml",
        "--hidden-import", "sklearn",
        "--hidden-import", "sklearn.cluster",
        "--hidden-import", "skimage",
        "--hidden-import", "skimage.measure",
        "--hidden-import", "skimage.morphology",
        "--hidden-import", "scipy",
        "--hidden-import", "scipy.spatial",
        "--hidden-import", "shapely",
        "--hidden-import", "shapely.geometry",
        "--hidden-import", "matplotlib",
        "--hidden-import", "matplotlib.backends.backend_tkagg",
        "--hidden-import", "segment_anything",
        "--hidden-import", "mobile_sam",
        "--hidden-import", "timm",
        # Hidden imports for stdlib (required by dependencies)
        "--hidden-import", "unittest",
        "--hidden-import", "unittest.mock",
        "--hidden-import", "pyparsing",
                        # Collect all packages
        "--collect-all", "ultralytics",
        "--collect-all", "torch",
        # Collect cv2 data files (includes opencv dependencies)
        "--collect-data", "cv2",
        "--collect-binaries", "cv2",
        "--copy-metadata", "opencv-python",
        # Exclude unnecessary packages (reduce size)
        # Note: unittest cannot be excluded - required by pyparsing
        "--exclude-module", "pytest",
        "--exclude-module", "tkinter.test",
        "--exclude-module", "matplotlib.tests",
        "--exclude-module", "numpy.random._examples",
        # Clean build
        "--clean",
        # One directory (faster startup, recommended)
        "--onedir",
    ]
    
    # Run PyInstaller
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print("\n[Build] Error: PyInstaller build failed!")
        return False
    
    print("\n" + "=" * 60)
    print("[Build] PyInstaller build completed successfully!")
    return True


def get_dist_dir():
    """Get the distribution directory based on platform."""
    if sys.platform == "darwin":
        # macOS creates .app bundle
        return Path("dist/FastMeasure.app/Contents/MacOS")
    else:
        # Windows and Linux
        return Path("dist/FastMeasure")


def copy_additional_files():
    """Copy additional files to distribution."""
    print("\n[Build] Copying additional files...")
    
    dist_dir = get_dist_dir()
    
    # Create models directory (empty, user needs to add models)
    models_dir = dist_dir / "models"
    models_dir.mkdir(exist_ok=True, parents=True)
    
    # Create results directory
    results_dir = dist_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Create README for users
    if sys.platform == "darwin":
        exe_name = "FastMeasure.app"
        run_cmd = "Double-click FastMeasure.app"
    elif sys.platform == "win32":
        exe_name = "FastMeasure.exe"
        run_cmd = "Double-click FastMeasure.exe"
    else:
        exe_name = "FastMeasure"
        run_cmd = "./FastMeasure"
    
    readme_content = f"""FastMeasure - Rock Grain Segmentation

Getting Started:
1. Place your YOLO and SAM model files in the 'models' folder:
   - best_yolo_20260107.pt (YOLO detection model)
   - FastSAM-s.pt (FastSAM model, optional)
   - mobile_sam.pt (MobileSAM model, optional)

2. {run_cmd} to start the GUI

3. Select an image or folder and click 'Run Segmentation'

For more information:
https://github.com/KeranLi/FastMeasure
"""
    
    readme_path = dist_dir / "README.txt"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    print(f"[Build] Created models directory: {models_dir}")
    print(f"[Build] Created results directory: {results_dir}")
    print(f"[Build] Created user README")


def create_installer_script():
    """Create Inno Setup script for Windows installer (optional)."""
    iss_content = """; Inno Setup Script for FastMeasure
; Requires Inno Setup: https://jrsoftware.org/isinfo.php

#define MyAppName "FastMeasure"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "FastMeasure Team"
#define MyAppURL "https://github.com/KeranLi/FastMeasure"
#define MyAppExeName "FastMeasure.exe"

[Setup]
AppId={{FASTMEASURE-APP-ID}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\\{#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=LICENSE
OutputDir=installer
OutputBaseFilename=FastMeasure_Setup
SetupIconFile=assets\\icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\\FastMeasure\\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\\FastMeasure\\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: "{app}\\models"; Permissions: users-full
Name: "{app}\\results"; Permissions: users-full

[Icons]
Name: "{autoprograms}\\{#MyAppName}"; Filename: "{app}\\{#MyAppExeName}"
Name: "{autodesktop}\\{#MyAppName}"; Filename: "{app}\\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
"""
    
    with open("installer.iss", "w") as f:
        f.write(iss_content)
    
    print("\n[Build] Created Inno Setup script: installer.iss")
    print("[Build] You can use Inno Setup to create a Windows installer")


def print_summary():
    """Print build summary."""
    print("\n" + "=" * 60)
    print("Build Summary")
    print("=" * 60)
    
    # Determine output based on platform
    if sys.platform == "darwin":
        dist_dir = Path("dist/FastMeasure.app")
        exe_path = dist_dir / "Contents/MacOS/FastMeasure"
        package_name = "FastMeasure.app"
    else:
        dist_dir = Path("dist/FastMeasure")
        exe_path = dist_dir / ("FastMeasure.exe" if sys.platform == "win32" else "FastMeasure")
        package_name = dist_dir.name
    
    if exe_path.exists():
        # Calculate total size of distribution
        total_size = sum(
            f.stat().st_size for f in dist_dir.rglob("*") if f.is_file()
        )
        size_mb = total_size / (1024 * 1024)
        
        print(f"Package: {dist_dir}")
        print(f"Executable: {exe_path}")
        print(f"Total size: {size_mb:.1f} MB")
        print(f"\nTo run:")
        if sys.platform == "darwin":
            print(f"  1. Copy model files to: {dist_dir}/Contents/MacOS/models/")
            print(f"  2. Double-click: {package_name}")
        else:
            print(f"  1. Copy model files to: {dist_dir}/models/")
            print(f"  2. Run: {exe_path.name}")
        print(f"\nTo distribute:")
        print(f"  - Zip the entire package: {dist_dir}")
        if sys.platform == "win32":
            print(f"  - Or create installer with: installer.iss")
    else:
        print("Warning: Executable not found!")
        print(f"Expected at: {exe_path}")
    
    print("=" * 60)


def main():
    """Main build process."""
    print("=" * 60)
    print("FastMeasure Executable Builder")
    print("=" * 60)
    
    # Clean previous build
    clean_build()
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Build executable
    if not build_executable():
        return 1
    
    # Copy additional files
    copy_additional_files()
    
    # Print summary
    print_summary()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
