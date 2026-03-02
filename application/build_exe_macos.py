#!/usr/bin/env python
"""
PyInstaller Build Script for FastMeasure - macOS Standalone
完全独立的 GUI 应用，不依赖系统 Python

用法:
    cd application
    python build_exe_macos.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def clean_build():
    """Clean previous build files."""
    print("[Build] Cleaning previous build files...")
    for dir_name in ["../build", "../dist"]:
        path = Path(dir_name)
        if path.exists():
            shutil.rmtree(path)
            print(f"[Build] Removed {dir_name}/")
    for spec_file in Path("..").glob("*.spec"):
        spec_file.unlink()
        print(f"[Build] Removed {spec_file.name}")


def build_app():
    """Build macOS GUI app bundle."""
    print("\n[Build] Starting PyInstaller build...")
    print("=" * 60)
    
    sep = ":"  # macOS separator
    
    cmd = [
        sys.executable, "-m", "PyInstaller",
        # 使用 GUI 作为主入口
        "gui_launcher_simple.py",
        # App 名称
        "--name", "FastMeasure",
        # 窗口模式（无控制台）
        "--windowed",
        # macOS App Bundle
        "--onedir",
        # Hooks 目录
        "--additional-hooks-dir", "pyinstaller_hooks",
        # 数据文件 - 核心代码（注意路径从父目录开始）
        "--add-data", f"../core{sep}core",
        "--add-data", f"../fastsam{sep}fastsam",
        "--add-data", f"../mobilesam{sep}mobilesam", 
        "--add-data", f"../geometry{sep}geometry",
        "--add-data", f"../segmenteverygrain{sep}segmenteverygrain",
        # 数据文件 - 配置文件
        "--add-data", f"../configs{sep}configs",
        
        
        # 数据文件 - 启动脚本
        "--add-data", f"main_launcher.py{sep}.",
        "--add-data", f"gui_interactive.py{sep}.",
        "--add-data", f"../run.py{sep}.",
        "--add-data", f"../run_fastsam.py{sep}.",
        "--add-data", f"../run_mobilesam.py{sep}.",
        # Hidden imports - 核心模块
        "--hidden-import", "core",
        "--hidden-import", "core.seg_tools",
        "--hidden-import", "core.cli_base",
        "--hidden-import", "core.model_manager",
        "--hidden-import", "core.scale_calibration",
        "--hidden-import", "core.seg_optimize",
        "--hidden-import", "core.yolo_trainer",
        # Hidden imports - Python 标准库
        "--hidden-import", "unittest",
        "--hidden-import", "unittest.mock",
        "--hidden-import", "importlib.metadata",
        # Hidden imports - FastSAM/MobileSAM 系统
        "--hidden-import", "fastsam.rock_fastsam_system",
        "--hidden-import", "fastsam.yolo_fastsam",
        "--hidden-import", "fastsam.seg_tools",
        "--hidden-import", "mobilesam.rock_mobilesam_system",
        "--hidden-import", "fastsam_interactive",
        "--hidden-import", "mobilesam_interactive",
        # Hidden imports - 第三方包
        "--hidden-import", "cv2",
        "--hidden-import", "cv2.cv2",
        "--hidden-import", "numpy",
        "--hidden-import", "pandas",
        "--hidden-import", "PIL",
        "--hidden-import", "yaml",
        "--hidden-import", "sklearn",
        "--hidden-import", "skimage",
        "--hidden-import", "shapely",
        "--hidden-import", "matplotlib",
        "--hidden-import", "matplotlib.backends.backend_tkagg",
        "--hidden-import", "matplotlib.backends.backend_macosx",
        "--hidden-import", "segment_anything",
        "--hidden-import", "mobile_sam",
        "--hidden-import", "ultralytics",
        "--hidden-import", "ultralytics.nn.modules",
        "--hidden-import", "torch",
        "--hidden-import", "torchvision",
        "--hidden-import", "scipy",
        # Hidden imports - tkinter
        "--hidden-import", "tkinter",
        "--hidden-import", "tkinter.filedialog",
        "--hidden-import", "tkinter.messagebox",
        "--hidden-import", "tkinter.scrolledtext",
        "--hidden-import", "tkinter.ttk",
        # 收集数据
        "--collect-data", "cv2",
        "--collect-all", "yaml",
        "--collect-all", "ultralytics",
        "--collect-all", "torch",
        "--collect-all", "matplotlib",
        "--collect-all", "skimage",
        "--collect-all", "sklearn",
        "--collect-all", "unittest",
        # 排除测试模块
        "--exclude-module", "pytest",
        "--exclude-module", "unittest",
        "--exclude-module", "matplotlib.tests",
        # 清理
        "--clean",
        # 输出到父目录
        "--distpath", "../dist",
        "--workpath", "../build",
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print("\n[Build] Error: Build failed!")
        return False
    
    print("\n[Build] Build completed successfully!")
    return True


def post_process():
    """Post-build setup."""
    print("\n[Build] Post-processing...")
    
    app_path = Path("../dist/FastMeasure.app")
    if not app_path.exists():
        app_path = Path("../dist/FastMeasure")
    
    if app_path.exists():
        macos_dir = app_path / "Contents/MacOS" if app_path.suffix == ".app" else app_path
        
        # 创建 models 和 results 目录
        (macos_dir / "models").mkdir(exist_ok=True)
        (macos_dir / "results").mkdir(exist_ok=True)
        
        # 创建 README
        readme = '''FastMeasure for macOS
===================

完全独立的应用，无需安装 Python！

使用方法:
---------
1. 将模型文件放入 models/ 文件夹:
   - best_yolo_20260107.pt (YOLO检测模型，必需)
   - FastSAM-s.pt (FastSAM模型，可选)
   - mobile_sam.pt (MobileSAM模型，可选)

2. 双击 FastMeasure.app 启动 GUI

3. 首次运行如果提示"无法打开":
   系统偏好设置 > 安全性与隐私 > 仍要打开

功能说明:
---------
- Auto模式: 自动处理单张图片
- Batch模式: 批量处理文件夹
- Interactive模式: 交互式分割（需要选择图片）

模型下载:
---------
https://github.com/KeranLi/FastMeasure/releases

问题反馈:
---------
https://github.com/KeranLi/FastMeasure/issues
'''
        with open(macos_dir / "README.txt", "w") as f:
            f.write(readme)
        
        print(f"[Build] Created models/ and results/ directories")
        print(f"[Build] Output: {app_path}")
    
    return True


def main():
    """Main build process."""
    print("=" * 60)
    print("FastMeasure Standalone GUI App Builder")
    print("  - Fully independent, no Python required!")
    print("=" * 60)
    
    clean_build()
    
    if not build_app():
        return 1
    
    post_process()
    
    print("\n" + "=" * 60)
    print("Build Complete!")
    print("=" * 60)
    print("Output: ../dist/FastMeasure.app")
    print("\n下一步:")
    print("1. 复制模型文件:")
    print("   cp ../models/*.pt ../dist/FastMeasure.app/Contents/MacOS/models/")
    print("2. 运行应用:")
    print("   open ../dist/FastMeasure.app")
    print("\n注意: 首次运行需要在 系统偏好设置 > 安全性 中允许")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
