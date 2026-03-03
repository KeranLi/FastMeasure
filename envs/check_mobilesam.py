#!/usr/bin/env python3
"""
MobileSAM 环境检查脚本
用于排查 mobile_sam 安装问题
"""

import sys
import subprocess

def check_package(name, import_name=None):
    """检查包是否安装及版本"""
    if import_name is None:
        import_name = name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        path = getattr(module, '__file__', 'unknown')
        return True, version, path
    except ImportError as e:
        return False, str(e), None

def main():
    print("=" * 70)
    print("MobileSAM Environment Checker")
    print("=" * 70)
    
    # Python 版本
    print(f"\nPython: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # 关键包检查
    print("\n" + "-" * 70)
    print("Key Package Versions:")
    print("-" * 70)
    
    packages = [
        ('mobile_sam', 'mobile_sam'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('numpy', 'numpy'),
        ('opencv-python', 'cv2'),
        ('Pillow', 'PIL'),
        ('timm', 'timm'),  # mobile_sam 依赖
    ]
    
    for pkg_name, import_name in packages:
        installed, version, path = check_package(pkg_name, import_name)
        status = "[OK]" if installed else "[FAIL]"
        print(f"\n{status} {pkg_name}")
        if installed:
            print(f"   Version: {version}")
            print(f"   Path: {path}")
        else:
            print(f"   Error: {version}")
    
    # 尝试导入 mobile_sam 组件
    print("\n" + "-" * 70)
    print("MobileSAM Components:")
    print("-" * 70)
    
    try:
        import mobile_sam
        print("[OK] mobile_sam imported")
        
        # 检查具体组件
        components = [
            'sam_model_registry',
            'SamPredictor', 
            'build_sam',
            'setup',
        ]
        
        for comp in components:
            try:
                obj = getattr(mobile_sam, comp)
                print(f"  [OK] {comp}")
            except AttributeError:
                print(f"  [MISSING] {comp}")
                
    except Exception as e:
        print(f"[FAIL] mobile_sam import failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 检查 pip list
    print("\n" + "-" * 70)
    print("Pip installed packages (related):")
    print("-" * 70)
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list'],
            capture_output=True,
            text=True
        )
        
        for line in result.stdout.split('\n'):
            if any(x in line.lower() for x in ['mobile', 'sam', 'torch', 'numpy', 'opencv', 'pillow', 'timm']):
                print(f"  {line}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "=" * 70)
    print("建议:")
    print("=" * 70)
    print("1. 如果 mobile_sam 未安装，运行:")
    print("   pip install git+https://github.com/ChaoningZhang/MobileSAM.git")
    print("\n2. 如果已安装但导入失败，可能是依赖版本问题，尝试:")
    print("   pip install --upgrade mobile_sam")
    print("   pip install --upgrade torch torchvision")
    print("\n3. 如果还有问题，尝试卸载重装:")
    print("   pip uninstall mobile_sam -y")
    print("   pip install git+https://github.com/ChaoningZhang/MobileSAM.git")

if __name__ == "__main__":
    main()
