#!/usr/bin/env python3
"""
FastMeasure - Unified Launcher for PyInstaller
打包后的统一入口点，不依赖系统 Python
"""

import sys
import os
from pathlib import Path

# 确保可以导入项目模块
if getattr(sys, 'frozen', False):
    # 打包后的环境
    bundle_dir = Path(sys._MEIPASS)
else:
    # 开发环境
    bundle_dir = Path(__file__).parent

sys.path.insert(0, str(bundle_dir))

def run_fastsam_auto(image_path, batch=False):
    """运行 FastSAM Auto/Batch 模式"""
    try:
        from fastsam.rock_fastsam_system import RockUltraSystem
        
        system = RockUltraSystem("config.yaml")
        if not system.initialize_models():
            print("模型初始化失败")
            return 1
        
        system.show_system_info()
        
        if batch:
            results = system.batch_process(image_path)
        else:
            results = system.process_single_image(image_path)
        
        print(f"\n结果保存至: {system.output_root}")
        return 0
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

def run_mobilesam_auto(image_path, batch=False):
    """运行 MobileSAM Auto/Batch 模式"""
    try:
        from mobilesam.rock_mobilesam_system import RockMobileSystem
        
        system = RockMobileSystem("config_mobilesam.yaml")
        if not system.initialize_models():
            print("模型初始化失败")
            return 1
        
        system.show_system_info()
        
        input_path = Path(image_path).absolute()
        if batch:
            results = system.batch_process(str(input_path))
        else:
            results = system.process_single_image(str(input_path))
        
        print(f"\n结果保存至: {system.output_root}")
        return 0
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

def run_interactive(model, image_path):
    """运行 Interactive 模式"""
    # 设置 matplotlib 后端（必须在导入 pyplot 之前）
    import matplotlib
    matplotlib.use('TkAgg')  # 使用 TkAgg 后端，在打包应用中更稳定
    
    try:
        if model == "fastsam":
            from fastsam.rock_fastsam_system import RockUltraSystem
            system = RockUltraSystem("config.yaml")
        else:
            from mobilesam.rock_mobilesam_system import RockMobileSystem
            system = RockMobileSystem("config_mobilesam.yaml")
        
        print(f"\nInitializing {model.upper()} system...")
        if not system.initialize_models():
            print("模型初始化失败")
            return 1
        
        system.show_system_info()
        print(f"\n启动 Interactive 模式...")
        print(f"图片: {image_path}")
        print("=" * 60)
        
        # 运行 interactive 模式
        system.run_interactive_mode(image_path)
        
        print("=" * 60)
        print("Interactive mode ended")
        return 0
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: FastMeasure <fastsam|mobilesam> [选项]")
        return 1
    
    model = sys.argv[1].lower()
    if model not in ['fastsam', 'mobilesam']:
        print(f"错误: 未知模式 '{model}'")
        return 1
    
    # 解析参数
    args = sys.argv[2:]
    image_path = None
    batch_mode = False
    interactive_mode = False
    
    i = 0
    while i < len(args):
        if args[i] in ['--input', '-i'] and i + 1 < len(args):
            image_path = args[i + 1]
            i += 2
        elif args[i] in ['--batch', '-b']:
            batch_mode = True
            i += 1
        elif args[i] in ['--interactive', '-t']:
            interactive_mode = True
            i += 1
        else:
            i += 1
    
    # 检查必要的模型文件
    yolo_model = bundle_dir / "models" / "best_yolo_20260107.pt"
    if not yolo_model.exists():
        print(f"错误: 找不到 YOLO 模型文件: {yolo_model}")
        print("请将模型文件放入 models/ 目录")
        return 1
    
    # 执行对应模式
    if interactive_mode:
        if not image_path:
            print("错误: Interactive 模式需要指定图片路径 (--input)")
            return 1
        return run_interactive(model, image_path)
    else:
        if not image_path:
            print("错误: 需要指定输入路径 (--input)")
            return 1
        
        if model == "fastsam":
            return run_fastsam_auto(image_path, batch_mode)
        else:
            return run_mobilesam_auto(image_path, batch_mode)

if __name__ == "__main__":
    # 设置 multiprocessing 启动方法
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    sys.exit(main())
