#!/usr/bin/env python3
"""
FastMeasure - Unified Entry Point
统一入口脚本，支持 FastSAM 和 MobileSAM 两种模式

此脚本作为统一的命令行入口，实际功能由 run_fastsam.py 和 run_mobilesam.py 提供

Usage:
    python run.py fastsam                    # FastSAM 终端交互模式
    python run.py fastsam --input image.tif  # FastSAM 处理单图
    python run.py fastsam --input folder/ --batch  # FastSAM 批量处理
    python run.py fastsam --interactive      # FastSAM 交互模式
    
    python run.py mobilesam                  # MobileSAM 终端交互模式
    python run.py mobilesam --input image.tif # MobileSAM 处理单图
    python run.py mobilesam --input folder/ --batch  # MobileSAM 批量处理
    python run.py mobilesam --interactive    # MobileSAM 交互模式

Options:
    --input, -i <path>      输入图像或文件夹路径
    --batch, -b             批量处理模式
    --interactive, -t       交互式分割模式
    --config <path>         配置文件路径
    --conf <float>          置信度阈值
    --min-area <int>        最小颗粒面积
    --output <path>         输出目录
    --device <cpu|cuda>     处理设备
    --quiet, -q             静默模式
    --debug, -d             调试模式
    --help, -h              显示帮助信息
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def print_usage():
    """打印使用说明"""
    print(__doc__)
    print("\n详细说明:")
    print("  FastSAM 模式 - 快速分割，推荐用于大批量处理")
    print("  MobileSAM 模式 - 精度更高，推荐用于精细标注")
    print("\n示例:")
    print("  # 启动终端交互向导")
    print("  python run.py fastsam")
    print("  python run.py mobilesam")
    print("")
    print("  # 处理单张图像")
    print("  python run.py fastsam --input image.tif")
    print("  python run.py mobilesam --input image.tif")
    print("")
    print("  # 批量处理文件夹")
    print("  python run.py fastsam --input folder/ --batch")
    print("  python run.py mobilesam --input folder/ --batch")
    print("")
    print("  # 交互式分割（GUI）")
    print("  python run.py fastsam --interactive --input image.tif")
    print("  python run.py mobilesam --interactive --input image.tif")
    print("")
    print("  # 使用自定义配置")
    print("  python run.py fastsam --config configs/fastsam.yaml --input image.tif")


def run_fastsam():
    """运行 FastSAM 模式 - 直接复用 run_fastsam.py"""
    # 移除第一个参数 'fastsam'，让 run_fastsam.py 看到的是标准参数
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    # 导入并运行 run_fastsam.py 的 main 函数
    from run_fastsam import main
    return main()


def run_mobilesam():
    """运行 MobileSAM 模式 - 直接复用 run_mobilesam.py"""
    # 移除第一个参数 'mobilesam'，让 run_mobilesam.py 看到的是标准参数
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    # 导入并运行 run_mobilesam.py 的 main 函数
    from run_mobilesam import main
    return main()


def main():
    """主入口函数"""
    if len(sys.argv) < 2:
        print_usage()
        return 1
    
    mode = sys.argv[1].lower()
    
    # 处理帮助选项
    if mode in ['-h', '--help', 'help']:
        print_usage()
        return 0
    
    # 根据模式分派到对应脚本
    if mode == 'fastsam':
        return run_fastsam()
    elif mode == 'mobilesam':
        return run_mobilesam()
    else:
        print(f"错误: 未知模式 '{mode}'")
        print("请使用 'fastsam' 或 'mobilesam'")
        print("")
        print_usage()
        return 1


if __name__ == "__main__":
    sys.exit(main())
