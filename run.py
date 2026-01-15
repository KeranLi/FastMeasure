#!/usr/bin/env python3
"""
岩石分割系统启动脚本
文件名：run.py
功能：提供命令行接口启动岩石分割系统
使用方式：
  1. 单张图片: python run.py --input 图片路径
  2. 批量处理: python run.py --input 文件夹路径 --batch
  3. 交互模式: python run.py --interactive
  4. 显示帮助: python run.py --help
"""

import os
import sys
import argparse
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 导入岩石分割系统
#from rock import RockSegmentationSystem
from rock import RockSegmentationSystem


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="岩石颗粒自动分割系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单张图片
  python run.py --input path/to/image.tif
  
  # 批量处理文件夹中的所有图片
  python run.py --input path/to/folder --batch
  
  # 交互式模式（图形界面选择文件）
  python run.py --interactive
  
  # 使用自定义配置文件
  python run.py --config my_config.yaml --input image.tif
  
  # 修改处理参数
  python run.py --input image.tif --conf 0.3 --min-area 50
        """
    )
    
    # 输入参数
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="输入图片或文件夹路径"
    )
    
    # 处理模式
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="批量处理模式（当输入是文件夹时）"
    )
    
    parser.add_argument(
        "--interactive", "-t",
        action="store_true",
        help="交互式模式（图形界面选择文件）"
    )
    
    # 配置文件
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="new/config.yaml",
        help="配置文件路径（默认: config.yaml）"
    )
    
    # 处理参数（覆盖配置文件）
    parser.add_argument(
        "--conf",
        type=float,
        help="检测置信度阈值（0-1，默认: 0.25）"
    )
    
    parser.add_argument(
        "--min-area",
        type=int,
        help="最小颗粒面积（像素数，默认: 30）"
    )
    
    parser.add_argument(
        "--min-bbox-area",
        type=int,
        help="最小检测框面积（像素数，默认: 20）"
    )
    
    parser.add_argument(
        "--remove-edge",
        action="store_true",
        help="移除边缘颗粒"
    )
    
    # 输出参数
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出目录路径（默认: results）"
    )
    
    # 其他选项
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="安静模式，减少输出信息"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="岩石颗粒自动分割系统 v1.0"
    )
    
    return parser.parse_args()


def update_config_from_args(system, args):
    """根据命令行参数更新配置"""
    config_updated = False
    
    if args.conf is not None:
        system.config['processing']['confidence_threshold'] = args.conf
        config_updated = True
    
    if args.min_area is not None:
        system.config['processing']['min_area'] = args.min_area
        config_updated = True
    
    if args.min_bbox_area is not None:
        system.config['processing']['min_bbox_area'] = args.min_bbox_area
        config_updated = True
    
    if args.remove_edge:
        system.config['processing']['remove_edge_grains'] = True
        config_updated = True
    
    if args.output is not None:
        system.config['output']['root_dir'] = args.output
        system.output_root = Path(args.output)
        system.output_root.mkdir(parents=True, exist_ok=True)
        config_updated = True
    
    if args.quiet:
        system.config['logging']['show_in_console'] = False
        config_updated = True
    
    if config_updated:
        print("🔄 根据命令行参数更新了配置")
    
    return system


def print_welcome():
    """显示欢迎信息"""
    print("\n" + "=" * 60)
    print("        🪨 岩石颗粒自动分割系统 🪨")
    print("=" * 60)
    print("功能：自动检测、分割并统计岩石显微图像中的颗粒")
    print("=" * 60)


def print_summary(results):
    """显示处理结果摘要"""
    if not results:
        return
    
    print("\n" + "=" * 50)
    print("处理结果摘要")
    print("=" * 50)
    
    if 'total' in results:  # 批量处理结果
        print(f"总图片数: {results['total']}")
        print(f"成功处理: {results['success']}")
        print(f" 处理失败: {results['failed']}")
        print(f" 总颗粒数: {results['total_grains']}")
        
        if results.get('failed_images'):
            print(f"\n 失败图片列表已保存到报告文件中")
    else:  # 单张图片结果
        print(f" 图片: {results.get('image_name', '未知')}")
        print(f" 处理状态: {'成功' if results.get('success') else '失败'}")
        if results.get('success'):
            print(f" 颗粒数量: {results.get('grains_count', 0)}")
            print(f"  处理时间: {results.get('processing_time', 0):.2f}秒")
            print(f" 输出文件数: {len(results.get('output_files', []))}")
    
    print("=" * 50)


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 显示欢迎信息
    if not args.quiet:
        print_welcome()
    
    # 创建岩石分割系统实例
    try:
        system = RockSegmentationSystem(args.config)
    except Exception as e:
        print(f" 系统初始化失败: {e}")
        print(" 请检查配置文件是否存在且格式正确")
        return 1
    
    # 根据命令行参数更新配置
    system = update_config_from_args(system, args)
    
    # 显示系统信息
    if not args.quiet:
        system.show_system_info()
    
    # 初始化AI模型
    if not system.initialize_models():
        print(" 模型初始化失败，请检查模型文件路径")
        return 1
    
    # 根据参数选择运行模式
    results = None
    
    if args.interactive:
        # 交互式模式
        system.run_interactive_mode()
        
    elif args.input:
        # 检查输入路径是否存在
        if not os.path.exists(args.input):
            print(f" 输入路径不存在: {args.input}")
            return 1
        
        if os.path.isfile(args.input):
            # 单张图片处理模式
            print(f"\n 开始处理单张图片: {args.input}")
            results = system.process_single_image(args.input)
            
        elif os.path.isdir(args.input):
            if args.batch:
                # 批量处理模式
                print(f"\n 开始批量处理文件夹: {args.input}")
                results = system.process_batch(args.input)
            else:
                print(f"\n 输入路径是文件夹，但未启用批量处理模式")
                print(" 请使用 --batch 参数进行批量处理")
                print(f" 或指定具体的图片文件路径")
                return 1
        else:
            print(f" 输入路径既不是文件也不是文件夹: {args.input}")
            return 1
    else:
        # 没有指定输入，显示帮助信息
        print("\n 未指定输入路径或模式")
        print(" 请使用以下方式之一:")
        print("   1. 处理单张图片: python run.py --input 图片路径")
        print("   2. 批量处理文件夹: python run.py --input 文件夹路径 --batch")
        print("   3. 交互式模式: python run.py --interactive")
        print("\n 更多帮助: python run.py --help")
        return 1
    
    # 显示处理结果摘要
    if results and not args.quiet:
        print_summary(results)
    
    # 显示最终输出目录信息
    if not args.quiet:
        print(f"\n 所有结果已保存到: {system.output_root}")
        print("=" * 60)
        print("处理完成！🎉")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())