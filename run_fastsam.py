#!/usr/bin/env python3
"""
UltraFastSAM启动脚本
文件名：run_fastsam.py
功能：提供命令行接口启动UltraFastSAM系统
"""

import os
import sys
import argparse
import time
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 导入UltraFastSAM系统
from fastsam.rock_fastsam_system import RockUltraSystem


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="UltraFastSAM岩石颗粒自动分割系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单张图片
  python run_fastsam.py --input path/to/image.tif
  
  # 批量处理文件夹中的所有图片
  python run_fastsam.py --input path/to/folder --batch
  
  # 使用自定义配置文件
  python run_fastsam.py --config config.yaml --input image.tif
  
  # 修改处理参数
  python run_fastsam.py --input image.tif --conf 0.3 --min-area 50
  
  # 性能监控模式
  python run_fastsam.py --input image.tif --performance
  
  # 安静模式（减少输出）
  python run_fastsam.py --input image.tif --quiet
        """
    )
    
    # 输入参数
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入图片或文件夹路径"
    )
    
    # 处理模式
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="批量处理模式（当输入是文件夹时）"
    )
    
    # 配置文件
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="配置文件路径（默认: config.yaml）"
    )
    
    # 处理参数（覆盖配置文件）
    parser.add_argument(
        "--conf",
        type=float,
        help="YOLO检测置信度阈值（0-1，默认: 0.25）"
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
        help="输出目录路径（默认: results_ultra_fastsam）"
    )
    
    # 性能参数
    parser.add_argument(
        "--performance", "-p",
        action="store_true",
        help="启用性能监控模式"
    )
    
    # 调试参数
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="启用调试模式（保存更多信息）"
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
        version="UltraFastSAM岩石颗粒自动分割系统 v2.0.0"
    )
    
    return parser.parse_args()


def update_config_from_args(system, args):
    """根据命令行参数更新配置"""
    config_updated = False
    
    if args.conf is not None:
        system.config['processing']['yolo_confidence'] = args.conf
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
    
    if args.performance:
        system.config['processing']['performance_monitoring'] = True
        system.config['output']['save_performance'] = True
        config_updated = True
    
    if args.debug:
        system.config['output']['save_debug_info'] = True
        system.config['logging']['level'] = 'DEBUG'
        config_updated = True
    
    if args.quiet:
        system.config['logging']['show_in_console'] = False
        config_updated = True
    
    if config_updated:
        print("🔄 根据命令行参数更新了配置")
    
    return system


def print_welcome():
    """显示欢迎信息"""
    print("\n" + "=" * 70)
    print("-----UltraFastSAM岩石颗粒自动分割系统-----")
    print("=" * 70)
    print("功能：岩石显微图像颗粒分割")
    print("v1.0.0")
    print("=" * 70)


def print_summary(results):
    """显示处理结果摘要"""
    if not results:
        return
    
    print("\n" + "=" * 60)
    print("UltraFastSAM处理结果摘要")
    print("=" * 60)
    
    if 'total' in results:  # 批量处理结果
        print(f"总图片数: {results['total']}")
        print(f"成功处理: {results['success']}")
        print(f"处理失败: {results['failed']}")
        print(f"总颗粒数: {results['total_grains']}")
        
        if results.get('failed_images'):
            print(f"\n失败图片列表已保存到报告文件中")
    else:  # 单张图片结果
        print(f"图片: {results.get('image_name', '未知')}")
        print(f"处理状态: {'✅ 成功' if results.get('success') else '❌ 失败'}")
        
        if results.get('success'):
            print(f"颗粒数量: {results.get('grains_count', 0)}")
            print(f"处理时间: {results.get('processing_time', 0):.2f}秒")
            
            if results.get('scale_detection_success'):
                print(f"比例因子: {results.get('scale_factor', 'N/A')} μm/px")
            
            output_files = results.get('output_files', [])
            print(f"输出文件数: {len(output_files)}")
            
            if output_files:
                print(f"生成的文件:")
                for i, file in enumerate(output_files[:5], 1):
                    file_name = Path(file).name
                    print(f"  {i}. {file_name}")
                
                if len(output_files) > 5:
                    print(f"  ... 还有 {len(output_files)-5} 个文件")
    
    if results.get('error_message'):
        print(f"错误信息: {results.get('error_message')}")
    
    print("=" * 60)


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 显示欢迎信息
    if not args.quiet:
        print_welcome()
        print(f"\n启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查输入路径
    if not os.path.exists(args.input):
        print(f"❌ 输入路径不存在: {args.input}")
        return 1
    
    # 创建UltraFastSAM系统实例
    try:
        system = RockUltraSystem(args.config)
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        print("请检查配置文件是否存在且格式正确")
        return 1
    
    # 根据命令行参数更新配置
    system = update_config_from_args(system, args)
    
    # 显示系统信息
    if not args.quiet:
        system.show_system_info()
    
    # 初始化AI模型
    print("\n🔄 初始化AI模型...")
    if not system.initialize_models():
        print("❌ 模型初始化失败，请检查模型文件路径")
        return 1
    
    print("✅ AI模型初始化成功")
    
    # 根据参数选择运行模式
    results = None
    
    if os.path.isfile(args.input):
        # 单张图片处理模式
        print(f"\n🚀 开始处理单张图片: {args.input}")
        results = system.process_single_image(args.input)
        
    elif os.path.isdir(args.input):
        if args.batch:
            # 批量处理模式
            print(f"\n🚀 开始批量处理文件夹: {args.input}")
            results = system.batch_process(args.input)
        else:
            print(f"\n❌ 输入路径是文件夹，但未启用批量处理模式")
            print("请使用 --batch 参数进行批量处理")
            print(f"或指定具体的图片文件路径")
            return 1
    else:
        print(f"❌ 输入路径既不是文件也不是文件夹: {args.input}")
        return 1
    
    # 显示处理结果摘要
    if results and not args.quiet:
        print_summary(results)
    
    # 显示最终输出目录信息
    if not args.quiet:
        print(f"\n📁 所有结果已保存到: {system.output_root}")
        
        # 显示生成的文件列表
        if results and results.get('success') and 'output_files' in results:
            output_files = results['output_files']
            if output_files:
                print(f"\n📄 生成的文件:")
                for i, file_path in enumerate(output_files, 1):
                    file_name = Path(file_path).name
                    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                    file_size_str = f"{file_size/1024:.1f}KB" if file_size < 1024*1024 else f"{file_size/1024/1024:.1f}MB"
                    print(f"  {i:2d}. {file_name} ({file_size_str})")
        
        print("\n" + "=" * 70)
        print("🎉 UltraFastSAM处理完成！")
        print("=" * 70)
    
    return 0


if __name__ == "__main__":
    # 启动系统
    sys.exit(main())