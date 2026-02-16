#!/usr/bin/env python3
"""
UltraFastSAM启动脚本 - （支持GUI交互模式和终端交互向导）
文件名：run_fastsam.py
功能：提供命令行接口启动UltraFastSAM系统，支持多种交互方式
"""

import os
import sys
import argparse
import time
import traceback
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 导入UltraFastSAM系统
try:
    from fastsam.rock_fastsam_system import RockUltraSystem
    SYSTEM_AVAILABLE = True
except ImportError as e:
    SYSTEM_AVAILABLE = False
    print(f"导入UltraFastSAM系统失败: {e}")
    print("请检查 rock_fastsam_system.py 文件是否存在")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="UltraFastSAM岩石颗粒自动分割系统 - 支持GUI交互模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 终端交互向导模式
  python run_fastsam.py
  
  # 处理单张图片
  python run_fastsam.py --input path/to/image.tif
  
  # 批量处理文件夹中的所有图片
  python run_fastsam.py --input path/to/folder --batch
  
  # GUI交互式模式（手动分割颗粒）
  python run_fastsam.py --interactive [--input 图片路径]
  
  # 使用自定义配置文件
  python run_fastsam.py --config config.yaml --input image.tif
  
  # 修改处理参数
  python run_fastsam.py --input image.tif --conf 0.3 --min-area 50
  
  # 性能监控模式
  python run_fastsam.py --input image.tif --performance
  
  # 安静模式（减少输出）
  python run_fastsam.py --input image.tif --quiet
  
  # 显示帮助
  python run_fastsam.py --help
        """
    )
    
    # 输入参数
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="输入图片或文件夹路径（交互模式可省略）"
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
        help="GUI交互式模式（手动分割颗粒，需要图形界面）"
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
        version="UltraFastSAM岩石颗粒自动分割系统 v2.0.0（支持GUI交互模式）"
    )
    
    return parser.parse_args()


def terminal_interactive_wizard():
    """终端交互向导模式（当用户直接运行脚本时进入）"""
    print("\n" + "=" * 70)
    print("    UltraFastSAM终端交互向导")
    print("=" * 70)
    print("欢迎使用UltraFastSAM！我将引导您完成设置。")
    print("=" * 70)
    
    # 创建一个简单的命名空间对象来模拟args
    class SimpleArgs:
        def __init__(self):
            self.input = None
            self.batch = False
            self.interactive = False
            self.config = "config.yaml"
            self.conf = None
            self.min_area = None
            self.min_bbox_area = None
            self.remove_edge = False
            self.output = None
            self.performance = False
            self.debug = False
            self.quiet = False
    
    args = SimpleArgs()
    
    # 1. 选择处理模式
    print("\n请选择处理模式:")
    print("  1.  自动处理模式（YOLO+FastSAM自动分割）")
    print("  2.  批量处理模式（处理整个文件夹）")
    print("  3.  GUI交互式分割（手动选择颗粒，需要图形界面）")
    print("  4.  退出程序")
    
    while True:
        try:
            choice = input("\n请输入选项编号 (1-4): ").strip()
            if choice == '1':
                print("已选择: 自动处理模式")
                break
            elif choice == '2':
                print("已选择: 批量处理模式")
                args.batch = True
                break
            elif choice == '3':
                print("已选择: GUI交互式分割模式")
                args.interactive = True
                break
            elif choice == '4':
                print("退出程序")
                sys.exit(0)
            else:
                print("无效选项，请重新输入")
        except KeyboardInterrupt:
            print("\n用户中断，退出程序")
            sys.exit(0)
    
    # 2. 获取输入路径（如果不是交互模式）
    if not args.interactive:
        print("\n请输入图片或文件夹路径:")
        print("提示: 可以直接拖拽文件/文件夹到终端")
        print("示例: C:/Users/用户名/Desktop/岩石图片.png 或 ./images/rock.tif")
        
        while True:
            try:
                if args.batch:
                    user_input = input("请输入文件夹路径: ").strip()
                else:
                    user_input = input("请输入图片路径: ").strip()
                
                # 检查路径是否存在
                if user_input:
                    input_path = Path(user_input)
                    
                    # 支持相对路径和绝对路径
                    if not input_path.exists():
                        # 尝试在当前目录下查找
                        current_dir = Path.cwd() / user_input
                        if current_dir.exists():
                            input_path = current_dir
                    
                    if input_path.exists():
                        args.input = str(input_path.absolute())
                        
                        # 检查是文件还是文件夹
                        if input_path.is_file():
                            print(f"找到文件: {input_path.name}")
                        elif input_path.is_dir():
                            # 统计图片文件数量
                            try:
                                image_files = [f for f in os.listdir(input_path) 
                                            if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp'))]
                                print(f"找到文件夹，包含 {len(image_files)} 个图片文件")
                            except:
                                print(f"找到文件夹")
                        break
                    else:
                        print(f"路径不存在: {user_input}")
                        print("请重新输入有效路径")
                else:
                    print("路径不能为空")
            
            except KeyboardInterrupt:
                print("\n用户中断，退出程序")
                sys.exit(0)
    
    # 3. 设置参数（仅自动处理模式）
    if not args.interactive:
        print("\n  参数设置 (按Enter使用默认值):")
        
        # 置信度阈值
        print("\nYOLO检测置信度阈值 (0.0-1.0)")
        print("默认: 0.25 | 建议: 0.25-0.35 (岩石颗粒)")
        
        while True:
            try:
                conf_input = input("请输入置信度阈值 (默认0.25): ").strip()
                
                if conf_input == "":
                    args.conf = 0.25
                    print("使用默认置信度: 0.25")
                    break
                else:
                    conf_value = float(conf_input)
                    if 0.0 <= conf_value <= 1.0:
                        args.conf = conf_value
                        print(f"设置置信度: {conf_value}")
                        break
                    else:
                        print("置信度必须在0.0到1.0之间")
            
            except ValueError:
                print("请输入有效的数字")
            except KeyboardInterrupt:
                print("\n用户中断，退出程序")
                sys.exit(0)
        
        # 输出目录
        print("\n输出目录设置")
        print("默认: results_ultra_fastsam (自动创建)")
        
        while True:
            try:
                output_input = input("请输入输出目录 (默认results_ultra_fastsam): ").strip()
                
                if output_input == "":
                    args.output = "results_ultra_fastsam"
                    print("使用默认输出目录: results_ultra_fastsam")
                    break
                else:
                    output_path = Path(output_input)
                    output_path.mkdir(parents=True, exist_ok=True)
                    args.output = output_input
                    print(f"设置输出目录: {output_path}")
                    break
            
            except KeyboardInterrupt:
                print("\n用户中断，退出程序")
                sys.exit(0)
        
        # 高级设置
        print("\n高级设置 (可选)")
        
        # 最小面积
        while True:
            try:
                min_area_input = input("最小颗粒面积像素数 (默认30，按Enter跳过): ").strip()
                
                if min_area_input == "":
                    break
                else:
                    min_area = int(min_area_input)
                    if min_area > 0:
                        args.min_area = min_area
                        print(f"设置最小颗粒面积: {min_area}像素")
                        break
                    else:
                        print("最小面积必须大于0")
            
            except ValueError:
                print("请输入有效的整数")
            except KeyboardInterrupt:
                print("\n用户中断，退出程序")
                sys.exit(0)
        
        # 是否移除边缘颗粒
        while True:
            try:
                remove_edge_input = input("是否移除边缘颗粒？(y/n，默认n): ").strip().lower()
                
                if remove_edge_input in ['', 'n', 'no']:
                    print("保留边缘颗粒")
                    break
                elif remove_edge_input in ['y', 'yes']:
                    args.remove_edge = True
                    print("将移除边缘颗粒")
                    break
                else:
                    print("请输入 y 或 n")
            
            except KeyboardInterrupt:
                print("\n用户中断，退出程序")
                sys.exit(0)
    else:
        print("\nGUI交互式模式说明:")
        print("将使用GUI界面手动分割颗粒")
        print("需要图形界面支持")
        print("支持框选和点选两种交互方式")
        print("结果保存格式与自动处理模式一致")
    
    # 4. 确认设置
    print("\n" + "=" * 70)
    print("   配置确认")
    print("=" * 70)
    
    if args.interactive:
        print("   模式: GUI交互式分割")
        if args.input:
            print(f"   输入: {args.input}")
        else:
            print("   输入: 使用GUI选择文件")
    elif args.batch:
        print(f"   模式: 批量处理")
        print(f"   输入: {args.input}")
    else:
        print(f"   模式: 自动处理")
        print(f"   输入: {args.input}")
    
    if args.conf:
        print(f"   置信度: {args.conf}")
    
    print(f"   输出: {args.output or 'results_ultra_fastsam'}")
    
    if args.min_area:
        print(f"   最小面积: {args.min_area}像素")
    
    if args.remove_edge:
        print("   边缘处理: 移除边缘颗粒")
    
    print("=" * 70)
    
    # 确认开始处理
    while True:
        try:
            confirm = input("\n是否开始处理？(y/n): ").strip().lower()
            
            if confirm in ['y', 'yes']:
                print("开始处理...")
                return args
            elif confirm in ['n', 'no']:
                print("用户取消，退出程序")
                sys.exit(0)
            else:
                print("请输入 y 或 n")
        
        except KeyboardInterrupt:
            print("\n用户中断，退出程序")
            sys.exit(0)


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
        print("根据命令行参数更新了配置")
    
    return system


def print_welcome():
    """显示欢迎信息"""
    print("\n" + "=" * 70)
    print("     UltraFastSAM岩石颗粒自动分割系统")
    print("     （支持GUI交互模式）")
    print("=" * 70)
    print("功能：岩石显微图像颗粒分割")
    print("模式：自动分割 | 批量处理 | GUI交互式分割")
    print("版本：v2.0.0")
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
        print(f"处理状态: {'成功' if results.get('success') else '失败'}")
        
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
    # 检查系统可用性
    if not SYSTEM_AVAILABLE:
        print("UltraFastSAM系统不可用，无法启动")
        return 1
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 检查是否进入终端交互向导模式（当没有参数时）
    if len(sys.argv) == 1:
        # 终端交互向导模式
        print_welcome()
        terminal_args = terminal_interactive_wizard()
        
        # 将终端交互向导的参数应用到args对象
        for key, value in vars(terminal_args).items():
            setattr(args, key, value)
    
    # 显示欢迎信息（如果不是quiet模式）
    if not args.quiet and len(sys.argv) > 1:
        print_welcome()
        print(f"\n启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查是否需要交互式模式
    if args.interactive:
        # GUI交互式模式
        try:
            system = RockUltraSystem(args.config)
            
            if not args.quiet:
                system.show_system_info()
            
            print("\n启动GUI交互式模式...")
            print("提示:")
            print("  1. 左键拖动：绘制选择框")
            print("  2. 左键单击：点选小颗粒")
            print("  3. 按 'Shift+S' 快速保存结果")
            print("  4. 按 'h' 键查看帮助")
            
            # 运行交互式模式
            system.run_interactive_mode(args.input)
            
            # GUI关闭后处理
            print("\n" + "=" * 60)
            print("GUI交互已结束")
            print("=" * 60)
            
            # 检查是否有交互结果可以保存
            if system.interactive_system and hasattr(system.interactive_system, 'grains'):
                grains_count = len(system.interactive_system.grains)
                if grains_count > 0:
                    print(f"交互过程中标记了 {grains_count} 个颗粒")
                    save_choice = input("是否保存交互结果？(y/n): ").strip().lower()
                    if save_choice in ['y', 'yes']:
                        # 调用保存功能
                        if hasattr(system.interactive_system, '_generate_complete_outputs'):
                            output_dir = system.interactive_system._generate_complete_outputs()
                            if output_dir:
                                print(f"结果已保存到: {output_dir}")
                else:
                    print("没有标记任何颗粒，无需保存")
            else:
                print("没有可保存的交互结果")
            
            return 0
                
        except Exception as e:
            print(f"交互模式执行失败: {e}")
            traceback.print_exc()
            return 1
    
    # 检查输入路径（非交互模式需要输入）
    if not args.input and not args.interactive:
        print("未指定输入路径或模式")
        print("\n请使用以下方式之一:")
        print("  1. 自动处理模式: python run_fastsam.py --input 图片路径")
        print("  2. 批量处理模式: python run_fastsam.py --input 文件夹路径 --batch")
        print("  3. GUI交互模式: python run_fastsam.py --interactive [--input 图片路径]")
        print("\n更多帮助: python run_fastsam.py --help")
        return 1
    
    if args.input and not os.path.exists(args.input):
        print(f"输入路径不存在: {args.input}")
        return 1
    
    # 创建UltraFastSAM系统实例
    try:
        system = RockUltraSystem(args.config)
    except Exception as e:
        print(f"系统初始化失败: {e}")
        print("请检查配置文件是否存在且格式正确")
        return 1
    
    # 根据命令行参数更新配置
    system = update_config_from_args(system, args)
    
    # 显示系统信息
    if not args.quiet:
        system.show_system_info()
    
    # 初始化AI模型
    print("\n初始化AI模型...")
    if not system.initialize_models():
        print("模型初始化失败，请检查模型文件路径")
        return 1
    
    print("AI模型初始化成功")
    
    # 根据参数选择运行模式
    results = None
    
    if os.path.isfile(args.input):
        # 单张图片处理模式
        print(f"\n开始处理单张图片: {args.input}")
        results = system.process_single_image(args.input)
        
    elif os.path.isdir(args.input):
        if args.batch:
            # 批量处理模式
            print(f"\n开始批量处理文件夹: {args.input}")
            results = system.batch_process(args.input)
        else:
            print(f"\n输入路径是文件夹，但未启用批量处理模式")
            print("请使用 --batch 参数进行批量处理")
            print(f"或指定具体的图片文件路径")
            return 1
    else:
        print(f"输入路径既不是文件也不是文件夹: {args.input}")
        return 1
    
    # 显示处理结果摘要
    if results and not args.quiet:
        print_summary(results)
    
    # 显示最终输出目录信息
    if not args.quiet:
        print(f"\n所有结果已保存到: {system.output_root}")
        
        # 显示生成的文件列表
        if results and results.get('success') and 'output_files' in results:
            output_files = results['output_files']
            if output_files:
                print(f"\n生成的文件:")
                for i, file_path in enumerate(output_files, 1):
                    file_name = Path(file_path).name
                    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                    file_size_str = f"{file_size/1024:.1f}KB" if file_size < 1024*1024 else f"{file_size/1024/1024:.1f}MB"
                    print(f"  {i:2d}. {file_name} ({file_size_str})")
        
        print("\n" + "=" * 70)
        print("UltraFastSAM处理完成！")
        print("=" * 70)
    
    return 0


if __name__ == "__main__":
    # 启动系统
    sys.exit(main())