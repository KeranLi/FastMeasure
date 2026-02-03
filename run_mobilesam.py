#!/usr/bin/env python3
"""
MobileSAM超级无敌融合版启动脚本
文件名：run_mobilesam.py
功能：智能环境检测 + 终端交互模式 + 专业命令行 + 完美兼容性
使用方式：
  1. 直接运行进入终端交互模式：python run_mobilesam.py
  2. 单张图片：python run_mobilesam.py --input 图片路径
  3. 批量处理：python run_mobilesam.py --input 文件夹路径 --batch
  4. 交互模式：python run_mobilesam.py --interactive [--input 图片路径]
  5. 显示帮助：python run_mobilesam.py --help
"""

import os
import sys
import argparse
import time
import traceback
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

# ===================== 智能matplotlib后端设置（根据模式选择）=====================
import matplotlib

def setup_matplotlib_backend():
    """根据运行模式智能设置matplotlib后端"""
    
    # 检查是否是交互式模式
    is_interactive_mode = '--interactive' in sys.argv or '-t' in sys.argv
    
    if is_interactive_mode:
        # 交互式模式：尝试使用GUI后端
        print(" 检测到交互式模式，尝试启用GUI...")
        
        # 尝试的后端列表（按优先级）
        backend_options = ['TkAgg', 'Qt5Agg', 'Qt4Agg', 'WXAgg', 'MacOSX']
        
        for backend in backend_options:
            try:
                matplotlib.use(backend)
                print(f" 成功设置后端: {backend}")
                break
            except:
                continue
        else:
            # 所有GUI后端都失败，使用Agg并警告
            matplotlib.use('Agg')
            print(" 所有GUI后端都失败，使用Agg后端（无GUI）")
            print(" 交互式模式需要GUI支持，请检查系统环境")
    else:
        # 自动处理模式：使用Agg提高性能
        matplotlib.use('Agg')
        print(" 自动处理模式：使用Agg后端（无GUI，高性能）")
    
    # 设置一些通用参数
    matplotlib.rcParams.update({
        'figure.max_open_warning': 100  # 提高警告阈值
    })

# 调用设置函数
setup_matplotlib_backend()

def smart_environment_setup():
    """
    智能环境设置（不破坏GUI，自动适配）
    
    核心原则：
    1. 本地有GUI环境 → 保留GUI功能
    2. 服务器无GUI环境 → 自动适配
    3. 不强制，不破坏用户环境
    """
    # 1. 智能路径设置（安全）
    current_dir = Path(__file__).parent.absolute()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # 2. 智能编码设置（仅在需要时）
    try:
        # 尝试使用默认编码，如果失败再调整
        import locale
        locale.getpreferredencoding(do_setlocale=True)
    except Exception:
        # 只有出现编码问题时才设置
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        print(" 设置编码为UTF-8")
    
    return current_dir

def check_dependencies():
    """智能依赖检查：提前发现问题，但不强制中断"""
    required_deps = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'yaml': 'PyYAML',
        'matplotlib': 'Matplotlib',
        'numpy': 'NumPy',
        'ultralytics': 'YOLOv8',
        'shapely': 'Shapely',
        'scikit-image': 'scikit-image'
    }
    
    missing_deps = []
    optional_deps = []
    
    for dep, name in required_deps.items():
        try:
            __import__(dep)
        except ImportError:
            if dep in ['torch', 'cv2', 'matplotlib', 'numpy']:
                missing_deps.append(name)
            else:
                optional_deps.append(name)
    
    if missing_deps:
        print("\n 缺少核心依赖（程序可能无法运行）:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print(f"\n 安装命令: pip install {' '.join(missing_deps).lower()}")
        response = input("\n是否继续运行？(y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            sys.exit(1)
    elif optional_deps:
        print("\n  缺少可选依赖（部分功能可能受限）:")
        for dep in optional_deps:
            print(f"  - {dep}")
        print(" 建议安装以获得完整功能")
    
    return len(missing_deps) == 0

def print_welcome():
    """显示欢迎信息"""
    print("\n" + "=" * 70)
    print("         MobileSAM")
    print("=" * 70)
    print(" 特点：")
    print("  • 智能环境检测（不破坏GUI）")
    print("  • 终端交互模式")
    print("  • 命令行")
    print("  • ")
    print("  • 多模式支持")
    print("=" * 70)
    print(f" 启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

def terminal_interactive_mode():
    """
    终端命令行交互模式
    当用户直接运行 python run_mobilesam.py 时进入此模式
    """
    print("\n" + "=" * 70)
    print("  终端交互模式")
    print("=" * 70)
    print("欢迎使用MobileSAM！我将引导您完成设置。")
    print("=" * 70)
    
    # 创建一个简单的命名空间对象来模拟args
    class SimpleArgs:
        def __init__(self):
            self.input = None
            self.batch = False
            self.interactive = False
            self.config = "config_mobilesam.yaml"
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
    print("\n 请选择处理模式:")
    print("  1.  自动处理模式（YOLO+MobileSAM自动分割）")
    print("  2.  批量处理模式（处理整个文件夹）")
    print("  3.  交互式分割（手动点选颗粒，需要GUI）")
    print("  4.  退出程序")
    
    while True:
        try:
            choice = input("\n请输入选项编号 (1-4): ").strip()
            if choice == '1':
                print(" 已选择: 自动处理模式")
                break
            elif choice == '2':
                print(" 已选择: 批量处理模式")
                args.batch = True
                break
            elif choice == '3':
                print(" 已选择: 交互式分割模式")
                args.interactive = True
                break
            elif choice == '4':
                print(" 退出程序")
                sys.exit(0)
            else:
                print(" 无效选项，请重新输入")
        except KeyboardInterrupt:
            print("\n 用户中断，退出程序")
            sys.exit(0)
    
    # 2. 获取输入路径
    if not args.interactive:
        print("\n 请输入图片或文件夹路径:")
        print(" 提示: 可以直接拖拽文件/文件夹到终端")
        print(" 示例: C:/Users/用户名/Desktop/岩石图片.png 或 ./images/rock.tif")
        
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
                            print(f" 找到文件: {input_path.name}")
                        elif input_path.is_dir():
                            # 统计图片文件数量
                            try:
                                image_files = [f for f in os.listdir(input_path) 
                                            if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp'))]
                                print(f" 找到文件夹，包含 {len(image_files)} 个图片文件")
                            except:
                                print(f" 找到文件夹")
                        break
                    else:
                        print(f" 路径不存在: {user_input}")
                        print(" 请重新输入有效路径")
                else:
                    print(" 路径不能为空")
            
            except KeyboardInterrupt:
                print("\n 用户中断，退出程序")
                sys.exit(0)
    
    # 3. 设置参数（仅自动处理模式）
    if not args.interactive:
        print("\n  参数设置 (按Enter使用默认值):")
        
        # 置信度阈值
        print("\n YOLO检测置信度阈值 (0.0-1.0)")
        print(" 默认: 0.15 | 建议: 0.15-0.25 (岩石颗粒)")
        
        while True:
            try:
                conf_input = input("请输入置信度阈值 (默认0.15): ").strip()
                
                if conf_input == "":
                    args.conf = 0.15
                    print(" 使用默认置信度: 0.15")
                    break
                else:
                    conf_value = float(conf_input)
                    if 0.0 <= conf_value <= 1.0:
                        args.conf = conf_value
                        print(f" 设置置信度: {conf_value}")
                        break
                    else:
                        print(" 置信度必须在0.0到1.0之间")
            
            except ValueError:
                print(" 请输入有效的数字")
            except KeyboardInterrupt:
                print("\n 用户中断，退出程序")
                sys.exit(0)
        
        # 输出目录
        print("\n 输出目录设置")
        print(" 默认: results_mobilesam (自动创建)")
        
        while True:
            try:
                output_input = input("请输入输出目录 (默认results_mobilesam): ").strip()
                
                if output_input == "":
                    args.output = "results_mobilesam"
                    print(" 使用默认输出目录: results_mobilesam")
                    break
                else:
                    output_path = Path(output_input)
                    output_path.mkdir(parents=True, exist_ok=True)
                    args.output = output_input
                    print(f" 设置输出目录: {output_path}")
                    break
            
            except KeyboardInterrupt:
                print("\n 用户中断，退出程序")
                sys.exit(0)
        
        # 高级设置
        print("\n 高级设置 (可选)")
        
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
                        print(f" 设置最小颗粒面积: {min_area}像素")
                        break
                    else:
                        print(" 最小面积必须大于0")
            
            except ValueError:
                print(" 请输入有效的整数")
            except KeyboardInterrupt:
                print("\n 用户中断，退出程序")
                sys.exit(0)
        
        # 是否移除边缘颗粒
        while True:
            try:
                remove_edge_input = input("是否移除边缘颗粒？(y/n，默认n): ").strip().lower()
                
                if remove_edge_input in ['', 'n', 'no']:
                    print(" 保留边缘颗粒")
                    break
                elif remove_edge_input in ['y', 'yes']:
                    args.remove_edge = True
                    print(" 将移除边缘颗粒")
                    break
                else:
                    print(" 请输入 y 或 n")
            
            except KeyboardInterrupt:
                print("\n 用户中断，退出程序")
                sys.exit(0)
    else:
        print("\n 交互式模式说明:")
        print("💡 将使用GUI界面手动分割颗粒")
        print("💡 需要图形界面支持")
        print("💡 如果GUI无法启动，请使用自动处理模式")
    
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
    
    print(f"   输出: {args.output or 'results_mobilesam'}")
    
    if args.min_area:
        print(f"   最小面积: {args.min_area}像素")
    
    if args.remove_edge:
        print("   边缘处理: 移除边缘颗粒")
    
    print("=" * 70)
    
    # 确认开始处理
    while True:
        try:
            confirm = input("\n✅ 是否开始处理？(y/n): ").strip().lower()
            
            if confirm in ['y', 'yes']:
                print(" 开始处理...")
                return args
            elif confirm in ['n', 'no']:
                print(" 用户取消，退出程序")
                sys.exit(0)
            else:
                print(" 请输入 y 或 n")
        
        except KeyboardInterrupt:
            print("\n 用户中断，退出程序")
            sys.exit(0)

def parse_arguments():
    """解析命令行参数（专业模式）"""
    parser = argparse.ArgumentParser(
        description="MobileSAM岩石颗粒自动分割系统（超级无敌融合版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 终端交互模式（新手推荐）
  python run_mobilesam.py
  
  # 单张图片自动处理
  python run_mobilesam.py --input path/to/image.tif
  
  # 批量处理文件夹
  python run_mobilesam.py --input path/to/folder --batch
  
  # GUI交互式分割
  python run_mobilesam.py --interactive
  
  # 高级用法
  python run_mobilesam.py --input image.tif --conf 0.2 --min-area 50 --output my_results
  
环境变量:
  MPLBACKEND=Agg     # 强制无GUI模式
  MPLBACKEND=TkAgg   # 强制使用Tkinter GUI
  MPLBACKEND=Qt5Agg  # 强制使用Qt5 GUI
  
配置文件:
  默认使用 config_mobilesam.yaml，可通过 --config 指定
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
        help="GUI交互式模式（手动点选颗粒，需要图形界面）"
    )
    
    # 配置文件
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config_mobilesam.yaml",
        help="配置文件路径（默认: config_mobilesam.yaml）"
    )
    
    # 处理参数（覆盖配置文件）
    parser.add_argument(
        "--conf",
        type=float,
        help="YOLO检测置信度阈值（0-1，默认: 0.15）"
    )
    
    parser.add_argument(
        "--min-area",
        type=int,
        help="最小颗粒面积（像素数，默认: 30）"
    )
    
    parser.add_argument(
        "--min-bbox-area",
        type=int,
        help="最小检测框面积（像素数，默认: 15）"
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
        help="输出目录路径（默认: results_mobilesam）"
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
        "--gui-backend",
        type=str,
        choices=['auto', 'agg', 'tkagg', 'qt5agg', 'webagg'],
        default='auto',
        help="指定matplotlib后端 (默认: auto自动检测)"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="MobileSAM岩石颗粒自动分割系统 v3.0.0（超级无敌融合版）"
    )
    
    return parser.parse_args()

def print_summary(results):
    """显示处理结果摘要"""
    if not results:
        return
    
    print("\n" + "=" * 60)
    print(" MobileSAM处理结果摘要")
    print("=" * 60)
    
    if 'total' in results:  # 批量处理结果
        print(f" 总图片数: {results['total']}")
        print(f" 成功处理: {results['success']}")
        print(f" 处理失败: {results['failed']}")
        print(f" 总颗粒数: {results['total_grains']}")
        
        if results.get('failed_images'):
            print(f"\n 失败图片列表已保存到报告文件中")
    else:  # 单张图片结果
        print(f"📷 图片: {results.get('image_name', '未知')}")
        
        if results.get('success'):
            print(f" 处理状态: 成功")
            print(f" 颗粒数量: {results.get('grains_count', 0)}")
            print(f" 处理时间: {results.get('processing_time', 0):.2f}秒")
            
            if results.get('scale_detection_success'):
                print(f" 比例因子: {results.get('scale_factor', 'N/A')} μm/px")
            
            output_files = results.get('output_files', [])
            print(f" 输出文件数: {len(output_files)}")
            
            if output_files:
                print(f"\n 生成的文件:")
                for i, file in enumerate(output_files[:5], 1):
                    file_name = Path(file).name
                    print(f"  {i}. {file_name}")
                
                if len(output_files) > 5:
                    print(f"  ... 还有 {len(output_files)-5} 个文件")
        else:
            print(f" 处理状态: 失败")
    
    if results.get('error_message'):
        print(f"\n  错误信息: {results.get('error_message')}")
    
    print("=" * 60)

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
    
    return system

def main():
    """主函数"""
    # 1. 智能环境设置（不破坏GUI）
    current_dir = smart_environment_setup()
    
    # 2. 解析命令行参数
    args = parse_arguments()
    
    # 3. 检查是否进入终端交互模式
    is_interactive_mode = len(sys.argv) == 1
    
    # 4. 显示欢迎信息
    if not args.quiet or is_interactive_mode:
        print_welcome()
    
    # 5. 检查依赖
    if not check_dependencies() and not is_interactive_mode:
        print("  依赖检查失败，程序可能无法正常运行")
    
    try:
        # 7. 导入主系统（项目一导入修改）
        from mobilesam.rock_mobilesam_system import RockMobileSystem
        
        # 8. 处理模式选择
        if is_interactive_mode:
            # 终端交互模式
            print("\n 进入终端交互模式...")
            terminal_args = terminal_interactive_mode()
            
            # 创建主系统实例
            print(f"\n 初始化MobileSAM主系统...")
            system = RockMobileSystem(terminal_args.config)
            
            # 初始化AI模型
            print(" 初始化AI模型...")
            if not system.initialize_models():
                print(" 模型初始化失败，请检查模型文件路径")
                return 1
            
            print(" AI模型初始化成功")
            
            # 根据终端交互模式返回的参数更新配置
            system = update_config_from_args(system, terminal_args)
            
            # 显示系统信息
            system.show_system_info()
            
            # 根据参数选择运行模式
            if terminal_args.interactive:
                # GUI交互式模式
                print(f"\n 启动GUI交互式模式...")
                # 添加GUI保持机制
                try:
                    system.run_interactive_mode(terminal_args.input)
                    
                    # GUI关闭后，询问是否要保存结果
                    print("\n" + "=" * 60)
                    print(" GUI交互已结束")
                    print("=" * 60)
                    
                    # 检查是否有交互结果
                    if system.interactive_system and hasattr(system.interactive_system, 'grains'):
                        grains_count = len(system.interactive_system.grains)
                        if grains_count > 0:
                            print(f" 交互过程中标记了 {grains_count} 个颗粒")
                            save_choice = input("是否保存交互结果？(y/n): ").strip().lower()
                            if save_choice in ['y', 'yes']:
                                # 调用保存功能
                                if hasattr(system.interactive_system, '_generate_complete_outputs'):
                                    output_dir = system.interactive_system._generate_complete_outputs()
                                    if output_dir:
                                        print(f" 结果已保存到: {output_dir}")
                        else:
                            print("ℹ  没有标记任何颗粒，无需保存")
                    else:
                        print("ℹ  没有可保存的交互结果")
                        
                except Exception as e:
                    print(f" 交互模式执行失败: {e}")
                    import traceback
                    traceback.print_exc()
                    return 1
                    
            elif terminal_args.input:
                # 自动处理模式
                input_path = Path(terminal_args.input)
                
                if input_path.is_file():
                    # 单张图片处理模式
                    print(f"\n 开始处理单张图片: {terminal_args.input}")
                    results = system.process_single_image(terminal_args.input)
                    
                    # 显示处理结果摘要
                    if results:
                        print_summary(results)
                        
                elif input_path.is_dir():
                    if terminal_args.batch:
                        # 批量处理模式
                        print(f"\n 开始批量处理文件夹: {terminal_args.input}")
                        results = system.batch_process(terminal_args.input)
                        
                        # 显示处理结果摘要
                        if results:
                            print_summary(results)
                    else:
                        print(f"\n 输入路径是文件夹，但未启用批量处理模式")
                        return 1
                else:
                    print(f"\n 输入路径既不是文件也不是文件夹: {terminal_args.input}")
                    return 1
                
                # 显示最终输出目录信息
                print(f"\n 所有结果已保存到: {system.output_root}")
                print("\n" + "=" * 70)
                print(" MobileSAM处理完成！")
                print("=" * 70)
        
        else:
            # 命令行参数模式
            print(f"\n 初始化MobileSAM主系统...")
            system = RockMobileSystem(args.config)
            
            # 初始化AI模型
            print(" 初始化AI模型...")
            if not system.initialize_models():
                print(" 模型初始化失败，请检查模型文件路径")
                return 1
            
            print(" AI模型初始化成功")
            
            # 根据命令行参数更新配置
            system = update_config_from_args(system, args)
            
            # 显示系统信息
            if not args.quiet:
                system.show_system_info()
            
            # 根据参数选择运行模式
            if args.interactive:
                # GUI交互式模式
                print(f"\n 启动GUI交互式模式...")
                # 添加GUI保持机制
                try:
                    system.run_interactive_mode(args.input)
                    
                    # GUI关闭后，询问是否要保存结果
                    print("\n" + "=" * 60)
                    print(" GUI交互已结束")
                    print("=" * 60)
                    
                    # 检查是否有交互结果
                    if system.interactive_system and hasattr(system.interactive_system, 'grains'):
                        grains_count = len(system.interactive_system.grains)
                        if grains_count > 0:
                            print(f" 交互过程中标记了 {grains_count} 个颗粒")
                            save_choice = input("是否保存交互结果？(y/n): ").strip().lower()
                            if save_choice in ['y', 'yes']:
                                # 调用保存功能
                                if hasattr(system.interactive_system, '_generate_complete_outputs'):
                                    output_dir = system.interactive_system._generate_complete_outputs()
                                    if output_dir:
                                        print(f" 结果已保存到: {output_dir}")
                        else:
                            print("ℹ  没有标记任何颗粒，无需保存")
                    else:
                        print("ℹ 没有可保存的交互结果")
                        
                except Exception as e:
                    print(f" 交互模式执行失败: {e}")
                    import traceback
                    traceback.print_exc()
                    return 1
                    
            elif args.input:
                # 自动处理模式
                input_path = Path(args.input).absolute()
                
                if not input_path.exists():
                    print(f" 输入路径不存在: {input_path}")
                    return 1
                
                if input_path.is_file():
                    # 单张图片处理模式
                    print(f"\n 开始处理单张图片: {input_path}")
                    results = system.process_single_image(str(input_path))
                    
                    # 显示处理结果摘要
                    if results and not args.quiet:
                        print_summary(results)
                        
                elif input_path.is_dir():
                    if args.batch:
                        # 批量处理模式
                        print(f"\n 开始批量处理文件夹: {input_path}")
                        results = system.batch_process(str(input_path))
                        
                        # 显示处理结果摘要
                        if results and not args.quiet:
                            print_summary(results)
                    else:
                        print(f"\n 输入路径是文件夹，但未启用批量处理模式")
                        print(" 请使用 --batch 参数进行批量处理")
                        return 1
                else:
                    print(f"\n 输入路径既不是文件也不是文件夹: {input_path}")
                    return 1
                
                # 显示最终输出目录信息
                if not args.quiet:
                    print(f"\n 所有结果已保存到: {system.output_root}")
                    print("\n" + "=" * 70)
                    print(" MobileSAM处理完成！")
                    print("=" * 70)
            else:
                # 没有指定输入
                print("\n 未指定输入路径或模式")
                print("💡 请使用以下方式之一:")
                print("  1. python run_mobilesam.py (终端交互模式)")
                print("  2. python run_mobilesam.py --input 图片路径")
                print("  3. python run_mobilesam.py --input 文件夹路径 --batch")
                print("  4. python run_mobilesam.py --interactive")
                print("\n 更多帮助: python run_mobilesam.py --help")
                return 1
        
        return 0
        
    except ImportError as e:
        print(f" 导入MobileSAM系统失败: {e}")
        print(" 请检查:")
        print("  1. mobilesam文件夹是否存在")
        print("  2. rock_mobilesam_system.py文件是否存在")
        print("  3. 是否安装了所有依赖")
        return 1
    except Exception as e:
        print(f" 程序运行出错: {e}")
        if args.debug or is_interactive_mode:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n 程序被用户中断")
        sys.exit(1)