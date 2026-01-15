"""
SuperFastSAM
文件名：rock_super_fastsam.py
"""

import os
import sys
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
import yaml

import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ultralytics import YOLO

# 导入SuperFastSAM引擎
try:
    from fastsam_optimized import SuperFastSAM
    from yolo_super_fastsam import yolo_super_fastsam_segmentation
    print("✅ 成功导入SuperFastSAM引擎")
except ImportError as e:
    print(f"❌ 导入SuperFastSAM引擎失败: {e}")
    sys.exit(1)

# 导入工具函数
try:
    from utils import (
        check_image_file_pro,
        validate_image_data,
        convert_to_rgb,
        normalize_image
    )
    print("✅ 成功导入工具函数")
except ImportError:
    print("⚠️ 使用简化工具函数")
    from utils_simple import *

# 导入比例尺检测模块
try:
    from scale_detector import ScaleDetector
    SCALE_DETECTOR_AVAILABLE = True
    print("✅ 成功导入比例尺检测模块")
except ImportError as e:
    print(f"⚠️ 导入比例尺检测模块失败: {e}")
    SCALE_DETECTOR_AVAILABLE = False

# 导入颗粒标注模块
try:
    from grain_marker import add_grain_labels, add_labels_with_config
    GRAIN_MARKER_AVAILABLE = True
    print("✅ 成功导入颗粒标注模块")
except ImportError as e:
    print(f"⚠️ 导入颗粒标注模块失败: {e}")
    GRAIN_MARKER_AVAILABLE = False


class RockSegmentationSystemSuper:
    """SuperFastSAM生产级岩石分割系统"""
    
    VERSION = "1.0.0"
    
    def __init__(self, config_path: str = "new/config_super_fastsam.yaml"):
        """
        初始化SuperFastSAM系统
        
        Args:
            config_path: 配置文件路径
        """
        # 先初始化logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        print("=" * 60)
        print(f"🏭 SuperFastSAM岩石分割系统 v{self.VERSION}")
        print("=" * 60)
        
        # 加载配置文件
        self.config = self._load_config(config_path)
        
        # 设置输出目录
        self.output_root = Path(self.config['output']['root_dir'])
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # 初始化日志系统
        self._setup_logging()
        
        # 初始化模型
        self.yolo_model = None
        self.super_fastsam = None
        
        # 初始化比例尺检测器
        self.scale_detector = None
        if SCALE_DETECTOR_AVAILABLE:
            self._init_scale_detector()
        
        # 读取颗粒标注配置
        self.grain_label_config = self.config.get('grain_labeling', {})
        
        # 处理空字符串背景为None
        if 'bg_color' in self.grain_label_config and self.grain_label_config['bg_color'] == '':
            self.grain_label_config['bg_color'] = None
        
        self.logger.info(f"SuperFastSAM系统初始化完成")
        self.logger.info(f"输出目录: {self.output_root}")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = Path(__file__).parent / Path(config_path).name
        
        if not config_file.exists():
            print(f"⚠️ 配置文件 {config_path} 不存在，使用默认配置")
            return self._get_default_config()
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ 配置文件加载成功: {config_file}")
            return config
        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
            print("使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'model_paths': {
                'yolo': '../models/best.pt',
                'fastsam': '../models/FastSAM-s.pt',
                'device': 'cpu'
            },
            'scale_detection': {
                'enabled': True,
                'known_length_um': 1000.0,
                'detection_params': {
                    'red_lower1': [0, 120, 120],
                    'red_upper1': [10, 255, 255],
                    'red_lower2': [160, 120, 120],
                    'red_upper2': [180, 255, 255],
                    'crop_height': 220,
                    'crop_width': 600,
                    'search_margin': 80,
                    'min_aspect_ratio': 8,
                    'min_horizontal_score': 0.6
                }
            },
            'processing': {
                'yolo_confidence': 0.25,
                'fastsam_confidence': 0.35,
                'min_area': 30,
                'min_bbox_area': 20,
                'remove_edge_grains': False,
                'plot_results': True,
                'performance_monitor': True
            },
            'output': {
                'root_dir': 'results_super_fastsam',
                'create_subdirs': True,
                'save_visualization': True,
                'save_mask': True,
                'save_statistics': True,
                'save_summary': True,
                'save_performance': True
            },
            'batch_processing': {
                'supported_formats': ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp'],
                'skip_corrupted': True,
                'max_workers': 1,
                'log_errors': True
            },
            'logging': {
                'level': 'INFO',
                'save_to_file': True,
                'show_in_console': True,
                'log_format': '%(asctime)s - %(levelname)s - %(message)s'
            },
            'grain_labeling': {
                'enabled': True,
                'font_size': 11,
                'text_color': 'yellow',
                'bg_color': '',
                'show_area': True,
                'max_labels': 1000,
                'min_area': 0,
                'text_outline': True,
                'outline_color': 'black',
                'outline_width': 2.0
            },
            'performance': {
                'enable_monitoring': True,
                'save_timings': True,
                'alert_threshold_sec': 300
            }
        }
    
    def _init_scale_detector(self):
        """初始化比例尺检测器"""
        scale_config = self.config.get('scale_detection', {})
        if scale_config.get('enabled', False) and SCALE_DETECTOR_AVAILABLE:
            try:
                self.scale_detector = ScaleDetector(self.config)
                self.logger.info("✅ 比例尺检测器初始化成功")
            except Exception as e:
                self.logger.warning(f"比例尺检测器初始化失败: {e}")
                self.scale_detector = None
        else:
            self.scale_detector = None
            self.logger.info("比例尺检测功能已禁用")
    
    def _setup_logging(self):
        """设置日志系统"""
        log_config = self.config['logging']
        log_dir = self.output_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"super_fastsam_{timestamp}.log"
        
        log_level = getattr(logging, log_config['level'])
        
        # 配置根logger
        logger = logging.getLogger()
        logger.setLevel(log_level)
        logger.handlers.clear()
        
        # 文件handler
        if log_config['save_to_file']:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_formatter = logging.Formatter(log_config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # 控制台handler
        if log_config['show_in_console']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"日志系统初始化完成，日志文件: {log_file}")
    
    def initialize_models(self) -> bool:
        """初始化YOLO和SuperFastSAM模型"""
        self.logger.info("=" * 50)
        self.logger.info("初始化SuperFastSAM AI模型...")
        
        model_paths = self.config['model_paths']
        device = model_paths.get('device', 'cpu')
        
        self.logger.info(f"运行设备: {device}")
        
        try:
            # 加载YOLO模型
            yolo_path = model_paths['yolo']
            if not Path(yolo_path).exists():
                self.logger.error(f"YOLO模型文件不存在: {yolo_path}")
                return False
            
            self.logger.info(f"加载YOLO模型: {yolo_path}")
            self.yolo_model = YOLO(yolo_path)
            self.logger.info("✅ YOLO模型加载成功")
            
            # 加载SuperFastSAM引擎
            fastsam_path = model_paths['fastsam']
            if not Path(fastsam_path).exists():
                self.logger.error(f"FastSAM模型文件不存在: {fastsam_path}")
                return False
            
            self.logger.info(f"加载SuperFastSAM引擎: {fastsam_path}")
            self.super_fastsam = SuperFastSAM(
                model_path=fastsam_path,
                device=device
            )
            self.logger.info("✅ SuperFastSAM引擎加载成功")
            
            self.logger.info("=" * 50)
            return True
            
        except Exception as e:
            self.logger.error(f"模型初始化失败: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def process_single_image(self, image_path: str) -> Dict:
        """
        处理单张岩石图片
        """
        result = {
            'image_path': str(image_path),
            'image_name': Path(image_path).name,
            'success': False,
            'grains_count': 0,
            'error_message': None,
            'output_files': [],
            'processing_time': 0,
            'performance_metrics': {},
            'timestamp': datetime.now().isoformat(),
            'scale_factor': None,
            'scale_detection_success': False
        }
        
        start_time = time.time()
        
        try:
            # 检查图片文件
            is_valid, message = check_image_file_pro(image_path)
            if not is_valid:
                self.logger.warning(f"⚠️ 图片文件检查警告: {image_path} - {message}")
            
            # 创建输出目录
            output_dir = self.create_output_structure(Path(image_path))
            
            # 加载图片
            self.logger.info(f"🖼️  加载图片: {image_path}")
            image = self._load_image_safely(image_path)
            
            if image is None:
                result['error_message'] = "无法加载图片"
                result['processing_time'] = time.time() - start_time
                self.logger.error(f"❌ 图片加载失败: {image_path}")
                return result
            
            # 验证图像数据
            is_valid, valid_msg = validate_image_data(image)
            if not is_valid:
                result['error_message'] = valid_msg
                result['processing_time'] = time.time() - start_time
                self.logger.error(f"❌ 图像数据验证失败: {valid_msg}")
                return result
            
            self.logger.info(f"📊 图片最终尺寸: {image.shape}, 数据类型: {image.dtype}")
            
            # 检测比例尺
            scale_factor = None
            scale_detection_success = False
            
            if self.scale_detector and SCALE_DETECTOR_AVAILABLE:
                try:
                    self.logger.info("检测图片中的比例尺...")
                    scale_factor, scale_success = self.scale_detector.detect(image_path)
                    
                    if scale_success:
                        result['scale_factor'] = float(scale_factor)
                        result['scale_detection_success'] = True
                        scale_detection_success = True
                        self.logger.info(f"比例尺检测成功: {scale_factor:.4f} μm/px")
                    else:
                        self.logger.warning("比例尺检测失败，仅输出像素面积")
                except Exception as e:
                    self.logger.warning(f"比例尺检测异常: {e}")
            
            # 获取处理参数
            processing_config = self.config['processing']
            
            # 运行SuperFastSAM分割流水线
            all_grains, labels, mask_all, grain_data, fig, ax = yolo_super_fastsam_segmentation(
                image=image,
                yolo_model=self.yolo_model,
                super_fastsam=self.super_fastsam,
                conf_threshold=processing_config['yolo_confidence'],
                min_area=processing_config['min_area'],
                min_bbox_area=processing_config['min_bbox_area'],
                remove_edge_grains=processing_config['remove_edge_grains'],
                plot_image=processing_config['plot_results'],
                class_id=None
            )
            
            # 更新结果
            result['grains_count'] = len(all_grains)
            result['success'] = True
            
            # 保存结果文件
            output_files = []
            
            # 保存可视化结果（与项目一完全相同的三张图）
            if self.config['output']['save_visualization'] and fig is not None:
                # 1. 第一张图：原始分割结果
                plot_path = output_dir / "segmentation_result.png"
                fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                output_files.append(str(plot_path))
                self.logger.info(f"原始结果图保存至: {plot_path}")
                
                # 2. 第二张图：带标注的结果图
                if (GRAIN_MARKER_AVAILABLE and 
                    self.grain_label_config.get('enabled', True) and 
                    grain_data is not None and 
                    not grain_data.empty):
                    
                    try:
                        # 创建带标注的图像
                        fig_labeled, ax_labeled = plt.subplots(figsize=(15, 10))
                        ax_labeled.imshow(image)
                        ax_labeled.axis('off')
                        
                        # 添加颗粒标注（与项目一相同）
                        if 'add_labels_with_config' in globals():
                            ax_labeled = add_labels_with_config(
                                ax=ax_labeled,
                                grain_data=grain_data,
                                image_shape=image.shape,
                                config=self.grain_label_config
                            )
                        
                        # 隐藏坐标轴和边框
                        ax_labeled.set_xticks([])
                        ax_labeled.set_yticks([])
                        ax_labeled.set_xlim([0, image.shape[1]])
                        ax_labeled.set_ylim([image.shape[0], 0])
                        plt.tight_layout()
                        
                        # 保存带标注的图
                        labeled_path = output_dir / "segmentation_labeled.png"
                        fig_labeled.savefig(labeled_path, dpi=300, bbox_inches='tight', 
                                           pad_inches=0, facecolor='white')
                        output_files.append(str(labeled_path))
                        plt.close(fig_labeled)
                        
                        self.logger.info(f"带标注结果图保存至: {labeled_path}")
                        
                    except Exception as e:
                        self.logger.warning(f"生成带标注结果图失败: {e}")
                
                # 3. 关闭原始图形
                plt.close(fig)
            
            # 3. 第三张图：分割掩码图
            if self.config['output']['save_mask'] and mask_all is not None and np.max(mask_all) > 0:
                mask_path = output_dir / "segmentation_mask.png"
                mask_uint8 = (mask_all > 0).astype(np.uint8) * 255
                Image.fromarray(mask_uint8).save(mask_path)
                output_files.append(str(mask_path))
                self.logger.info(f"分割掩码保存至: {mask_path}")
            
            # 保存统计表格（与项目一格式相同）
            if self.config['output']['save_statistics'] and grain_data is not None and not grain_data.empty:
                # 确保是DataFrame
                if not isinstance(grain_data, pd.DataFrame):
                    grain_data = pd.DataFrame(grain_data)
                
                # 如果比例尺检测成功，计算真实面积
                if scale_detection_success and scale_factor:
                    if 'area' in grain_data.columns:
                        grain_data['area'] = pd.to_numeric(grain_data['area'], errors='coerce')
                        valid_areas = grain_data['area'].dropna()
                        
                        if len(valid_areas) > 0:
                            grain_data['area_um2'] = valid_areas * (scale_factor ** 2)
                            grain_data['diameter_um'] = 2 * np.sqrt(grain_data['area_um2'] / np.pi)
                
                # 保存CSV
                csv_path = output_dir / "grain_statistics.csv"
                grain_data.to_csv(csv_path, index=False, encoding='utf-8')
                output_files.append(str(csv_path))
                self.logger.info(f"颗粒数据保存至: {csv_path}")
                
                # 保存JSON汇总信息
                if self.config['output']['save_summary']:
                    summary = self._create_summary_dict(
                        image_path, image, grain_data, scale_detection_success, scale_factor
                    )
                    
                    json_path = output_dir / "summary.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(summary, f, indent=2, ensure_ascii=False)
                    output_files.append(str(json_path))
            
            # 保存性能指标
            if self.config['output']['save_performance']:
                performance_data = {
                    'image_name': Path(image_path).name,
                    'processing_time': time.time() - start_time,
                    'grains_count': len(all_grains),
                    'timestamp': datetime.now().isoformat(),
                    'image_size': f"{image.shape[1]}x{image.shape[0]}"
                }
                
                perf_path = output_dir / "performance.json"
                with open(perf_path, 'w', encoding='utf-8') as f:
                    json.dump(performance_data, f, indent=2, ensure_ascii=False)
                output_files.append(str(perf_path))
            
            result['output_files'] = output_files
            result['processing_time'] = time.time() - start_time
            self.logger.info(f"图片处理完成，耗时: {result['processing_time']:.2f}秒")
            
        except Exception as e:
            result['success'] = False
            result['error_message'] = str(e)
            result['processing_time'] = time.time() - start_time
            self.logger.error(f"图片处理失败: {image_path}")
            self.logger.error(f"错误信息: {e}")
            self.logger.error(traceback.format_exc())
            
        return result
    
    def _load_image_safely(self, image_path: str) -> Optional[np.ndarray]:
        """安全加载图片（多重回退机制）"""
        methods = [
            self._load_with_skimage,
            self._load_with_pil,
            self._load_with_opencv,
            self._load_with_binary
        ]
        
        for method in methods:
            try:
                image = method(image_path)
                if image is not None:
                    # 转换为RGB格式
                    image = convert_to_rgb(image)
                    # 归一化到0-255
                    image = normalize_image(image)
                    return image
            except Exception as e:
                self.logger.debug(f"图片加载方法失败: {method.__name__} - {e}")
                continue
        
        return None
    
    def _load_with_skimage(self, image_path: str) -> Optional[np.ndarray]:
        """使用skimage加载"""
        try:
            from skimage import io
            image = io.imread(image_path)
            return image
        except:
            return None
    
    def _load_with_pil(self, image_path: str) -> Optional[np.ndarray]:
        """使用PIL加载"""
        try:
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            return np.array(pil_image)
        except:
            return None
    
    def _load_with_opencv(self, image_path: str) -> Optional[np.ndarray]:
        """使用OpenCV加载"""
        try:
            import cv2
            img_bgr = cv2.imread(image_path)
            if img_bgr is not None:
                return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except:
            return None
    
    def _load_with_binary(self, image_path: str) -> Optional[np.ndarray]:
        """使用二进制加载"""
        try:
            with open(image_path, 'rb') as f:
                data = f.read()
                pil_image = Image.open(io.BytesIO(data))
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                return np.array(pil_image)
        except:
            return None
    
    def create_output_structure(self, image_path: Path) -> Path:
        """创建输出目录结构"""
        if self.config['output']['create_subdirs']:
            image_name = image_path.stem
            output_dir = self.output_root / "images" / image_name
        else:
            output_dir = self.output_root
        
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _create_summary_dict(self, image_path, image, grain_data, scale_success, scale_factor):
        """创建汇总信息字典（与项目一格式相同）"""
        summary = {
            'image_name': Path(image_path).name,
            'image_size': {
                'height': image.shape[0],
                'width': image.shape[1],
                'channels': image.shape[2]
            },
            'total_grains': int(len(grain_data)),
            'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'system_version': self.VERSION
        }
        
        # 面积统计
        if 'area' in grain_data.columns:
            grain_data['area'] = pd.to_numeric(grain_data['area'], errors='coerce')
            valid_areas = grain_data['area'].dropna()
            
            if len(valid_areas) > 0:
                summary['area_statistics_pixels'] = {
                    'total': float(valid_areas.sum()),
                    'average': float(valid_areas.mean()),
                    'min': float(valid_areas.min()),
                    'max': float(valid_areas.max()),
                    'std': float(valid_areas.std())
                }
        
        # 真实面积统计
        if scale_success:
            summary['scale_detection'] = {
                'success': True,
                'scale_factor_um_per_px': float(scale_factor)
            }
            
            if 'area_um2' in grain_data.columns:
                grain_data['area_um2'] = pd.to_numeric(grain_data['area_um2'], errors='coerce')
                valid_areas_um2 = grain_data['area_um2'].dropna()
                
                if len(valid_areas_um2) > 0:
                    summary['area_statistics_um2'] = {
                        'total': float(valid_areas_um2.sum()),
                        'average': float(valid_areas_um2.mean()),
                        'min': float(valid_areas_um2.min()),
                        'max': float(valid_areas_um2.max())
                    }
        
        return summary
    
    def show_system_info(self):
        """显示系统信息"""
        print("=" * 60)
        print(f"🏭 SuperFastSAM岩石颗粒自动分割系统 v{self.VERSION}")
        print("=" * 60)
        print(f"系统时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"输出目录: {self.output_root}")
        print(f"设备模式: {self.config['model_paths'].get('device', 'cpu')}")
        print(f"YOLO模型: {self.config['model_paths']['yolo']}")
        print(f"FastSAM模型: {self.config['model_paths']['fastsam']}")
        
        scale_config = self.config.get('scale_detection', {})
        if scale_config.get('enabled', False):
            print(f"比例尺检测: 已启用")
            print(f"已知长度: {scale_config.get('known_length_um', 'N/A')} μm")
        else:
            print(f"比例尺检测: 已禁用")
        
        if self.grain_label_config.get('enabled', True):
            print(f"颗粒标注: 已启用")
            bg_color = self.grain_label_config.get('bg_color', '')
            if bg_color is None or bg_color == '':
                print(f"标注样式: 无背景, {self.grain_label_config.get('font_size', 11)}px黄色文字")
            else:
                print(f"标注样式: 有背景, {self.grain_label_config.get('font_size', 9)}px黑色文字")
            print(f"最大标注数: {self.grain_label_config.get('max_labels', 1000)}")
        else:
            print(f"颗粒标注: 已禁用")
        
        print(f"性能监控: {'已启用' if self.config.get('performance', {}).get('enable_monitoring', False) else '已禁用'}")
        print("=" * 60)


if __name__ == "__main__":
    print("这是一个模块文件，请通过 run_super_fastsam.py 来启动系统")
    print("或者直接使用:")
    print("  from rock_super_fastsam import RockSegmentationSystemSuper")
    print("  system = RockSegmentationSystemSuper()")
    print("  system.initialize_models()")
    print("  system.process_single_image('图片路径')")