"""
岩石颗粒自动分割与统计系统 - 核心主程序（集成比例尺功能）
文件名：rock.py
功能：批量处理岩石显微图像，自动检测、分割并统计颗粒信息
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
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# 导入比例尺检测模块
from scale_detector import ScaleDetector

# 导入颗粒标注模块
try:
    from grain_marker import add_grain_labels, add_labels_with_config
    GRAIN_MARKER_AVAILABLE = True
    print("✅ 成功导入颗粒标注模块")
except ImportError as e:
    print(f"⚠️ 导入颗粒标注模块失败: {e}")
    GRAIN_MARKER_AVAILABLE = False
    add_grain_labels = None
    add_labels_with_config = None

# 允许加载截断的图片
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ===== 【导入路径修正】 =====
# 导入项目流水线函数
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
try:
    from segmenteverygrain.yolo_sam_segmentation import yolo_sam_segmentation
    print("✅ 成功导入分割流水线函数")
except ImportError as e:
    print(f" 导入分割流水线函数失败: {e}")
    print("请确保 segmenteverygrain 模块在正确的位置")
    sys.exit(1)


class RockSegmentationSystem:
    """岩石分割系统主类（集成比例尺功能）"""
    
    def __init__(self, config_path: str = "new/config.yaml"):
        """
        初始化岩石分割系统（带比例尺检测）
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置文件
        self.config = self._load_config(config_path)
        
        # 设置输出目录
        self.output_root = Path(self.config['output']['root_dir'])
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # 初始化日志系统
        self._setup_logging()
        
        # 设置 logger
        self.logger = logging.getLogger(__name__)
        
        # 初始化模型
        self.yolo_model = None
        self.sam_predictor = None
        self.device = None
        
        # 初始化比例尺检测器
        self.scale_detector = None
        self._init_scale_detector()
        
        # 读取颗粒标注配置
        self.grain_label_config = self.config.get('grain_labeling', {})
        
        # 处理空字符串背景为None
        if 'bg_color' in self.grain_label_config and self.grain_label_config['bg_color'] == '':
            self.grain_label_config['bg_color'] = None
        
        self.logger.info("岩石分割系统初始化完成")
    
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
            print(f"配置文件加载成功: {config_file}")
            return config
        except Exception as e:
            print(f" 配置文件加载失败: {e}")
            print(" 使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'model_paths': {
                'yolo': 'models/best.pt',
                'sam': 'models/sam_vit_h_4b8939.pth',
                'sam_type': 'vit_h'
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
                'confidence_threshold': 0.25,
                'min_area': 30,
                'min_bbox_area': 20,
                'remove_edge_grains': False,
                'plot_results': True
            },
            'output': {
                'root_dir': 'results',
                'create_subdirs': True,
                'save_visualization': True,
                'save_mask': True,
                'save_statistics': True,
                'save_summary': True
            },
            'batch_processing': {
                'supported_formats': ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp'],
                'skip_corrupted': True,
                'log_errors': True
            },
            'logging': {
                'level': 'INFO',
                'save_to_file': True,
                'show_in_console': True
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
            }
        }
    
    def _init_scale_detector(self):
        """初始化比例尺检测器"""
        scale_config = self.config.get('scale_detection', {})
        if scale_config.get('enabled', False):
            try:
                self.scale_detector = ScaleDetector(self.config)
                self.logger.info(" 比例尺检测器初始化成功")
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
        log_file = log_dir / f"rock_segmentation_{timestamp}.log"
        
        log_level = getattr(logging, log_config['level'])
        
        logger = logging.getLogger()
        logger.setLevel(log_level)
        
        logger.handlers.clear()
        
        if log_config['save_to_file']:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        if log_config['show_in_console']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
    
    def initialize_models(self) -> bool:
        """初始化YOLO和SAM模型"""
        self.logger.info("=" * 50)
        self.logger.info("开始初始化AI模型...")
        
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.logger.info(f"检测到GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            self.logger.warning("未检测到GPU，将使用CPU模式，速度会较慢")
        
        model_paths = self.config['model_paths']
        
        try:
            # 加载YOLO模型
            yolo_path = model_paths['yolo']
            if not Path(yolo_path).exists():
                self.logger.error(f"YOLO模型文件不存在: {yolo_path}")
                return False
            
            self.logger.info(f"加载YOLO模型: {yolo_path}")
            self.yolo_model = YOLO(yolo_path)
            self.logger.info(" YOLO模型加载成功")
            
            # 加载SAM模型
            sam_path = model_paths['sam']
            if not Path(sam_path).exists():
                self.logger.error(f"SAM模型文件不存在: {sam_path}")
                return False
            
            self.logger.info(f"加载SAM模型: {model_paths['sam_type']}")
            sam = sam_model_registry[model_paths['sam_type']](
                checkpoint=sam_path
            )
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
            self.logger.info(" SAM模型加载成功")
            
            self.logger.info("=" * 50)
            return True
            
        except Exception as e:
            self.logger.error(f"模型初始化失败: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def check_image_file(self, image_path: str) -> Tuple[bool, str]:
        """检查图片文件是否有效"""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True, "图片文件有效"
        except Exception as e:
            return False, f"图片文件损坏: {str(e)}"
    
    def find_image_files(self, input_path: str) -> List[Path]:
        """查找输入路径中的所有图片文件"""
        input_path = Path(input_path)
        image_files = []
        
        if not input_path.exists():
            self.logger.error(f"输入路径不存在: {input_path}")
            return image_files
        
        supported_formats = self.config['batch_processing']['supported_formats']
        
        if input_path.is_file():
            suffix = input_path.suffix.lower()
            if suffix in supported_formats:
                image_files.append(input_path)
        else:
            for format_ext in supported_formats:
                image_files.extend(input_path.glob(f"*{format_ext}"))
                image_files.extend(input_path.glob(f"*{format_ext.upper()}"))
                image_files.extend(input_path.rglob(f"*{format_ext}"))
                image_files.extend(input_path.rglob(f"*{format_ext.upper()}"))
        
        image_files = list(set(image_files))
        
        self.logger.info(f"找到 {len(image_files)} 张图片文件")
        return image_files
    
    def create_output_structure(self, image_path: Path) -> Path:
        """为图片创建输出目录结构"""
        if self.config['output']['create_subdirs']:
            image_name = image_path.stem
            output_dir = self.output_root / "images" / image_name
        else:
            output_dir = self.output_root
        
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def process_single_image(self, image_path: str) -> Dict:
        """
        处理单张岩石图片（集成比例尺检测）
        """
        result = {
            'image_path': str(image_path),
            'image_name': Path(image_path).name,
            'success': False,
            'grains_count': 0,
            'error_message': None,
            'output_files': [],
            'processing_time': 0,
            'timestamp': datetime.now().isoformat(),
            'scale_factor': None,
            'scale_detection_success': False
        }
        
        start_time = time.time()
        
        try:
            # 检查图片文件
            is_valid, message = self.check_image_file(image_path)
            if not is_valid:
                result['error_message'] = message
                result['processing_time'] = time.time() - start_time
                self.logger.warning(f"图片文件无效: {image_path} - {message}")
                return result
            
            # 创建输出目录
            output_dir = self.create_output_structure(Path(image_path))
            
            # 加载图片
            self.logger.info(f"处理图片: {image_path}")
            image = np.array(Image.open(image_path).convert('RGB'))
            self.logger.info(f"图片尺寸: {image.shape}")
            
            # 检测比例尺
            scale_factor = None
            scale_detection_success = False
            
            if self.scale_detector:
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
            
            # 运行分割流水线
            all_grains, labels, mask_all, grain_data, fig, ax = yolo_sam_segmentation(
                image=image,
                yolo_model=self.yolo_model,
                sam_predictor=self.sam_predictor,
                conf_threshold=processing_config['confidence_threshold'],
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
            
            # 保存可视化结果
            if self.config['output']['save_visualization'] and fig is not None:
                # 1. 保存原始结果图
                plot_path = output_dir / "segmentation_result.png"
                fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                output_files.append(str(plot_path))
                self.logger.info(f"原始结果图保存至: {plot_path}")
                
                # 2. 保存带标注的结果图（基于保存的原始图）
                if (GRAIN_MARKER_AVAILABLE and 
                    self.grain_label_config.get('enabled', True) and 
                    grain_data is not None and 
                    not grain_data.empty):
                    
                    try:
                        # 加载刚刚保存的图片
                        saved_image = mpimg.imread(plot_path)
                        
                        # 创建新的图形
                        fig_labeled, ax_labeled = plt.subplots(figsize=(15, 10))
                        ax_labeled.imshow(saved_image)
                        ax_labeled.axis('off')
                        
                        # 计算坐标缩放因子（原始图像 -> 保存的图像）
                        original_height, original_width = image.shape[:2]
                        saved_height, saved_width = saved_image.shape[:2]
                        
                        scale_x = saved_width / original_width
                        scale_y = saved_height / original_height
                        
                        # 创建调整后的颗粒数据副本
                        if 'centroid-0' in grain_data.columns and 'centroid-1' in grain_data.columns:
                            adjusted_grain_data = grain_data.copy()
                            
                            # 确保是数值类型
                            adjusted_grain_data['centroid-0'] = pd.to_numeric(adjusted_grain_data['centroid-0'], errors='coerce')
                            adjusted_grain_data['centroid-1'] = pd.to_numeric(adjusted_grain_data['centroid-1'], errors='coerce')
                            
                            # 调整质心坐标（原始图像坐标 -> 保存图片坐标）
                            adjusted_grain_data['centroid-0'] = adjusted_grain_data['centroid-0'] * scale_y
                            adjusted_grain_data['centroid-1'] = adjusted_grain_data['centroid-1'] * scale_x
                            
                            # 过滤掉无效坐标
                            adjusted_grain_data = adjusted_grain_data.dropna(subset=['centroid-0', 'centroid-1'])
                        else:
                            adjusted_grain_data = grain_data
                        
                        # 使用配置函数添加标注
                        if add_labels_with_config:
                            ax_labeled = add_labels_with_config(
                                ax=ax_labeled,
                                grain_data=adjusted_grain_data,
                                image_shape=saved_image.shape,
                                config=self.grain_label_config
                            )
                        else:
                            # 使用直接参数调用
                            ax_labeled = add_grain_labels(
                                ax=ax_labeled,
                                grain_data=adjusted_grain_data,
                                image_shape=saved_image.shape,
                                scale_factor=scale_factor if scale_detection_success else None,
                                font_size=self.grain_label_config.get('font_size', 11),
                                text_color=self.grain_label_config.get('text_color', 'yellow'),
                                bg_color=self.grain_label_config.get('bg_color', None),
                                show_area=self.grain_label_config.get('show_area', True),
                                max_labels=self.grain_label_config.get('max_labels', 1000),
                                min_area=self.grain_label_config.get('min_area', 0),
                                text_outline=self.grain_label_config.get('text_outline', True),
                                outline_color=self.grain_label_config.get('outline_color', 'black'),
                                outline_width=self.grain_label_config.get('outline_width', 2.0)
                            )
                        
                        # 隐藏坐标轴和边框
                        ax_labeled.set_xticks([])
                        ax_labeled.set_yticks([])
                        ax_labeled.set_xlim([0, saved_image.shape[1]])
                        ax_labeled.set_ylim([saved_image.shape[0], 0])
                        plt.tight_layout()
                        
                        # 保存带标注的图
                        labeled_path = output_dir / "segmentation_labeled.png"
                        fig_labeled.savefig(labeled_path, dpi=300, bbox_inches='tight', 
                                           pad_inches=0, facecolor='white')
                        output_files.append(str(labeled_path))
                        plt.close(fig_labeled)
                        
                        self.logger.info(f"带标注结果图保存至: {labeled_path}")
                        self.logger.info(f"  基于图片: {plot_path.name} 添加标注")
                        self.logger.info(f"  坐标缩放: {scale_x:.2f}x (宽), {scale_y:.2f}x (高)")
                        
                    except Exception as e:
                        self.logger.warning(f"生成带标注结果图失败: {e}")
                        self.logger.error(traceback.format_exc())
                
                # 关闭原始图形
                plt.close(fig)
            
            # 保存掩码
            if self.config['output']['save_mask'] and mask_all is not None and np.max(mask_all) > 0:
                mask_path = output_dir / "segmentation_mask.png"
                Image.fromarray((mask_all * 255).astype(np.uint8)).save(mask_path)
                output_files.append(str(mask_path))
                self.logger.info(f"分割掩码保存至: {mask_path}")
            
            # 保存统计表格
            if self.config['output']['save_statistics'] and grain_data is not None and not grain_data.empty:
                if not isinstance(grain_data, pd.DataFrame):
                    self.logger.warning("grain_data不是DataFrame，尝试转换")
                    grain_data = pd.DataFrame(grain_data)
                
                # 如果比例尺检测成功，计算真实面积
                if scale_detection_success and scale_factor:
                    if 'area' in grain_data.columns:
                        grain_data['area'] = pd.to_numeric(grain_data['area'], errors='coerce')
                        
                        valid_areas = grain_data['area'].dropna()
                        if len(valid_areas) > 0:
                            grain_data['area_um2'] = valid_areas * (scale_factor ** 2)
                            
                            grain_data['diameter_um'] = 2 * np.sqrt(grain_data['area_um2'] / np.pi)
                            
                            columns = list(grain_data.columns)
                            if 'area' in columns:
                                area_index = columns.index('area')
                                if 'area_um2' in columns:
                                    columns.remove('area_um2')
                                if 'diameter_um' in columns:
                                    columns.remove('diameter_um')
                                
                                columns.insert(area_index + 1, 'area_um2')
                                columns.insert(area_index + 2, 'diameter_um')
                                grain_data = grain_data.reindex(columns=columns)
                
                csv_path = output_dir / "grain_statistics.csv"
                grain_data.to_csv(csv_path, index=False, encoding='utf-8')
                output_files.append(str(csv_path))
                self.logger.info(f"颗粒数据保存至: {csv_path}")
                
                # 保存JSON汇总信息
                if self.config['output']['save_summary']:
                    summary = {
                        'image_name': Path(image_path).name,
                        'image_size': {
                            'height': image.shape[0],
                            'width': image.shape[1],
                            'channels': image.shape[2]
                        },
                        'total_grains': int(len(grain_data)),
                        'processing_time': time.time() - start_time,
                        'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    if 'area' in grain_data.columns:
                        grain_data['area'] = pd.to_numeric(grain_data['area'], errors='coerce')
                        valid_areas = grain_data['area'].dropna()
                        if len(valid_areas) > 0:
                            summary['total_area'] = float(valid_areas.sum())
                            summary['average_area'] = float(valid_areas.mean())
                            summary['min_area'] = float(valid_areas.min())
                            summary['max_area'] = float(valid_areas.max())
                    
                    for col in ['major_axis_length', 'minor_axis_length']:
                        if col in grain_data.columns:
                            grain_data[col] = pd.to_numeric(grain_data[col], errors='coerce')
                            valid_values = grain_data[col].dropna()
                            summary[f'average_{col}'] = float(valid_values.mean()) if len(valid_values) > 0 else 0.0
                    
                    if scale_detection_success:
                        summary['scale_detection'] = {
                            'success': True,
                            'scale_factor_um_per_px': float(scale_factor),
                            'known_length_um': float(self.scale_detector.known_length_um)
                        }
                        
                        if 'area_um2' in grain_data.columns:
                            try:
                                grain_data['area_um2'] = pd.to_numeric(grain_data['area_um2'], errors='coerce')
                                valid_area_um2 = grain_data['area_um2'].dropna()
                                
                                if len(valid_area_um2) > 0:
                                    diameter_mean = 0.0
                                    if 'diameter_um' in grain_data.columns:
                                        grain_data['diameter_um'] = pd.to_numeric(grain_data['diameter_um'], errors='coerce')
                                        valid_diameter = grain_data['diameter_um'].dropna()
                                        if len(valid_diameter) > 0:
                                            diameter_mean = float(valid_diameter.mean())
                                    
                                    summary['real_area_statistics'] = {
                                        'total_area_um2': float(valid_area_um2.sum()),
                                        'average_area_um2': float(valid_area_um2.mean()),
                                        'min_area_um2': float(valid_area_um2.min()),
                                        'max_area_um2': float(valid_area_um2.max()),
                                        'average_diameter_um': diameter_mean
                                    }
                            except Exception as e:
                                self.logger.error(f"计算真实面积统计时出错: {e}")
                    
                    json_path = output_dir / "summary.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(summary, f, indent=2, ensure_ascii=False)
                    output_files.append(str(json_path))
            
            result['output_files'] = output_files
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            self.logger.info(f" 图片处理完成，耗时: {processing_time:.2f}秒")
            
        except Exception as e:
            result['success'] = False
            result['error_message'] = str(e)
            result['processing_time'] = time.time() - start_time
            self.logger.error(f" 图片处理失败: {image_path}")
            self.logger.error(f"错误信息: {e}")
            self.logger.error(traceback.format_exc())
            
        return result
    
    def process_batch(self, input_path: str) -> Dict:
        """
        批量处理图片
        """
        image_files = self.find_image_files(input_path)
        
        if not image_files:
            self.logger.error(f"未找到支持的图片文件: {input_path}")
            return {
                'total': 0,
                'success': 0,
                'failed': 0,
                'failed_images': [],
                'total_grains': 0,
                'total_area_um2': 0,
                'scale_detection_stats': {'success': 0, 'failed': 0},
                'processing_start': datetime.now().isoformat(),
                'processing_end': datetime.now().isoformat()
            }
        
        self.logger.info("=" * 50)
        self.logger.info(f"开始批量处理 {len(image_files)} 张图片")
        self.logger.info("=" * 50)
        
        results = {
            'total': len(image_files),
            'success': 0,
            'failed': 0,
            'failed_images': [],
            'total_grains': 0,
            'total_area_um2': 0,
            'scale_detection_stats': {'success': 0, 'failed': 0},
            'processing_start': datetime.now().isoformat(),
            'individual_results': []
        }
        
        skip_corrupted = self.config['batch_processing']['skip_corrupted']
        
        for i, image_file in enumerate(image_files, 1):
            self.logger.info(f"处理进度: {i}/{len(image_files)} - {image_file.name}")
            
            if skip_corrupted:
                is_valid, message = self.check_image_file(str(image_file))
                if not is_valid:
                    self.logger.warning(f"跳过损坏图片: {image_file.name} - {message}")
                    results['failed'] += 1
                    results['failed_images'].append({
                        'path': str(image_file),
                        'error': message,
                        'skipped': True
                    })
                    continue
            
            result = self.process_single_image(str(image_file))
            results['individual_results'].append(result)
            
            if result['success']:
                results['success'] += 1
                results['total_grains'] += result['grains_count']
                
                if result.get('scale_detection_success', False):
                    results['scale_detection_stats']['success'] += 1
                else:
                    results['scale_detection_stats']['failed'] += 1
                
                self.logger.info(f" 成功: {image_file.name} ({result['grains_count']}个颗粒)")
            else:
                results['failed'] += 1
                results['failed_images'].append({
                    'path': str(image_file),
                    'error': result['error_message'],
                    'skipped': False
                })
                self.logger.warning(f" 失败: {image_file.name} - {result['error_message']}")
        
        results['processing_end'] = datetime.now().isoformat()
        self._generate_batch_report(results)
        
        self.logger.info("=" * 50)
        self.logger.info(f"批量处理完成！")
        self.logger.info(f"成功: {results['success']}/{results['total']}")
        self.logger.info(f"失败: {results['failed']}/{results['total']}")
        self.logger.info(f"总颗粒数: {results['total_grains']}")
        
        if self.scale_detector:
            scale_success = results['scale_detection_stats']['success']
            scale_total = results['success']
            if scale_total > 0:
                self.logger.info(f"比例尺检测: {scale_success}/{scale_total} (成功率: {scale_success/scale_total*100:.1f}%)")
        
        self.logger.info("=" * 50)
        
        return results
    
    def _generate_batch_report(self, results: Dict):
        """生成批量处理报告"""
        report_path = self.output_root / "batch_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("岩石颗粒分割批量处理报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"处理开始时间: {results['processing_start']}\n")
            f.write(f"处理结束时间: {results['processing_end']}\n")
            f.write(f"处理总时长: {self._calculate_duration(results['processing_start'], results['processing_end'])}\n\n")
            
            f.write(f"总图片数: {results['total']}\n")
            f.write(f"成功处理: {results['success']}\n")
            f.write(f"处理失败: {results['failed']}\n")
            f.write(f"总检测颗粒数: {results['total_grains']}\n")
            
            if 'scale_detection_stats' in results:
                scale_stats = results['scale_detection_stats']
                f.write(f"比例尺检测成功: {scale_stats['success']}\n")
                f.write(f"比例尺检测失败: {scale_stats['failed']}\n")
            
            f.write("\n")
            
            if results['failed'] > 0:
                f.write("失败/跳过图片列表:\n")
                f.write("-" * 60 + "\n")
                for i, fail in enumerate(results['failed_images'], 1):
                    f.write(f"{i}. 图片: {fail['path']}\n")
                    if fail.get('skipped', False):
                        f.write(f"   原因: 文件损坏，已跳过 - {fail['error']}\n")
                    else:
                        f.write(f"   原因: 处理失败 - {fail['error']}\n")
                    f.write("-" * 60 + "\n")
            
            successful_results = [r for r in results['individual_results'] if r.get('success')]
            if successful_results:
                f.write("\n成功处理图片统计:\n")
                f.write("-" * 60 + "\n")
                for i, result in enumerate(successful_results, 1):
                    f.write(f"{i}. {result['image_name']}\n")
                    f.write(f"   颗粒数: {result['grains_count']}\n")
                    f.write(f"   处理时间: {result['processing_time']:.2f}秒\n")
                    if result.get('scale_detection_success', False):
                        f.write(f"   比例因子: {result.get('scale_factor', 'N/A')} μm/px\n")
                    if result['output_files']:
                        f.write(f"   输出文件: {len(result['output_files'])}个\n")
                f.write("-" * 60 + "\n")
        
        self.logger.info(f"批量处理报告保存至: {report_path}")
        
        json_report_path = self.output_root / "batch_report.json"
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        self.logger.info(f"JSON报告保存至: {json_report_path}")
    
    def _calculate_duration(self, start_iso: str, end_iso: str) -> str:
        """计算处理时长"""
        try:
            start_time = datetime.fromisoformat(start_iso)
            end_time = datetime.fromisoformat(end_iso)
            duration = end_time - start_time
            
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            seconds = duration.seconds % 60
            
            return f"{hours}小时{minutes}分钟{seconds}秒"
        except:
            return "未知"
    
    def show_system_info(self):
        """显示系统信息"""
        print("=" * 60)
        print("岩石颗粒自动分割系统")
        print("=" * 60)
        print(f"系统时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"输出目录: {self.output_root}")
        print(f"设备模式: {self.device}")
        print(f"YOLO模型: {self.config['model_paths']['yolo']}")
        print(f"SAM模型: {self.config['model_paths']['sam_type']}")
        
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
        
        print("=" * 60)
    
    def run_interactive_mode(self):
        """运行交互式模式"""
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox
            
            print("\n 交互式模式启动中...")
            
            root = tk.Tk()
            root.withdraw()
            
            mode = input("\n请选择模式:\n1. 处理单张图片\n2. 批量处理文件夹\n请输入选择(1/2): ").strip()
            
            if mode == "1":
                image_path = filedialog.askopenfilename(
                    title="选择岩石显微图像",
                    filetypes=[
                        ("图像文件", "*.tif *.tiff *.jpg *.jpeg *.png *.bmp"),
                        ("所有文件", "*.*")
                    ]
                )
                
                if not image_path:
                    print("未选择图片，退出交互模式")
                    return
                
                result = self.process_single_image(image_path)
                
                if result['success']:
                    print(f" 处理完成！检测到 {result['grains_count']} 个颗粒")
                    if result.get('scale_detection_success', False):
                        print(f" 比例因子: {result.get('scale_factor', 'N/A')} μm/px")
                    print(f" 结果保存在: {self.output_root}")
                    
                    open_folder = input("是否打开结果文件夹？(y/n): ").strip().lower()
                    if open_folder == 'y':
                        self._open_folder(self.output_root / "images" / Path(image_path).stem)
                else:
                    print(f" 处理失败: {result['error_message']}")
                    
            elif mode == "2":
                folder_path = filedialog.askdirectory(title="选择包含岩石图像的文件夹")
                
                if not folder_path:
                    print(" 未选择文件夹，退出交互模式")
                    return
                
                confirm = input(f"将处理文件夹: {folder_path}\n确认开始批量处理？(y/n): ").strip().lower()
                if confirm != 'y':
                    print(" 用户取消操作")
                    return
                
                results = self.process_batch(folder_path)
                
                print(f"\n批量处理完成！")
                print(f"成功: {results['success']}/{results['total']}")
                print(f"总颗粒数: {results['total_grains']}")
                if self.scale_detector:
                    scale_stats = results.get('scale_detection_stats', {})
                    print(f"比例尺检测成功: {scale_stats.get('success', 0)}")
                print(f" 详细报告保存在: {self.output_root}/batch_report.txt")
                
                open_report = input("是否打开处理报告？(y/n): ").strip().lower()
                if open_report == 'y':
                    report_path = self.output_root / "batch_report.txt"
                    if report_path.exists():
                        self._open_file(str(report_path))
                
            else:
                print("无效选择，退出交互模式")
                
        except ImportError:
            print(" 交互模式需要tkinter支持")
            print(" 在Linux上安装: sudo apt-get install python3-tk")
        except Exception as e:
            print(f" 交互模式运行失败: {e}")
    
    def _open_folder(self, folder_path: Path):
        """打开文件夹"""
        try:
            if sys.platform == "win32":
                os.startfile(str(folder_path))
            elif sys.platform == "darwin":
                os.system(f'open "{folder_path}"')
            else:
                os.system(f'xdg-open "{folder_path}"')
        except:
            print(f"无法自动打开文件夹，请手动访问: {folder_path}")
    
    def _open_file(self, file_path: str):
        """打开文件"""
        try:
            if sys.platform == "win32":
                os.startfile(file_path)
            elif sys.platform == "darwin":
                os.system(f'open "{file_path}"')
            else:
                os.system(f'xdg-open "{file_path}"')
        except:
            print(f"无法自动打开文件，请手动访问: {file_path}")


if __name__ == "__main__":
    print("这是一个模块文件，请通过 run.py 来启动系统")
    print("或者直接使用:")
    print("  from rock import RockSegmentationSystem")
    print("  system = RockSegmentationSystem()")
    print("  system.initialize_models()")
    print("  system.process_single_image('图片路径')")