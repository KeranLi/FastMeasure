"""
MobileSAM生产级系统
文件名：rock_mobilesam_system.py
功能：完整的生产级岩石颗粒分割系统，统一管理所有功能
"""

import os
import sys
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import json
import yaml

import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 导入核心模块
from .yolo_mobilesam import MobileSegmentationPipeline
from .seg_tools import ImageProcessor, FileUtils, PerformanceMonitor

# 导入geometry模块
from geometry.grain_metric import GrainShapeMetrics
from geometry.config_loader import load_geometry_config
from geometry.export_csv import select_columns_for_grain_statistics_csv

# 导入比例尺检测模块
try:
    from scale_detector import ScaleDetector
    SCALE_DETECTOR_AVAILABLE = True
    print("成功导入比例尺检测模块")
except ImportError as e:
    SCALE_DETECTOR_AVAILABLE = False
    print(f"导入比例尺检测模块失败: {e}")

# 导入颗粒标注模块
try:
    from grain_marker import add_grain_labels, add_labels_with_config
    GRAIN_MARKER_AVAILABLE = True
    print("成功导入颗粒标注模块")
except ImportError as e:
    GRAIN_MARKER_AVAILABLE = False
    print(f"导入颗粒标注模块失败: {e}")

# 导入交互式模块（修改为增强版）
try:
    from mobilesam_interactive import PureMobileSAMInteractiveEnhanced  # 核心修改：替换为增强版类名
    INTERACTIVE_MODULE_AVAILABLE = True
    print("成功导入增强版交互式模块")
except ImportError as e:
    INTERACTIVE_MODULE_AVAILABLE = False
    print(f"导入增强版交互式模块失败: {e}")


class RockMobileSystem:
    """MobileSAM生产级岩石分割系统（统一主系统）"""
    
    VERSION = "1.1.0"
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化MobileSAM系统
        
        Args:
            config_path: 配置文件路径
        """
        print("=" * 70)
        print(f"MobileSAM岩石颗粒自动分割系统 v{self.VERSION}（统一架构）")
        print("=" * 70)
        
        # 加载配置文件
        self.config = self._load_config(config_path)
        
        # 新增：加载geometry配置
        self.geometry_config = load_geometry_config("geometry_config.yaml")
        
        # 设置输出目录
        self.output_root = Path(self.config['output']['root_dir'])
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # 初始化日志系统
        self._setup_logging()
        
        # 设置logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化核心流水线
        self.pipeline = MobileSegmentationPipeline(self.config)
        
        # 初始化比例尺检测器
        self.scale_detector = None
        if SCALE_DETECTOR_AVAILABLE:
            self._init_scale_detector()
        
        # 读取颗粒标注配置
        self.grain_label_config = self.config.get('grain_labeling', {})
        if 'bg_color' in self.grain_label_config and self.grain_label_config['bg_color'] == '':
            self.grain_label_config['bg_color'] = None
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        self.processing_history = []
        
        # 交互式系统实例（延迟初始化）
        self.interactive_system = None
        
        self.logger.info(f"MobileSAM系统初始化完成")
        self.logger.info(f"输出目录: {self.output_root}")
        self.logger.info(f"配置文件: {config_path}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        config_file = Path(config_path)
        
        # 如果配置文件不存在，使用默认配置
        if not config_file.exists():
            print(f"配置文件 {config_path} 不存在，使用默认配置")
            return self._get_default_config()
        
        try:
            config = FileUtils.safe_load_yaml(str(config_file), default={})
            print(f"配置文件加载成功: {config_file}")
            return config
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'model_paths': {
                'yolo': '../models/best.pt',
                'mobilesam': '../models/mobile_sam.pt',
                'device': 'cuda',
                'sam_type': 'vit_t'
            },
            'mobilesam_params': {
                'general': {
                    'points_per_side': 32,
                    'points_per_batch': 64,
                    'pred_iou_thresh': 0.88,
                    'stability_score_thresh': 0.95,
                    'box_nms_thresh': 0.7,
                    'crop_n_layers': 0,
                    'crop_n_points_downscale_factor': 1
                },
                'grain_optimization': {
                    'use_box_prompt': True,
                    'use_center_point': True,
                    'box_expansion': 1.15,
                    'min_mask_confidence': 0.3,
                    'min_mask_area': 10,
                    'max_mask_area_ratio': 0.8
                },
                'multi_scale': {
                    'enabled': True,
                    'scales': [0.8, 1.0, 1.2],
                    'merge_strategy': 'union'
                }
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
                'yolo_confidence': 0.15,
                'min_area': 30,
                'min_bbox_area': 15,
                'remove_edge_grains': False,
                'plot_results': True,
                'performance_monitoring': True
            },
            'output': {
                'root_dir': 'results_mobilesam',
                'create_subdirs': True,
                'save_visualization': True,
                'save_mask': True,
                'save_statistics': True,
                'save_summary': True,
                'save_performance': True,
                'save_debug_info': False
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
            }
        }
    
    def _init_scale_detector(self):
        """初始化比例尺检测器"""
        scale_config = self.config.get('scale_detection', {})
        if scale_config.get('enabled', False) and SCALE_DETECTOR_AVAILABLE:
            try:
                self.scale_detector = ScaleDetector(self.config)
                self.logger.info("比例尺检测器初始化成功")
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
        log_file = log_dir / f"mobile_sam_{timestamp}.log"
        
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
        """初始化AI模型"""
        self.performance_monitor.start_timing('initialize_models')
        
        model_paths = self.config['model_paths']
        device = model_paths.get('device', 'cuda')
        sam_type = model_paths.get('sam_type', 'vit_t')
        
        self.logger.info(f"初始化AI模型 (设备: {device}, 类型: {sam_type})")
        
        try:
            success = self.pipeline.load_models(
                yolo_path=model_paths['yolo'],
                mobilesam_path=model_paths['mobilesam'],
                device=device,
                model_type=sam_type
            )
            
            if success:
                self.performance_monitor.end_timing('initialize_models')
                self.logger.info("AI模型初始化成功")
                return True
            else:
                self.logger.error("AI模型初始化失败")
                return False
                
        except Exception as e:
            self.performance_monitor.end_timing('initialize_models')
            self.logger.error(f"模型初始化失败: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def process_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        处理单张岩石图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            处理结果字典
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
            'scale_detection_success': False,
            'system_version': self.VERSION
        }
        
        self.performance_monitor.start_timing('total_processing')
        
        try:
            # 检查图片文件
            self.logger.info(f"处理图片: {image_path}")
            
            # 创建输出目录
            output_dir = self.create_output_structure(Path(image_path))
            
            # 加载图片
            self.performance_monitor.start_timing('image_loading')
            image = ImageProcessor.load_image_safely(image_path)
            
            if image is None:
                result['error_message'] = "无法加载图片"
                result['processing_time'] = self.performance_monitor.timings.get('total_processing', {}).get('elapsed', 0)
                self.logger.error(f"图片加载失败: {image_path}")
                return result
            
            # 验证图像数据
            is_valid, valid_msg = ImageProcessor.validate_image(image)
            if not is_valid:
                result['error_message'] = valid_msg
                result['processing_time'] = self.performance_monitor.timings.get('total_processing', {}).get('elapsed', 0)
                self.logger.error(f"图像数据验证失败: {valid_msg}")
                return result
            
            self.performance_monitor.end_timing('image_loading')
            self.logger.info(f"图片加载成功: {image.shape}")
            
            # 检测比例尺
            scale_factor = None
            scale_detection_success = False
            
            if self.scale_detector and SCALE_DETECTOR_AVAILABLE:
                try:
                    self.performance_monitor.start_timing('scale_detection')
                    self.logger.info("检测图片中的比例尺...")
                    
                    scale_factor, scale_success = self.scale_detector.detect(image_path)
                    
                    if scale_success:
                        result['scale_factor'] = float(scale_factor)
                        result['scale_detection_success'] = True
                        scale_detection_success = True
                        self.logger.info(f"比例尺检测成功: {scale_factor:.4f} μm/px")
                    else:
                        self.logger.warning("比例尺检测失败，仅输出像素面积")
                    
                    self.performance_monitor.end_timing('scale_detection')
                except Exception as e:
                    self.logger.warning(f"比例尺检测异常: {e}")
            
            # 获取处理参数
            processing_config = self.config['processing']
            
            # 运行MobileSAM分割
            self.performance_monitor.start_timing('mobile_sam_segmentation')
            
            all_grains, labels, mask_all, grain_data, fig, ax = self.pipeline.mobile_sam_segmentation(
                image=image,
                conf_threshold=processing_config['yolo_confidence'],
                min_area=processing_config['min_area'],
                min_bbox_area=processing_config['min_bbox_area'],
                remove_edge_grains=processing_config['remove_edge_grains'],
                plot_image=processing_config['plot_results']
            )
            
            self.performance_monitor.end_timing('mobile_sam_segmentation')
            
            # 更新结果
            result['grains_count'] = len(all_grains)
            result['success'] = True
            
            # 保存结果文件
            output_files = []
            
            # 保存可视化结果
            if self.config['output']['save_visualization'] and fig is not None:
                # 1. 保存原始分割结果
                plot_path = output_dir / "segmentation_result.png"
                fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                output_files.append(str(plot_path))
                self.logger.info(f"原始结果图保存至: {plot_path}")
                
                # 2. 保存带标注的结果图
                if (GRAIN_MARKER_AVAILABLE and 
                    self.grain_label_config.get('enabled', True) and 
                    grain_data is not None and 
                    not grain_data.empty):
                    
                    try:
                        # 创建带标注的图像
                        fig_labeled, ax_labeled = plt.subplots(figsize=(15, 10))
                        ax_labeled.imshow(image)
                        ax_labeled.axis('off')
                        
                        # 添加颗粒标注
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
                
                # 关闭原始图形
                plt.close(fig)
            
            # 3. 保存分割掩码
            if self.config['output']['save_mask'] and mask_all is not None and np.max(mask_all) > 0:
                mask_path = output_dir / "segmentation_mask.png"
                mask_uint8 = (mask_all > 0).astype(np.uint8) * 255
                Image.fromarray(mask_uint8).save(mask_path)
                output_files.append(str(mask_path))
                self.logger.info(f"分割掩码保存至: {mask_path}")
            
            # 保存统计表格（新增：使用geometry配置）
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
                
                # 新增：计算几何形状参数
                try:
                    shape_calculator = GrainShapeMetrics(grain_data)
                    grain_data = shape_calculator.compute_all_metrics()
                    self.logger.info("几何参数计算完成")
                except Exception as e:
                    self.logger.warning(f"几何参数计算失败: {e}")
                
                # 保存CSV
                csv_path = output_dir / "grain_statistics.csv"
                
                grain_data_to_save = select_columns_for_grain_statistics_csv(
                    grain_data,
                    self.geometry_config,
                    strict=False
                )
                
                grain_data_to_save.to_csv(csv_path, index=False, encoding="utf-8")
                output_files.append(str(csv_path))
                self.logger.info(f"颗粒数据保存至: {csv_path}")
                
                # 保存JSON汇总信息
                if self.config['output']['save_summary']:
                    summary = self._create_summary_dict(
                        image_path, image, grain_data, scale_detection_success, scale_factor
                    )
                    
                    json_path = output_dir / "summary.json"
                    FileUtils.safe_save_json(summary, str(json_path))
                    output_files.append(str(json_path))
            
            # 保存性能数据
            if self.config['output']['save_performance']:
                performance_data = self.pipeline.get_performance()
                perf_path = output_dir / "performance.json"
                FileUtils.safe_save_json(performance_data, str(perf_path))
                output_files.append(str(perf_path))
            
            # 保存调试信息（如果需要）
            if self.config['output'].get('save_debug_info', False):
                debug_info = {
                    'image_shape': image.shape,
                    'num_yolo_boxes': len(all_grains),
                    'scale_factor': scale_factor,
                    'config': self.config
                }
                
                debug_path = output_dir / "debug_info.json"
                FileUtils.safe_save_json(debug_info, str(debug_path))
                output_files.append(str(debug_path))
            
            result['output_files'] = output_files
            
        except Exception as e:
            result['success'] = False
            result['error_message'] = str(e)
            self.logger.error(f"图片处理失败: {image_path}")
            self.logger.error(f"错误信息: {e}")
            self.logger.error(traceback.format_exc())
        
        finally:
            # 结束总计时
            self.performance_monitor.end_timing('total_processing')
            
            # 计算总处理时间
            total_time = self.performance_monitor.timings.get('total_processing', {}).get('elapsed', 0)
            result['processing_time'] = total_time
            
            # 保存性能指标
            result['performance_metrics'] = self.performance_monitor.get_summary()
            
            # 记录处理历史
            self.processing_history.append(result.copy())
            
            self.logger.info(f"图片处理完成，耗时: {total_time:.2f}秒")
        
        return result
    
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
        """创建汇总信息字典"""
        summary = {
            'image_name': Path(image_path).name,
            'image_size': {
                'height': image.shape[0],
                'width': image.shape[1],
                'channels': image.shape[2]
            },
            'total_grains': int(len(grain_data)),
            'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'system_version': self.VERSION,
            'model_type': 'MobileSAM'
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
    
    def batch_process(self, input_folder: str) -> Dict[str, Any]:
        """
        批量处理图片
        
        Args:
            input_folder: 输入文件夹路径
            
        Returns:
            批量处理结果
        """
        self.logger.info(f"开始批量处理: {input_folder}")
        
        # 查找图片文件
        input_path = Path(input_folder)
        if not input_path.exists():
            self.logger.error(f"输入文件夹不存在: {input_folder}")
            return {'success': False, 'error': '输入文件夹不存在'}
        
        # 获取支持的图片格式
        supported_formats = self.config['batch_processing']['supported_formats']
        
        # 查找所有图片文件
        image_files = []
        for format_ext in supported_formats:
            image_files.extend(input_path.rglob(f"*{format_ext}"))
            image_files.extend(input_path.rglob(f"*{format_ext.upper()}"))
        
        image_files = list(set(image_files))
        
        if not image_files:
            self.logger.error(f"未找到支持的图片文件: {input_folder}")
            return {'success': False, 'error': '未找到支持的图片文件'}
        
        self.logger.info(f"找到 {len(image_files)} 张图片")
        
        # 批量处理结果
        batch_results = {
            'total': len(image_files),
            'success': 0,
            'failed': 0,
            'failed_images': [],
            'total_grains': 0,
            'processing_start': datetime.now().isoformat(),
            'individual_results': []
        }
        
        # 处理每张图片
        for i, image_file in enumerate(image_files, 1):
            self.logger.info(f"处理进度: {i}/{len(image_files)} - {image_file.name}")
            
            # 检查文件是否损坏
            skip_corrupted = self.config['batch_processing']['skip_corrupted']
            if skip_corrupted:
                image_data = ImageProcessor.load_image_safely(str(image_file))
                if image_data is None:
                    self.logger.warning(f"跳过损坏图片: {image_file.name}")
                    batch_results['failed'] += 1
                    batch_results['failed_images'].append({
                        'path': str(image_file),
                        'error': '文件损坏',
                        'skipped': True
                    })
                    continue
            
            # 处理单张图片
            result = self.process_single_image(str(image_file))
            batch_results['individual_results'].append(result)
            
            if result['success']:
                batch_results['success'] += 1
                batch_results['total_grains'] += result['grains_count']
                self.logger.info(f"成功: {image_file.name} ({result['grains_count']}个颗粒)")
            else:
                batch_results['failed'] += 1
                batch_results['failed_images'].append({
                    'path': str(image_file),
                    'error': result['error_message'],
                    'skipped': False
                })
                self.logger.warning(f"失败: {image_file.name} - {result['error_message']}")
        
        batch_results['processing_end'] = datetime.now().isoformat()
        
        # 生成批量报告
        self._generate_batch_report(batch_results)
        
        self.logger.info(f"批量处理完成: {batch_results['success']}/{batch_results['total']} 成功")
        
        return batch_results
    
    def _generate_batch_report(self, batch_results: Dict[str, Any]):
        """生成批量处理报告"""
        report_path = self.output_root / "batch_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("MobileSAM批量处理报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"处理开始时间: {batch_results['processing_start']}\n")
            f.write(f"处理结束时间: {batch_results['processing_end']}\n")
            f.write(f"处理总时长: {self._calculate_duration(batch_results['processing_start'], batch_results['processing_end'])}\n\n")
            
            f.write(f"总图片数: {batch_results['total']}\n")
            f.write(f"成功处理: {batch_results['success']}\n")
            f.write(f"处理失败: {batch_results['failed']}\n")
            f.write(f"总检测颗粒数: {batch_results['total_grains']}\n\n")
            
            if batch_results['failed'] > 0:
                f.write("失败/跳过图片列表:\n")
                f.write("-" * 70 + "\n")
                for i, fail in enumerate(batch_results['failed_images'], 1):
                    f.write(f"{i}. 图片: {fail['path']}\n")
                    if fail.get('skipped', False):
                        f.write(f"   原因: 文件损坏，已跳过 - {fail['error']}\n")
                    else:
                        f.write(f"   原因: 处理失败 - {fail['error']}\n")
                    f.write("-" * 70 + "\n")
            
            successful_results = [r for r in batch_results['individual_results'] if r.get('success')]
            if successful_results:
                f.write("\n成功处理图片统计:\n")
                f.write("-" * 70 + "\n")
                for i, result in enumerate(successful_results, 1):
                    f.write(f"{i}. {result['image_name']}\n")
                    f.write(f"   颗粒数: {result['grains_count']}\n")
                    f.write(f"   处理时间: {result['processing_time']:.2f}秒\n")
                    if result.get('scale_detection_success', False):
                        f.write(f"   比例因子: {result.get('scale_factor', 'N/A')} μm/px\n")
                    if result['output_files']:
                        f.write(f"   输出文件: {len(result['output_files'])}个\n")
                f.write("-" * 70 + "\n")
        
        # 保存JSON格式的报告
        json_report_path = self.output_root / "batch_report.json"
        FileUtils.safe_save_json(batch_results, str(json_report_path))
        
        self.logger.info(f"批量处理报告保存至: {report_path}")
    
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
        print("=" * 70)
        print(f" MobileSAM岩石颗粒自动分割系统 v{self.VERSION}（统一架构）")
        print("=" * 70)
        print(f"系统时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"输出目录: {self.output_root}")
        print(f"设备模式: {self.config['model_paths'].get('device', 'cuda')}")
        print(f"YOLO模型: {self.config['model_paths']['yolo']}")
        print(f"MobileSAM模型: {self.config['model_paths']['mobilesam']}")
        print(f"模型类型: {self.config['model_paths'].get('sam_type', 'vit_t')}")
        
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
        
        print(f"交互式模块: {'已安装' if INTERACTIVE_MODULE_AVAILABLE else '未安装'}")
        print("=" * 70)
    
    def run_interactive_mode(self, image_path: str = None):
        """
        运行交互式模式（使用增强版）
        
        Args:
            image_path: 可选，如果提供则直接加载该图片
        """
        if not INTERACTIVE_MODULE_AVAILABLE:
            print("\n 增强版交互式模块不可用")
            print("请确保 mobilesam_interactive.py 文件中存在 PureMobileSAMInteractiveEnhanced 类")
            print("或者使用自动处理模式: python run_mobilesam.py --input 图片路径")
            return
        
        print("\n" + "=" * 60)
        print("   增强版交互式MobileSAM系统")  # 核心修改：更新提示信息
        print("=" * 60)
        print("特点：支持完整结果输出，包含统计信息、CSV、JSON等")  # 核心修改：添加增强版特点说明
        print("注意: 交互式模式需要图形界面支持")
        print("      在无GUI的服务器上可能无法正常运行")
        print("=" * 60)
        
        try:
            # 从配置中获取模型路径
            model_paths = self.config['model_paths']
            
            # 创建增强版交互式实例  # 核心修改：替换为增强版类名
            interactive_system = PureMobileSAMInteractiveEnhanced(
                model_path=model_paths['mobilesam'],
                device=model_paths.get('device', 'cpu'),
                model_type=model_paths.get('sam_type', 'vit_t')
            )
            
            # 运行交互模式
            interactive_system.run_interactive_mode(image_path)
            
            print("\n" + "=" * 60)
            print("交互式模式完成")
            print("=" * 60)
            
        except Exception as e:
            print(f" 增强版交互式模式运行失败: {e}")
            traceback.print_exc()
    
    def get_processing_history(self) -> List[Dict[str, Any]]:
        """获取处理历史"""
        return self.processing_history.copy()
    
    def clear_processing_history(self):
        """清空处理历史"""
        self.processing_history = []
        self.logger.info("处理历史已清空")


if __name__ == "__main__":
    print("这是一个模块文件，请通过 run_mobilesam.py 来启动系统")
    print("或者直接使用:")
    print("  from mobilesam.rock_mobilesam_system import RockMobileSystem")
    print("  system = RockMobileSystem()")
    print("  system.initialize_models()")
    print("  system.process_single_image('图片路径')")