"""
UltraFastSAM Production System - Supports GUI Interactive Mode
File: rock_fastsam_system.py
Function: Complete grain segmentation system with GUI interactive mode support
"""

import os
import sys
from pathlib import Path
import time
import logging
import traceback
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import json
import yaml

import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Add project root directory to system path to enable imports from project root
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

# Import core modules
try:
    from fastsam.yolo_fastsam import UltraSegmentationPipeline
    from fastsam.seg_tools import ImageProcessor, FileUtils, PerformanceMonitor
except ImportError as e:
    print(f"Failed to import core modules: {e}")
    raise

# Import scale detector module
try:
    from scale_detector import ScaleDetector
    SCALE_DETECTOR_AVAILABLE = True
    print("Successfully imported scale detector module")
except ImportError as e:
    SCALE_DETECTOR_AVAILABLE = False
    print(f"Failed to import scale detector module: {e}")

# Import grain marker module
try:
    from grain_marker import add_grain_labels, add_labels_with_config
    GRAIN_MARKER_AVAILABLE = True
    print("Successfully imported grain marker module")
except ImportError as e:
    GRAIN_MARKER_AVAILABLE = False
    print(f"Failed to import grain marker module: {e}")

# Import various geometric dimension calculation functions
try:
    from geometry.grain_metric import GrainShapeMetrics
    from geometry.config_loader import load_geometry_config
    from geometry.export_csv import select_columns_for_grain_statistics_csv
    GEOMETRY_AVAILABLE = True
    print("Successfully imported geometry module")
except ImportError as e:
    GEOMETRY_AVAILABLE = False
    print(f"Failed to import geometry module: {e}")

# Import FastSAM interactive module - now imported from project root directory
try:
    # 从项目根目录导入（新位置）
    from fastsam_interactive import PureFastSAMInteractiveEnhanced
    FASTSAM_INTERACTIVE_AVAILABLE = True
    print("Successfully imported FastSAM interactive module (from project root)")
except ImportError as e:
    # 如果从项目根目录导入失败，尝试从当前目录导入（旧位置）
    try:
        from .fastsam_interactive import PureFastSAMInteractiveEnhanced
        FASTSAM_INTERACTIVE_AVAILABLE = True
        print("Successfully imported FastSAM interactive module (from fastsam subdirectory)")
    except ImportError as e2:
        FASTSAM_INTERACTIVE_AVAILABLE = False
        print(f"Failed to import FastSAM interactive module: {e}")

class RockUltraSystem:
    """UltraFastSAM Production Rock Segmentation System - Supports GUI Interactive Mode"""
    
    VERSION = "1.0.0"
    
    def __init__(self, config_path: str = "configs/fastsam.yaml"):
        """
        Initialize UltraFastSAM system (supports GUI interactive mode)
        
        Args:
            config_path: Configuration file path
        """
        print("=" * 70)
        print(f"UltraFastSAM Rock Grain Auto-segmentation System v{self.VERSION}")
        print("=" * 70)
        
        # Load configuration file
        self.config = self._load_config(config_path)

        # Read geometry_config
        try:
            self.geometry_config = load_geometry_config("configs/geometry.yaml")
            print("Geometry configuration loaded successfully")
        except Exception as e:
            self.geometry_config = {}
            print(f"Failed to load geometry configuration: {e}")
        
        # Set up unified output directory structure
        # results/{mode}/{type}/ (e.g., results/fastsam/auto/)
        output_config = self.config.get('output', {})
        root_dir = output_config.get('root_dir', 'results')
        mode_subdir = output_config.get('mode_subdir', 'fastsam')
        type_subdir = output_config.get('type_subdir', 'auto')
        
        self.output_root = Path(root_dir) / mode_subdir / type_subdir
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Also ensure parent directories exist for other modes
        (Path(root_dir) / 'logs').mkdir(parents=True, exist_ok=True)
        (Path(root_dir) / 'temp').mkdir(parents=True, exist_ok=True)
        
        # Initialize logging system
        self._setup_logging()
        
        # 设置logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize core pipeline
        self.pipeline = UltraSegmentationPipeline(self.config)
        
        # Initialize scale detector
        self.scale_detector = None
        if SCALE_DETECTOR_AVAILABLE:
            self._init_scale_detector()
        
        # Read grain labeling configuration
        self.grain_label_config = self.config.get('grain_labeling', {})
        if 'bg_color' in self.grain_label_config and self.grain_label_config['bg_color'] == '':
            self.grain_label_config['bg_color'] = None
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.processing_history = []
        
        # GUI interactive system
        self.interactive_system = None
        
        self.logger.info(f"UltraFastSAM system initialization completed")
        self.logger.info(f"Output directory: {self.output_root}")
        self.logger.info(f"Configuration file: {config_path}")
        self.logger.info(f"GUI interactive mode: {'Available' if FASTSAM_INTERACTIVE_AVAILABLE else 'Unavailable'}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file"""
        config_file = Path(config_path)
        
        # If config file doesn't exist, use default config
        if not config_file.exists():
            print(f"Configuration file {config_path} does not exist, using default config")
            return self._get_default_config()
        
        try:
            config = FileUtils.safe_load_yaml(str(config_file), default={})
            print(f"Configuration file loaded successfully: {config_file}")
            return config
        except Exception as e:
            print(f"Failed to load configuration file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
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
                'min_area': 30,
                'min_bbox_area': 20,
                'remove_edge_grains': False,
                'plot_results': True,
                'performance_monitoring': True
            },
            'output': {
                'root_dir': 'results_ultra_fastsam',
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
        """Initialize scale detector"""
        scale_config = self.config.get('scale_detection', {})
        if scale_config.get('enabled', False) and SCALE_DETECTOR_AVAILABLE:
            try:
                self.scale_detector = ScaleDetector(self.config)
                self.logger.info("Scale detector initialized successfully")
            except Exception as e:
                self.logger.warning(f"Scale detector initialization failed: {e}")
                self.scale_detector = None
        else:
            self.scale_detector = None
            self.logger.info("Scale detection feature is disabled")
    
    def _setup_logging(self):
        """Setup logging system with unified directory structure"""
        log_config = self.config['logging']
        
        # Unified log directory: results/logs/{mode}/
        output_config = self.config.get('output', {})
        root_dir = output_config.get('root_dir', 'results')
        log_subdir = log_config.get('log_subdir', 'fastsam')
        
        log_dir = Path(root_dir) / 'logs' / log_subdir
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"fastsam_{timestamp}.log"
        
        log_level = getattr(logging, log_config['level'])
        
        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(log_level)
        logger.handlers.clear()
        
        # File handler
        if log_config['save_to_file']:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_formatter = logging.Formatter(log_config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # Console handler
        if log_config['show_in_console']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Logging system initialized, log file: {log_file}")
    
    def initialize_models(self) -> bool:
        """Initialize AI models"""
        self.performance_monitor.start_timing('initialize_models')
        
        model_paths = self.config['model_paths']
        device = model_paths.get('device', 'cpu')
        
        self.logger.info(f"Initializing AI models (device: {device})")
        
        try:
            success = self.pipeline.load_models(
                yolo_path=model_paths['yolo'],
                fastsam_path=model_paths['fastsam'],
                device=device
            )
            
            if success:
                self.performance_monitor.end_timing('initialize_models')
                self.logger.info("AI models initialized successfully")
                return True
            else:
                self.logger.error("AI models initialization failed")
                return False
                
        except Exception as e:
            self.performance_monitor.end_timing('initialize_models')
            self.logger.error(f"Model initialization failed: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def process_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process single rock image
        
        Args:
            image_path: Image path
            
        Returns:
            Processing result dictionary
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
            # Check image file
            self.logger.info(f"Processing image: {image_path}")
            
            # Create output directory
            output_dir = self.create_output_structure(Path(image_path))
            
            # Load image
            self.performance_monitor.start_timing('image_loading')
            image = ImageProcessor.load_image_safely(image_path)
            
            if image is None:
                result['error_message'] = "Unable to load image"
                result['processing_time'] = self.performance_monitor.timings.get('total_processing', {}).get('elapsed', 0)
                self.logger.error(f"Image loading failed: {image_path}")
                return result
            
            # Validate image data
            is_valid, valid_msg = ImageProcessor.validate_image(image)
            if not is_valid:
                result['error_message'] = valid_msg
                result['processing_time'] = self.performance_monitor.timings.get('total_processing', {}).get('elapsed', 0)
                self.logger.error(f"Image data validation failed: {valid_msg}")
                return result
            
            self.performance_monitor.end_timing('image_loading')
            self.logger.info(f"Image loaded successfully: {image.shape}")
            
            # Detect scale
            scale_factor = None
            scale_detection_success = False
            
            if self.scale_detector and SCALE_DETECTOR_AVAILABLE:
                try:
                    self.performance_monitor.start_timing('scale_detection')
                    self.logger.info("Detecting scale in image...")
                    
                    scale_factor, scale_success = self.scale_detector.detect(image_path)
                    
                    if scale_success:
                        result['scale_factor'] = float(scale_factor)
                        result['scale_detection_success'] = True
                        scale_detection_success = True
                        self.logger.info(f"Scale detection successful: {scale_factor:.4f} μm/px")
                    else:
                        self.logger.warning("Scale detection failed, outputting pixel area only")
                    
                    self.performance_monitor.end_timing('scale_detection')
                except Exception as e:
                    self.logger.warning(f"Scale detection exception: {e}")
            
            # Get processing parameters
            processing_config = self.config['processing']
            
            # Run UltraFastSAM segmentation
            self.performance_monitor.start_timing('ultra_segmentation')
            
            all_grains, labels, mask_all, grain_data, fig, ax = self.pipeline.ultra_segmentation(
                image=image,
                conf_threshold=processing_config['yolo_confidence'],
                min_area=processing_config['min_area'],
                min_bbox_area=processing_config['min_bbox_area'],
                remove_edge_grains=processing_config['remove_edge_grains'],
                plot_image=processing_config['plot_results']
            )
            
            self.performance_monitor.end_timing('ultra_segmentation')
            
            # 更新结果
            result['grains_count'] = len(all_grains)
            result['success'] = True

            # Calculate grain shape parameters
            if GEOMETRY_AVAILABLE and grain_data is not None and not grain_data.empty:
                try:
                    shape_calculator = GrainShapeMetrics(grain_data)
                    grain_data = shape_calculator.compute_all_metrics()
                    self.logger.info("Advanced geometric parameters calculation completed")
                except Exception as e:
                    self.logger.warning(f"Geometric parameters calculation failed: {e}")
            
            # Save result files
            output_files = []
            
            # Save visualization results
            if self.config['output']['save_visualization'] and fig is not None:
                # 1. Save original segmentation result
                plot_path = output_dir / "segmentation_result.png"
                fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                output_files.append(str(plot_path))
                self.logger.info(f"Original result saved to: {plot_path}")
                
                # 2. Save labeled result image
                if (GRAIN_MARKER_AVAILABLE and 
                    self.grain_label_config.get('enabled', True) and 
                    grain_data is not None and 
                    not grain_data.empty):
                    
                    try:
                        # Create labeled image
                        fig_labeled, ax_labeled = plt.subplots(figsize=(15, 10))
                        ax_labeled.imshow(image)
                        ax_labeled.axis('off')
                        
                        # Add grain labels
                        if 'add_labels_with_config' in globals():
                            ax_labeled = add_labels_with_config(
                                ax=ax_labeled,
                                grain_data=grain_data,
                                image_shape=image.shape,
                                config=self.grain_label_config
                            )
                        
                        # Hide axes and borders
                        ax_labeled.set_xticks([])
                        ax_labeled.set_yticks([])
                        ax_labeled.set_xlim([0, image.shape[1]])
                        ax_labeled.set_ylim([image.shape[0], 0])
                        plt.tight_layout()
                        
                        # Save labeled image
                        labeled_path = output_dir / "segmentation_labeled.png"
                        fig_labeled.savefig(labeled_path, dpi=300, bbox_inches='tight', 
                                           pad_inches=0, facecolor='white')
                        output_files.append(str(labeled_path))
                        plt.close(fig_labeled)
                        
                        self.logger.info(f"Labeled result saved to: {labeled_path}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to generate labeled result: {e}")
                
                # Close original figure
                plt.close(fig)
            
            # 3. Save segmentation mask
            if self.config['output']['save_mask'] and mask_all is not None and np.max(mask_all) > 0:
                mask_path = output_dir / "segmentation_mask.png"
                mask_uint8 = (mask_all > 0).astype(np.uint8) * 255
                Image.fromarray(mask_uint8).save(mask_path)
                output_files.append(str(mask_path))
                self.logger.info(f"Segmentation mask saved to: {mask_path}")
            
            # Save statistics table
            if self.config['output']['save_statistics'] and grain_data is not None and not grain_data.empty:
                # 确保是DataFrame
                if not isinstance(grain_data, pd.DataFrame):
                    grain_data = pd.DataFrame(grain_data)
                
                # If scale detection successful, calculate real area
                if scale_detection_success and scale_factor:
                    if 'area' in grain_data.columns:
                        grain_data['area'] = pd.to_numeric(grain_data['area'], errors='coerce')
                        valid_areas = grain_data['area'].dropna()
                        
                        if len(valid_areas) > 0:
                            grain_data['area_um2'] = valid_areas * (scale_factor ** 2)
                            grain_data['diameter_um'] = 2 * np.sqrt(grain_data['area_um2'] / np.pi)
                
                # 保存CSV
                csv_path = output_dir / "grain_statistics.csv"
                
                if GEOMETRY_AVAILABLE and self.geometry_config:
                    try:
                        grain_data_to_save = select_columns_for_grain_statistics_csv(
                            grain_data,
                            self.geometry_config,
                            strict=False
                        )
                        
                        if grain_data_to_save is not None and not grain_data_to_save.empty:
                            grain_data_to_save.to_csv(csv_path, index=False, encoding='utf-8')
                        else:
                            grain_data.to_csv(csv_path, index=False, encoding='utf-8')
                    except Exception as e:
                        self.logger.warning(f"Configuration filtering failed: {e}")
                        grain_data.to_csv(csv_path, index=False, encoding='utf-8')
                else:
                    grain_data.to_csv(csv_path, index=False, encoding='utf-8')
                
                output_files.append(str(csv_path))
                self.logger.info(f"Grain data saved to: {csv_path}")
                
                # Save JSON summary
                if self.config['output']['save_summary']:
                    summary = self._create_summary_dict(
                        image_path, image, grain_data, scale_detection_success, scale_factor
                    )
                    
                    json_path = output_dir / "summary.json"
                    FileUtils.safe_save_json(summary, str(json_path))
                    output_files.append(str(json_path))
                    self.logger.info(f"JSON summary saved to: {json_path}")
            
            # Save performance data
            if self.config['output']['save_performance']:
                performance_data = self.pipeline.get_performance()
                perf_path = output_dir / "performance.json"
                FileUtils.safe_save_json(performance_data, str(perf_path))
                output_files.append(str(perf_path))
                self.logger.info(f"Performance data saved to: {perf_path}")
            
            # Save debug info (if needed)
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
                self.logger.info(f"Debug info saved to: {debug_path}")
            
            result['output_files'] = output_files
            
        except Exception as e:
            result['success'] = False
            result['error_message'] = str(e)
            self.logger.error(f"Image processing failed: {image_path}")
            self.logger.error(f"Error message: {e}")
            self.logger.error(traceback.format_exc())
        
        finally:
            # End total timing
            self.performance_monitor.end_timing('total_processing')
            
            # Calculate total processing time
            total_time = self.performance_monitor.timings.get('total_processing', {}).get('elapsed', 0)
            result['processing_time'] = total_time
            
            # Save performance metrics
            result['performance_metrics'] = self.performance_monitor.get_summary()
            
            # Record processing history
            self.processing_history.append(result.copy())
            
            self.logger.info(f"Image processing completed, time elapsed: {total_time:.2f}s")
        
        return result
    
    def create_output_structure(self, image_path: Path) -> Path:
        """Create output directory structure"""
        if self.config['output']['create_subdirs']:
            image_name = image_path.stem
            output_dir = self.output_root / "images" / image_name
        else:
            output_dir = self.output_root
        
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _create_summary_dict(self, image_path, image, grain_data, scale_success, scale_factor):
        """Create summary dictionary"""
        summary = {
            'image_name': Path(image_path).name,
            'image_size': {
                'height': image.shape[0],
                'width': image.shape[1],
                'channels': image.shape[2]
            },
            'total_grains': int(len(grain_data)) if grain_data is not None and not grain_data.empty else 0,
            'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'system_version': self.VERSION
        }
        
        # Area statistics
        if grain_data is not None and not grain_data.empty and 'area' in grain_data.columns:
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
        
        # Real area statistics
        if scale_success:
            summary['scale_detection'] = {
                'success': True,
                'scale_factor_um_per_px': float(scale_factor)
            }
            
            if grain_data is not None and 'area_um2' in grain_data.columns:
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
        Batch process images
        
        Args:
            input_folder: Input folder path
            
        Returns:
            Batch processing results
        """
        self.logger.info(f"Starting batch processing: {input_folder}")
        
        # Find image files
        input_path = Path(input_folder)
        if not input_path.exists():
            self.logger.error(f"输入文件夹不存在: {input_folder}")
            return {'success': False, 'error': '输入文件夹不存在'}
        
        # Get supported image formats
        supported_formats = self.config['batch_processing']['supported_formats']
        
        # Find all image files
        image_files = []
        for format_ext in supported_formats:
            image_files.extend(input_path.rglob(f"*{format_ext}"))
            image_files.extend(input_path.rglob(f"*{format_ext.upper()}"))
        
        image_files = list(set(image_files))
        
        if not image_files:
            self.logger.error(f"No supported image files found: {input_folder}")
            return {'success': False, 'error': '未找到支持的图片文件'}
        
        self.logger.info(f"Found {len(image_files)} images")
        
        # Batch processing results
        batch_results = {
            'total': len(image_files),
            'success': 0,
            'failed': 0,
            'failed_images': [],
            'total_grains': 0,
            'processing_start': datetime.now().isoformat(),
            'individual_results': []
        }
        
        # Process each image
        for i, image_file in enumerate(image_files, 1):
            self.logger.info(f"Processing progress: {i}/{len(image_files)} - {image_file.name}")
            
        # Check if file is corrupted
            skip_corrupted = self.config['batch_processing']['skip_corrupted']
            if skip_corrupted:
                image_data = ImageProcessor.load_image_safely(str(image_file))
                if image_data is None:
                    self.logger.warning(f"Skipping corrupted image: {image_file.name}")
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
                self.logger.info(f"Success: {image_file.name} ({result['grains_count']} grains)")
            else:
                batch_results['failed'] += 1
                batch_results['failed_images'].append({
                    'path': str(image_file),
                    'error': result['error_message'],
                    'skipped': False
                })
                self.logger.warning(f"Failed: {image_file.name} - {result['error_message']}")
        
        batch_results['processing_end'] = datetime.now().isoformat()
        
        # Generate batch report
        self._generate_batch_report(batch_results)
        
        self.logger.info(f"Batch processing completed: {batch_results['success']}/{batch_results['total']} successful")
        
        return batch_results
    
    def _generate_batch_report(self, batch_results: Dict[str, Any]):
        """Generate batch processing report"""
        report_path = self.output_root / "batch_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("UltraFastSAM Batch Processing Report\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Processing start time: {batch_results['processing_start']}\n")
            f.write(f"Processing end time: {batch_results['processing_end']}\n")
            f.write(f"Total processing time: {self._calculate_duration(batch_results['processing_start'], batch_results['processing_end'])}\n\n")
            
            f.write(f"Total images: {batch_results['total']}\n")
            f.write(f"Successful: {batch_results['success']}\n")
            f.write(f"Failed: {batch_results['failed']}\n")
            f.write(f"Total grains detected: {batch_results['total_grains']}\n\n")
            
            if batch_results['failed'] > 0:
                f.write("Failed/Skipped images list:\n")
                f.write("-" * 70 + "\n")
                for i, fail in enumerate(batch_results['failed_images'], 1):
                    f.write(f"{i}. Image: {fail['path']}\n")
                    if fail.get('skipped', False):
                        f.write(f"   Reason: File corrupted, skipped - {fail['error']}\n")
                    else:
                        f.write(f"   Reason: Processing failed - {fail['error']}\n")
                    f.write("-" * 70 + "\n")
            
            successful_results = [r for r in batch_results['individual_results'] if r.get('success')]
            if successful_results:
                f.write("\nSuccessfully processed images statistics:\n")
                f.write("-" * 70 + "\n")
                for i, result in enumerate(successful_results, 1):
                    f.write(f"{i}. {result['image_name']}\n")
                    f.write(f"   Grains count: {result['grains_count']}\n")
                    f.write(f"   Processing time: {result['processing_time']:.2f}s\n")
                    if result.get('scale_detection_success', False):
                        f.write(f"   Scale factor: {result.get('scale_factor', 'N/A')} μm/px\n")
                    if result['output_files']:
                        f.write(f"   Output files: {len(result['output_files'])}\n")
                f.write("-" * 70 + "\n")
        
        # 保存JSON格式的报告
        json_report_path = self.output_root / "batch_report.json"
        FileUtils.safe_save_json(batch_results, str(json_report_path))
        
        self.logger.info(f"Batch processing report saved to: {report_path}")
    
    def _calculate_duration(self, start_iso: str, end_iso: str) -> str:
        """Calculate processing duration"""
        try:
            start_time = datetime.fromisoformat(start_iso)
            end_time = datetime.fromisoformat(end_iso)
            duration = end_time - start_time
            
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            seconds = duration.seconds % 60
            
            return f"{hours}h {minutes}m {seconds}s"
        except:
            return "Unknown"
    
    def show_system_info(self):
        """Display system information"""
        print("=" * 70)
        print(f"UltraFastSAM Rock Grain Auto-segmentation System v{self.VERSION}")
        print("=" * 70)
        print(f"System time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {self.output_root}")
        print(f"Device mode: {self.config['model_paths'].get('device', 'cpu')}")
        print(f"YOLO model: {self.config['model_paths']['yolo']}")
        print(f"FastSAM model: {self.config['model_paths']['fastsam']}")
        
        scale_config = self.config.get('scale_detection', {})
        if scale_config.get('enabled', False):
            print(f"Scale detection: Enabled")
            print(f"Known length: {scale_config.get('known_length_um', 'N/A')} μm")
        else:
            print(f"Scale detection: Disabled")
        
        if self.grain_label_config.get('enabled', True):
            print(f"Grain labeling: Enabled")
            bg_color = self.grain_label_config.get('bg_color', '')
            if bg_color is None or bg_color == '':
                print(f"Label style: No background, {self.grain_label_config.get('font_size', 11)}px yellow text")
            else:
                print(f"Label style: With background, {self.grain_label_config.get('font_size', 9)}px black text")
            print(f"Max labels: {self.grain_label_config.get('max_labels', 1000)}")
        else:
            print(f"Grain labeling: Disabled")
        
        print(f"GUI interactive mode: {'Available' if FASTSAM_INTERACTIVE_AVAILABLE else 'Unavailable'}")
        print("=" * 70)
    
    def run_interactive_mode(self, image_path: str = None):
        """Run FastSAM interactive mode"""
        if not FASTSAM_INTERACTIVE_AVAILABLE:
            print("FastSAM interactive module is not available")
            print("Please check if fastsam_interactive.py file exists")
            return
        
        print("Starting enhanced interactive FastSAM system...")
        
        try:
            model_paths = self.config['model_paths']
            
            # Check if model file exists
            fastsam_path = model_paths.get('fastsam', 'models/FastSAM-s.pt')
            if not Path(fastsam_path).exists():
                print(f"FastSAM model file does not exist: {fastsam_path}")
                print("Please check model file path or re-download the model")
                return
            
            # Create interactive system instance
            interactive_system = PureFastSAMInteractiveEnhanced(
                model_path=fastsam_path,
                device=model_paths.get('device', 'cpu')
            )
            
            # 运行交互式模式
            interactive_system.run_interactive_mode(image_path)
            
            # Save reference to interactive system
            self.interactive_system = interactive_system
            
        except Exception as e:
            print(f"Enhanced interactive mode failed: {e}")
            import traceback
            traceback.print_exc()
    
    def get_processing_history(self) -> List[Dict[str, Any]]:
        """Get processing history"""
        return self.processing_history.copy()
    
    def clear_processing_history(self):
        """Clear processing history"""
        self.processing_history = []
        self.logger.info("Processing history cleared")


if __name__ == "__main__":
    print("This is a module file. Please start the system via run_fastsam.py")
    print("Or use directly:")
    print("  from rock_fastsam_system import RockUltraSystem")
    print("  system = RockUltraSystem()")
    print("  system.initialize_models()")
    print("  system.process_single_image('<image_path>')")