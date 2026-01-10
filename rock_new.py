# Add the SAM model loading and prediction functionality within your class
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
from ultralytics import SAM  # For MobileSAM from Ultralytics

# Additional module for SAM (if you're using this for segmentation tasks)
from segment_anything import sam_model_registry, SamPredictor

# Your own modules
from scale_detector import ScaleDetector
try:
    from grain_marker import add_grain_labels, add_labels_with_config
    GRAIN_MARKER_AVAILABLE = True
    print("✅ 成功导入颗粒标注模块")
except ImportError as e:
    print(f"⚠️ 导入颗粒标注模块失败: {e}")
    GRAIN_MARKER_AVAILABLE = False
    add_grain_labels = None
    add_labels_with_config = None

class RockSegmentationSystem:
    """岩石分割系统主类（集成比例尺功能）"""
    
    def __init__(self, config_path: str = "new/config.yaml"):
        """
        初始化岩石分割系统（带比例尺检测）
        
        Args:
            config_path: 配置文件路径
        """
        # Load configuration, setup paths, etc.
        self.config = self._load_config(config_path)
        self.output_root = Path(self.config['output']['root_dir'])
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging, scale detector, etc.
        self._setup_logging()
        
        # Initialize models
        self.yolo_model = None
        self.sam_predictor = None
        self.mobile_sam_model = None  # Add MobileSAM model variable
        self.device = None
        
        # Initialize the scale detector
        self._init_scale_detector()
        
        # Load models (including MobileSAM)
        self.initialize_models()
        
    def initialize_models(self) -> bool:
        """初始化YOLO、SAM和MobileSAM模型"""
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
            # Load YOLO model
            yolo_path = model_paths['yolo']
            if not Path(yolo_path).exists():
                self.logger.error(f"YOLO模型文件不存在: {yolo_path}")
                return False
            self.logger.info(f"加载YOLO模型: {yolo_path}")
            self.yolo_model = YOLO(yolo_path)
            self.logger.info(" YOLO模型加载成功")
            
            # Load SAM model
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
            
            # Load MobileSAM model (new addition from ultralytics)
            mobile_sam_path = model_paths.get('mobile_sam', None)
            if mobile_sam_path and Path(mobile_sam_path).exists():
                self.logger.info(f"加载MobileSAM模型: {mobile_sam_path}")
                self.mobile_sam_model = SAM(mobile_sam_path)  # Use SAM directly from ultralytics
                self.mobile_sam_model.to(device=self.device)
                self.logger.info(" MobileSAM模型加载成功")
            else:
                self.logger.warning("MobileSAM模型路径未指定或文件不存在，跳过加载")
            
            self.logger.info("=" * 50)
            return True
            
        except Exception as e:
            self.logger.error(f"模型初始化失败: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def process_single_image(self, image_path: str) -> Dict:
        """
        处理单张岩石图片（集成比例尺检测和MobileSAM处理）
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
            # Check image validity
            is_valid, message = self.check_image_file(image_path)
            if not is_valid:
                result['error_message'] = message
                result['processing_time'] = time.time() - start_time
                self.logger.warning(f"图片文件无效: {image_path} - {message}")
                return result
            
            # Create output directory
            output_dir = self.create_output_structure(Path(image_path))
            
            # Load image
            image = np.array(Image.open(image_path).convert('RGB'))
            self.logger.info(f"图片尺寸: {image.shape}")
            
            # Scale detection (same as existing logic)
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
            
            # Segmentation using MobileSAM (or fallback to YOLO+SAM)
            all_grains = []
            if self.mobile_sam_model:
                self.logger.info("使用MobileSAM进行分割...")
                everything_results = self.mobile_sam_model(image)
                # Assuming MobileSAM returns results in the format [boxes, masks, etc.]
                for result in everything_results:
                    # You can access various result attributes such as boxes, masks, keypoints here
                    boxes = result.boxes  # Bounding boxes
                    masks = result.masks  # Segmentation masks
                    keypoints = result.keypoints  # Keypoints for poses (if applicable)
                    probs = result.probs  # Probabilities for classification
                    obb = result.obb  # Oriented bounding boxes (OBB)
                    result.show()  # Show the result on screen
                    result.save(filename="result.jpg")  # Save the result to disk
                    
                result['grains_count'] = len(everything_results)  # Count the number of grains
                result['success'] = True
                self.logger.info(f"MobileSAM分割成功, 检测到 {len(everything_results)} 个颗粒")
            else:
                # Fallback to YOLO+SAM pipeline
                self.logger.info("使用YOLO和SAM进行分割...")
                all_grains, labels, mask_all, grain_data, fig, ax = yolo_sam_segmentation(
                    image=image,
                    yolo_model=self.yolo_model,
                    sam_predictor=self.sam_predictor,
                    conf_threshold=self.config['processing']['confidence_threshold'],
                    min_area=self.config['processing']['min_area'],
                    min_bbox_area=self.config['processing']['min_bbox_area'],
                    remove_edge_grains=self.config['processing']['remove_edge_grains'],
                    plot_image=self.config['processing']['plot_results'],
                    class_id=None
                )
                result['grains_count'] = len(all_grains)
                result['success'] = True
                self.logger.info(f"YOLO和SAM分割成功, 检测到 {len(all_grains)} 个颗粒")
            
            # Save output files, visualization, statistics etc. (same logic)
            output_files = []
            if self.config['output']['save_visualization'] and fig is not None:
                plot_path = output_dir / "segmentation_result.png"
                fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                output_files.append(str(plot_path))
                self.logger.info(f"原始结果图保存至: {plot_path}")
            
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
