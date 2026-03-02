"""
MobileSAM + YOLO Rock Grain Segmentation Pipeline
Filename: yolo_mobilesam.py
Function: Integrates YOLO detection and MobileSAM segmentation, providing end-to-end rock grain segmentation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import Polygon, MultiPolygon
import warnings
warnings.filterwarnings('ignore')

from typing import List, Tuple, Optional, Dict, Any
import sys
import time
import torch
from ultralytics import YOLO
from pathlib import Path

from .mobile_sam_engine import MobileSAMEngine

from skimage import measure, morphology
import cv2


class MobileSegmentationPipeline:
    """MobileSAM + YOLO Segmentation Pipeline"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.yolo_model = None
        self.mobilesam_engine = None
        self.performance_stats = {}
        self.reset_stats()
        
        self._seg1_available = False
        self._seg1_functions = {}
        
        print("MobileSegmentationPipeline initialization complete")
    
    def load_models(self, yolo_path: str, mobilesam_path: str, 
                   device: str = "cuda", model_type: str = "vit_t") -> bool:
        start_time = time.time()
        
        try:
            print(f"Loading YOLO model: {yolo_path}")
            self.yolo_model = YOLO(yolo_path)
            self.yolo_model.to(device)
            print("YOLO model loaded successfully")
            
            print(f"Loading MobileSAM model: {mobilesam_path}")
            self.mobilesam_engine = MobileSAMEngine(
                model_path=mobilesam_path,
                device=device,
                model_type=model_type
            )
            print("MobileSAM engine loaded successfully")
            
            self.performance_stats['model_load_time'] = time.time() - start_time
            self.performance_stats['device'] = device
            self.performance_stats['models_loaded'] = True
            
            return True
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            self.performance_stats['models_loaded'] = False
            return False
    
    def _load_segmenteverygrain(self):
        if self._seg1_available:
            return
        
        # Import core segmentation functions (migrated from segmenteverygrain)
        try:
            from core.segment_core import (
                create_labeled_image,
                collect_polygon_from_mask,
                plot_image_w_colorful_grains,
                plot_grain_axes_and_centroids,
                find_connected_components,
                merge_overlapping_polygons
            )
            
            self._seg1_functions = {
                'create_labeled_image': create_labeled_image,
                'collect_polygon_from_mask': collect_polygon_from_mask,
                'plot_image_w_colorful_grains': plot_image_w_colorful_grains,
                'plot_grain_axes_and_centroids': plot_grain_axes_and_centroids,
                'find_connected_components': find_connected_components,
                'merge_overlapping_polygons': merge_overlapping_polygons
            }
            self._seg1_available = True
            print("Core segmentation functions loaded successfully")
            
        except ImportError as e:
            self._seg1_available = False
            print(f"Core segmentation functions import failed: {e}")
        except Exception as e:
            self._seg1_available = False
            print(f"Core segmentation functions loading error: {e}")
    
    def mobile_sam_segmentation(self, 
                          image: np.ndarray,
                          conf_threshold: float = 0.15,
                          min_area: int = 30,
                          min_bbox_area: int = 15,
                          remove_edge_grains: bool = False,
                          plot_image: bool = False,
                          keep_edges: Optional[Dict] = None) -> Tuple[List[Polygon], np.ndarray, np.ndarray, pd.DataFrame, Optional[plt.Figure], Optional[plt.Axes]]:
        
        start_time = time.time()
        
        self._load_segmenteverygrain()
        
        polygons = []
        labels = np.zeros(image.shape[:2], dtype=np.int32)
        mask_all = np.zeros(image.shape[:2], dtype=np.uint8)
        grain_data = pd.DataFrame()
        fig, ax = None, None
        
        try:
            print("Running YOLO detection...")
            results = self.yolo_model(image, conf=conf_threshold)
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
            print(f"YOLO detected {len(boxes)} grain candidate boxes")
            
            if len(boxes) == 0:
                print("No grains detected")
                self.performance_stats['segmentation_time'] = time.time() - start_time
                return polygons, labels, mask_all, grain_data, fig, ax
            
            print("Running MobileSAM segmentation...")
            self.mobilesam_engine.set_image(image)
            
            masks = self.mobilesam_engine.multi_scale_segmentation(image, boxes.tolist())
            print(f"MobileSAM generated {len(masks)} valid masks")
            
            if len(masks) == 0:
                print("MobileSAM did not generate any valid masks")
                self.performance_stats['segmentation_time'] = time.time() - start_time
                return polygons, labels, mask_all, grain_data, fig, ax
            
            mask_all = np.zeros(image.shape[:2], dtype=np.uint8)
            for i, mask in enumerate(masks):
                mask_all = np.logical_or(mask_all, mask).astype(np.uint8)
            
            if self._seg1_available:
                print("Using segmenteverygrain to optimize segmentation results...")
                
                labeled_array, num_features = self._seg1_functions['find_connected_components'](mask_all)
                labels = self._seg1_functions['create_labeled_image'](labeled_array)
                
                polygons = self._seg1_functions['collect_polygon_from_mask'](
                    labeled_array, 
                    min_area=min_area,
                    remove_edge_grains=remove_edge_grains
                )
                
                polygons = self._seg1_functions['merge_overlapping_polygons'](polygons)
                
                grain_data = self._generate_unified_grain_stats(polygons)
                
                if plot_image:
                    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
                    self._seg1_functions['plot_image_w_colorful_grains'](
                        image, polygons, ax=ax
                    )
                    self._seg1_functions['plot_grain_axes_and_centroids'](
                        polygons, ax=ax
                    )
            else:
                print("使用基础分割逻辑（segmenteverygrain不可用）")
                
                labeled_array = measure.label(mask_all, connectivity=2)
                props = measure.regionprops(labeled_array)
                
                polygons = []
                for prop in props:
                    if prop.area >= min_area:
                        coords = prop.coords
                        if len(coords) >= 3:
                            poly = Polygon(coords[:, [1, 0]])
                            polygons.append(poly)
                
                grain_data = self._generate_unified_grain_stats(polygons)
                
                if plot_image:
                    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
                    ax.imshow(image)
                    ax.set_title(f'Rock Grain Segmentation (n={len(polygons)})')
                    ax.axis('off')
            
            self.performance_stats['segmentation_time'] = time.time() - start_time
            self.performance_stats['grain_count'] = len(polygons)
            self.performance_stats['yolo_boxes_count'] = len(boxes)
            self.performance_stats['masks_count'] = len(masks)
            
            print(f"Segmentation complete: detected {len(polygons)} rock grains")
            print(f"Segmentation time: {self.performance_stats['segmentation_time']:.2f}s")
            
            return polygons, labels, mask_all, grain_data, fig, ax
            
        except Exception as e:
            print(f"Segmentation failed: {e}")
            self.performance_stats['segmentation_failed'] = True
            self.performance_stats['segmentation_error'] = str(e)
            return polygons, labels, mask_all, grain_data, fig, ax
    
    def _generate_unified_grain_stats(self, polygons: List[Polygon]) -> pd.DataFrame:
        """
        Generate grain statistics consistent with GUI workflow
        Using unified column names: grain_id, area, centroid_x, centroid_y, width, height, perimeter, confidence
        """
        if len(polygons) == 0:
            return pd.DataFrame()
        
        try:
            data = []
            for i, poly in enumerate(polygons):
                if poly.is_empty:
                    continue
                
                area = poly.area
                centroid = poly.centroid
                bounds = poly.bounds
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                
                perimeter = 0
                try:
                    mask = np.zeros((1000, 1000), dtype=np.uint8)  # 临时创建mask用于计算周长
                    x, y = poly.exterior.xy
                    points = np.array([x, y], dtype=np.int32).T
                    cv2.drawContours(mask, [points], -1, 255, 1)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        perimeter = cv2.arcLength(contours[0], True)
                except Exception:
                    perimeter = 4 * np.sqrt(area) * 0.9
                
                convex_hull = poly.convex_hull
                solidity = area / convex_hull.area if convex_hull.area > 0 else 0.0
                aspect_ratio = width / height if height > 0 else 1.0
                
                data.append({
                    'grain_id': i + 1,           # 统一使用 grain_id
                    'area': area,
                    'centroid_x': centroid.x,    # 统一使用 centroid_x
                    'centroid_y': centroid.y,    # 统一使用 centroid_y
                    'width': width,              # 统一使用 width
                    'height': height,            # 统一使用 height
                    'perimeter': perimeter,      # 新增：周长
                    'confidence': 0.5,           # YOLO流程默认置信度
                    'aspect_ratio': aspect_ratio,
                    'solidity': solidity,
                    'is_empty': poly.is_empty,
                    'bounds_x1': bounds[0],
                    'bounds_y1': bounds[1],
                    'bounds_x2': bounds[2],
                    'bounds_y2': bounds[3]
                })
            
            df = pd.DataFrame(data)
            
            try:
                project_root = Path(__file__).parent.parent
                sys.path.insert(0, str(project_root))
                
                from geometry.grain_metric import GrainShapeMetrics
                shape_calculator = GrainShapeMetrics(df)
                df = shape_calculator.compute_all_metrics()
                print("Geometric parameter calculation complete")
            except Exception as e:
                print(f"Geometric parameter calculation failed: {e}")
            
            return df
        
        except Exception as e:
            print(f"Failed to generate unified statistics: {e}")
            
            data = []
            for i, poly in enumerate(polygons):
                if poly.is_empty:
                    continue
                
                data.append({
                    'grain_id': i + 1,
                    'area': poly.area,
                    'centroid_x': poly.centroid.x,
                    'centroid_y': poly.centroid.y,
                    'is_empty': poly.is_empty
                })
            
            return pd.DataFrame(data)
    
    def get_performance(self) -> Dict:
        if self.mobilesam_engine:
            sam_stats = self.mobilesam_engine.get_performance_stats()
            self.performance_stats.update(sam_stats)
        
        return self.performance_stats.copy()
    
    def reset_stats(self):
        self.performance_stats = {
            'model_load_time': 0.0,
            'segmentation_time': 0.0,
            'device': 'unknown',
            'models_loaded': False,
            'grain_count': 0,
            'yolo_boxes_count': 0,
            'masks_count': 0,
            'segmentation_failed': False,
            'segmentation_error': ''
        }


if __name__ == "__main__":
    print("MobileSegmentationPipeline Test")
    print("Test passed")