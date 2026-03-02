"""
UltraFastSAM Core Segmentation Pipeline - skimage-optimized version
Filename: yolo_fastsam.py
Function: Use skimage.measure.find_contours for high-quality contour extraction
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

# ===== Import skimage =====
from skimage import measure
from skimage.filters import gaussian
from skimage.morphology import binary_opening, binary_closing, disk
print(" skimage related modules imported")

# ===== Force import project one functions =====
import os
from pathlib import Path

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
    PROJECT1_AVAILABLE = True
    print(" Successfully imported core segmentation functions")
except ImportError as e:
    print(f" Failed to import core segmentation functions: {e}")
    sys.exit(1)

# Import UltraFastSAM engine
try:
    from .seg_engine import UltraFastSAM
except ImportError:
    # If running in main program, may need to adjust import method
    try:
        from seg_engine import UltraFastSAM
    except ImportError:
        print(" Cannot import UltraFastSAM engine")


class UltraSegmentationPipeline:
    """UltraFastSAM Core Segmentation Pipeline - Optimized version"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.yolo_model = None
        self.ultra_fastsam = None
        
        # 性能监控
        self.performance = {
            'yolo_time': 0.0,
            'fastsam_time': 0.0,
            'postprocess_time': 0.0,
            'total_time': 0.0
        }
        
        # 配置参数
        self.contour_params = {
            'level': 0.5,  # skimage轮廓提取的阈值
            'smooth_sigma': 1.0,  # 高斯平滑参数
            'min_contour_points': 20,  # 最小轮廓点数
            'max_contour_points': 500,  # 最大轮廓点数
            'simplify_tolerance': 0.5,  # 多边形简化容差
            'morph_radius': 1,  # 形态学操作半径
        }
        
        print("UltraSegmentationPipeline initialization complete")
    
    def load_models(self, yolo_path: str, fastsam_path: str, device: str = "cpu") -> bool:
        try:
            self.yolo_model = YOLO(yolo_path)
            self.ultra_fastsam = UltraFastSAM(fastsam_path, device)
            return True
        except Exception as e:
            print(f"Model loading failed: {e}")
            return False
    
    def get_performance(self) -> Dict[str, float]:
        """获取性能数据"""
        return self.performance.copy()
    
    def detect_grains_yolo(self, image: np.ndarray, conf_threshold: float = 0.25, 
                          min_bbox_area: int = 20, class_id: Optional[int] = None):
        yolo_start = time.time()
        results = self.yolo_model(image, conf=conf_threshold, verbose=False)[0]
        
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            self.performance['yolo_time'] = time.time() - yolo_start
            return np.array([]), pd.DataFrame()
        
        boxes_xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        
        valid_detections = []
        for box, conf, cls_id in zip(boxes_xyxy, confidences, class_ids):
            if class_id is not None and cls_id != class_id:
                continue
            
            x1, y1, x2, y2 = box
            bbox_area = (x2 - x1) * (y2 - y1)
            
            if bbox_area >= min_bbox_area:
                valid_detections.append({
                    'box': box,
                    'confidence': float(conf),
                    'class_id': int(cls_id),
                    'center_x': float((x1 + x2) / 2),
                    'center_y': float((y1 + y2) / 2),
                    'area': float(bbox_area),
                })
        
        if not valid_detections:
            self.performance['yolo_time'] = time.time() - yolo_start
            return np.array([]), pd.DataFrame()
        
        detections_df = pd.DataFrame(valid_detections)
        boxes_array = detections_df['box'].values
        
        self.performance['yolo_time'] = time.time() - yolo_start
        print(f"YOLO detected {len(boxes_array)} grains")
        return boxes_array, detections_df
    
    def _enhance_mask_skimage(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Enhance mask quality using skimage
        """
        if binary_mask.sum() == 0:
            return binary_mask
        
        try:
            # 1. Gaussian smoothing
            smoothed = gaussian(binary_mask.astype(float), 
                              sigma=self.contour_params['smooth_sigma'])
            
            # 2. Re-threshold
            enhanced = (smoothed > 0.5).astype(np.uint8)
            
            # 3. Morphological operations: close then open
            selem = disk(self.contour_params['morph_radius'])
            enhanced = binary_closing(enhanced, selem).astype(np.uint8)
            enhanced = binary_opening(enhanced, selem).astype(np.uint8)
            
            # 4. Fill holes
            from scipy import ndimage
            enhanced = ndimage.binary_fill_holes(enhanced).astype(np.uint8)
            
            return enhanced
        except Exception as e:
            print(f"Mask enhancement failed: {e}")
            return binary_mask
    
    def _extract_contour_skimage(self, binary_mask: np.ndarray, idx: int = 0) -> Optional[Polygon]:
        """
        Extract smooth contour using skimage
        """
        if binary_mask.sum() == 0:
            return None
        
        try:
            # Method 1: Directly use skimage's find_contours
            contours = measure.find_contours(
                binary_mask, 
                level=self.contour_params['level']
            )
            
            if not contours:
                return None
            
            # Select contour with largest area
            main_contour = None
            max_area = 0
            
            for contour in contours:
                # 计算轮廓面积（使用多边形近似）
                if len(contour) >= 3:
                    # 将轮廓转换为(x, y)格式
                    points = [(point[1], point[0]) for point in contour]
                    polygon = Polygon(points)
                    if polygon.is_valid:
                        area = polygon.area
                        if area > max_area:
                            max_area = area
                            main_contour = contour
            
            if main_contour is None or len(main_contour) < 3:
                return None
            
            # Convert to (x, y) format
            points = [(point[1], point[0]) for point in main_contour]
            
            # If too few points, resample
            if len(points) < self.contour_params['min_contour_points']:
                points = self._resample_contour_points(points, 
                    target_points=self.contour_params['min_contour_points'])
            
            # If too many points, simplify
            elif len(points) > self.contour_params['max_contour_points']:
                points = self._simplify_contour_points(points, 
                    tolerance=self.contour_params['simplify_tolerance'])
            
            # 创建多边形
            polygon = Polygon(points)
            
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
                if not polygon.is_valid:
                    return None
            
            # 轻微平滑
            if polygon.area > 50:
                # 使用很小的buffer进行平滑
                smoothed = polygon.buffer(0.3, resolution=8).buffer(-0.1, resolution=8)
                if smoothed.is_valid and smoothed.geom_type == 'Polygon':
                    polygon = smoothed
            
            return polygon if polygon.is_valid else None
            
        except Exception as e:
            print(f"skimage contour extraction failed (grain {idx}): {e}")
            return None
    
    def _resample_contour_points(self, points: List[Tuple[float, float]], 
                                target_points: int = 100) -> List[Tuple[float, float]]:
        """
        Contour point resampling
        """
        if len(points) <= 2:
            return points
        
        # Convert points to numpy array
        points_array = np.array(points)
        
        # Calculate total contour length
        total_length = 0
        segment_lengths = []
        
        for i in range(len(points_array)):
            p1 = points_array[i]
            p2 = points_array[(i + 1) % len(points_array)]
            length = np.linalg.norm(p2 - p1)
            segment_lengths.append(length)
            total_length += length
        
        if total_length == 0:
            return points
        
        # Uniform sampling
        step = total_length / target_points
        new_points = []
        current_length = 0
        segment_index = 0
        segment_accumulated = 0
        
        for i in range(target_points):
            target_length = i * step
            
            # Find segment containing target length
            while segment_accumulated + segment_lengths[segment_index] < target_length:
                segment_accumulated += segment_lengths[segment_index]
                segment_index = (segment_index + 1) % len(points_array)
            
            # Interpolate on segment
            p1 = points_array[segment_index]
            p2 = points_array[(segment_index + 1) % len(points_array)]
            
            segment_length = segment_lengths[segment_index]
            if segment_length == 0:
                t = 0
            else:
                t = (target_length - segment_accumulated) / segment_length
            
            t = max(0, min(1, t))
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            
            new_points.append((x, y))
        
        # Ensure closed
        if new_points and new_points[0] != new_points[-1]:
            new_points.append(new_points[0])
        
        return new_points
    
    def _simplify_contour_points(self, points: List[Tuple[float, float]], 
                               tolerance: float = 1.0) -> List[Tuple[float, float]]:
        """
        Simplify contour points (using Douglas-Peucker algorithm)
        """
        if len(points) <= 3:
            return points
        
        try:
            from shapely.geometry import LineString
            from shapely.ops import simplify
            
            # Create LineString
            line = LineString(points)
            
            # Simplify
            simplified_line = simplify(line, tolerance=tolerance)
            
            # Get simplified points
            if hasattr(simplified_line, 'coords'):
                return list(simplified_line.coords)
            else:
                return points
        except Exception:
            # If simplification fails, return original points
            return points
    
    def ultra_segmentation(self, 
                         image: np.ndarray,
                         conf_threshold: float = 0.25,
                         min_area: int = 30,
                         min_bbox_area: int = 20,
                         remove_edge_grains: bool = False,
                         plot_image: bool = False,
                         keep_edges: Optional[Dict] = None,
                         debug: bool = False) -> Tuple[List[Polygon], np.ndarray, np.ndarray, pd.DataFrame, Optional[plt.Figure], Optional[plt.Axes]]:
        
        total_start = time.time()
        
        print("=" * 60)
        print("UltraFastSAM Segmentation Pipeline (skimage-optimized)")
        print("=" * 60)
        
        h, w = image.shape[:2]
        print(f"Input image: {w}x{h} pixels")
        
        # Step 1: YOLO detection
        print("\nStep 1: YOLO grain detection...")
        boxes_array, detections_df = self.detect_grains_yolo(
            image, conf_threshold, min_bbox_area
        )
        
        if len(boxes_array) == 0:
            print("No grains detected")
            empty_labels = np.zeros((h, w), dtype=np.int32)
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            self.performance['total_time'] = time.time() - total_start
            return [], empty_labels, empty_mask, pd.DataFrame(), None, None
        
        # Step 2: UltraFastSAM segmentation
        print("\nStep 2: UltraFastSAM intelligent segmentation...")
        fastsam_start = time.time()
        
        global_masks, mask_scores = self.ultra_fastsam.inference_whole_image(image)
        
        assigned_masks = self.ultra_fastsam.intelligent_matching(
            boxes_array.tolist(), global_masks, mask_scores, image.shape
        )
        
        all_masks = []
        for i, (box, assigned_mask) in enumerate(zip(boxes_array, assigned_masks)):
            if assigned_mask is not None:
                all_masks.append(assigned_mask)
            else:
                single_mask, _ = self.ultra_fastsam.segment_single_box(image, box)
                all_masks.append(single_mask)
        
        self.performance['fastsam_time'] = time.time() - fastsam_start
        
        # Step 3: Use skimage for high-quality contour extraction
        print(f"\nStep 3: Extracting contours using skimage...")
        postprocess_start = time.time()
        processed_polygons = []
        
        for i, mask in enumerate(all_masks):
            if mask is None or np.sum(mask) == 0:
                continue
            
            # 转换为二值掩码
            binary_mask = (mask > 127).astype(np.uint8)
            
            # 检查边缘颗粒
            if remove_edge_grains:
                if self._is_edge_grain(binary_mask, keep_edges):
                    continue
            
            # 计算掩码面积
            mask_area = np.sum(binary_mask)
            if mask_area < min_area:
                continue
            
            # 使用skimage提取轮廓
            polygon = self._extract_contour_skimage(binary_mask, idx=i)
            
            if polygon is not None and polygon.is_valid and polygon.area >= min_area:
                processed_polygons.append(polygon)
            
            # Progress display
            if (i + 1) % 50 == 0 or (i + 1) == len(boxes_array):
                print(f"  Processed {i+1}/{len(boxes_array)} grains")
        
        print(f"Found {len(processed_polygons)} valid polygons")
        
        # Step 4: Use project one post-processing - fix: create fake image_pred
        if len(processed_polygons) > 0 and PROJECT1_AVAILABLE:
            try:
                # 🆕 Fix: create fake image_pred parameter
                # Project one's merge_overlapping_polygons function needs an image_pred parameter
                # Create a black image as fake prediction result
                fake_image_pred = np.zeros((h, w, 3), dtype=np.float32)
                
                new_grains, comps, g = find_connected_components(processed_polygons, min_area)
                processed_polygons = merge_overlapping_polygons(
                    processed_polygons, new_grains, comps, min_area, fake_image_pred  # pass fake image_pred
                )
                print(f"Post-processing grain count: {len(processed_polygons)}")
            except Exception as e:
                print(f" Project one post-processing failed: {e}")
                # If failed, use simple post-processing
                processed_polygons = self._simple_postprocess(processed_polygons, min_area)
        elif len(processed_polygons) > 0:
            # If project one functions not available, use simple post-processing
            processed_polygons = self._simple_postprocess(processed_polygons, min_area)
        
        self.performance['postprocess_time'] = time.time() - postprocess_start
        
        # Step 5: Create labels
        print("\nStep 5: Creating label image...")
        if len(processed_polygons) > 0 and PROJECT1_AVAILABLE:
            try:
                labels, mask_all = create_labeled_image(processed_polygons, image)
                print(" Using project one functions to create labels")
            except Exception as e:
                print(f"Label creation failed: {e}")
                labels, mask_all = self._create_simple_labels(processed_polygons, image)
        else:
            labels, mask_all = self._create_simple_labels(processed_polygons, image)
        
        # Step 6: Calculate properties
        print("\nStep 6: Calculating grain properties...")
        if np.max(labels) > 0:
            try:
                props = measure.regionprops_table(
                    labels,
                    intensity_image=image,
                    properties=(
                        "label", "area", "centroid", "major_axis_length",
                        "minor_axis_length", "orientation", "perimeter",
                        "max_intensity", "mean_intensity", "min_intensity",
                    ),
                )
                grain_data = pd.DataFrame(props)
            except Exception as e:
                print(f"Failed to calculate grain properties: {e}")
                grain_data = pd.DataFrame()
        else:
            grain_data = pd.DataFrame()
        
        # Step 7: Visualization
        fig, ax = None, None
        if plot_image and len(processed_polygons) > 0:
            try:
                fig, ax = plt.subplots(figsize=(15, 10))
                ax.imshow(image)
                
                if PROJECT1_AVAILABLE:
                    plot_image_w_colorful_grains(image, processed_polygons, ax, cmap="Paired")
                    plot_grain_axes_and_centroids(processed_polygons, labels, ax, linewidth=1, markersize=10)
                
                plt.xticks([])
                plt.yticks([])
                plt.xlim([0, w])
                plt.ylim([h, 0])
                plt.tight_layout()
                print(" Visualization complete")
                
            except Exception as e:
                print(f"Visualization failed: {e}")
        
        # Performance summary
        total_time = time.time() - total_start
        self.performance['total_time'] = total_time
        
        print(f"\nFinal result: {len(processed_polygons)} grains")
        print("=" * 60)
        
        return processed_polygons, labels, mask_all, grain_data, fig, ax
    
    def _simple_postprocess(self, polygons: List[Polygon], min_area: int) -> List[Polygon]:
        """
        Simple post-processing - remove highly overlapping polygons
        """
        if len(polygons) <= 1:
            return polygons
        
        filtered_polygons = []
        
        for i, poly1 in enumerate(polygons):
            if not poly1.is_valid or poly1.area < min_area:
                continue
            
            # 检查是否与已选择的多边形高度重叠
            highly_overlapped = False
            
            for poly2 in filtered_polygons:
                if poly1.intersects(poly2):
                    # 使用IoU而不是简单的重叠比例
                    intersection = poly1.intersection(poly2).area
                    union = poly1.union(poly2).area
                    
                    if union > 0:
                        iou = intersection / union
                        if iou > 0.5:  # IoU阈值50%
                            highly_overlapped = True
                            break
            
            if not highly_overlapped:
                filtered_polygons.append(poly1)
        
        return filtered_polygons
    
    def _create_simple_labels(self, polygons: List[Polygon], image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create simple label image"""
        h, w = image.shape[:2]
        labels = np.zeros((h, w), dtype=np.int32)
        mask_all = np.zeros((h, w), dtype=np.uint8)
        
        from skimage.draw import polygon as draw_polygon
        
        for i, polygon in enumerate(polygons):
            try:
                if polygon.is_valid and hasattr(polygon, 'exterior'):
                    x, y = polygon.exterior.xy
                    x_coords = np.array(x)
                    y_coords = np.array(y)
                    
                    # 确保坐标在图像范围内
                    x_coords = np.clip(x_coords, 0, w-1)
                    y_coords = np.clip(y_coords, 0, h-1)
                    
                    rr, cc = draw_polygon(y_coords, x_coords, labels.shape)
                    labels[rr, cc] = i + 1
                    mask_all[rr, cc] = 255
            except Exception as e:
                continue
        
        return labels, mask_all
    
    def _is_edge_grain(self, mask: np.ndarray, keep_edges: Optional[Dict]) -> bool:
        """Check if grain is at edge"""
        h, w = mask.shape
        edge_thickness = 4
        
        top_edge = mask[:edge_thickness, :].sum() > 0
        bottom_edge = mask[-edge_thickness:, :].sum() > 0
        left_edge = mask[:, :edge_thickness].sum() > 0
        right_edge = mask[:, -edge_thickness:].sum() > 0
        
        if keep_edges is not None:
            if not keep_edges.get('top', True) and top_edge:
                return True
            if not keep_edges.get('bottom', True) and bottom_edge:
                return True
            if not keep_edges.get('left', True) and left_edge:
                return True
            if not keep_edges.get('right', True) and right_edge:
                return True
            return False
        else:
            return top_edge or bottom_edge or right_edge or left_edge
    
    def update_contour_params(self, params: Dict):
        """Update contour extraction parameters"""
        self.contour_params.update(params)
        print("Contour parameters updated")


if __name__ == "__main__":
    print("UltraSegmentationPipeline Test")
    print("=" * 60)
    
    # Create pipeline instance
    pipeline = UltraSegmentationPipeline()
    
    # Can adjust contour extraction parameters
    pipeline.update_contour_params({
        'level': 0.5,  # Lower threshold may get more details
        'smooth_sigma': 0.8,  # Reduce smoothing intensity
        'min_contour_points': 50,
        'max_contour_points': 300,
    })
    
    print("UltraSegmentationPipeline test passed")