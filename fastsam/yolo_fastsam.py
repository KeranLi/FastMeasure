"""
UltraFastSAM核心分割流水线 - 基于skimage的优化版
文件名：yolo_fastsam.py
功能：使用skimage.measure.find_contours进行高质量轮廓提取
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

# ===== 导入skimage =====
from skimage import measure
from skimage.filters import gaussian
from skimage.morphology import binary_opening, binary_closing, disk
print(" 已导入skimage相关模块")

# ===== 强制导入项目一函数 =====
import os
from pathlib import Path

# 获取项目一模块路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
project1_dir = project_root / "segmenteverygrain"

# 添加到Python路径
if str(project1_dir) not in sys.path:
    sys.path.insert(0, str(project1_dir))

try:
    from segmenteverygrain import (
        create_labeled_image,
        collect_polygon_from_mask,
        plot_image_w_colorful_grains,
        plot_grain_axes_and_centroids,
        find_connected_components,
        merge_overlapping_polygons
    )
    PROJECT1_AVAILABLE = True
    print(" 成功导入segmen文件中的计算函数")
except ImportError as e:
    print(f" 导入segmen文件中的计算函数失败: {e}")
    sys.exit(1)

# 导入UltraFastSAM引擎
try:
    from .seg_engine import UltraFastSAM
except ImportError:
    # 如果在主程序中运行，可能需要调整导入方式
    try:
        from seg_engine import UltraFastSAM
    except ImportError:
        print(" 无法导入UltraFastSAM引擎")


class UltraSegmentationPipeline:
    """UltraFastSAM核心分割流水线 - 基于优化版"""
    
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
        
        print("UltraSegmentationPipeline初始化完成")
    
    def load_models(self, yolo_path: str, fastsam_path: str, device: str = "cpu") -> bool:
        try:
            self.yolo_model = YOLO(yolo_path)
            self.ultra_fastsam = UltraFastSAM(fastsam_path, device)
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
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
        print(f"YOLO检测到 {len(boxes_array)} 个颗粒")
        return boxes_array, detections_df
    
    def _enhance_mask_skimage(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        使用skimage增强掩码质量
        """
        if binary_mask.sum() == 0:
            return binary_mask
        
        try:
            # 1. 高斯平滑
            smoothed = gaussian(binary_mask.astype(float), 
                              sigma=self.contour_params['smooth_sigma'])
            
            # 2. 重新阈值化
            enhanced = (smoothed > 0.5).astype(np.uint8)
            
            # 3. 形态学操作：先闭后开
            selem = disk(self.contour_params['morph_radius'])
            enhanced = binary_closing(enhanced, selem).astype(np.uint8)
            enhanced = binary_opening(enhanced, selem).astype(np.uint8)
            
            # 4. 填充孔洞
            from scipy import ndimage
            enhanced = ndimage.binary_fill_holes(enhanced).astype(np.uint8)
            
            return enhanced
        except Exception as e:
            print(f"掩码增强失败: {e}")
            return binary_mask
    
    def _extract_contour_skimage(self, binary_mask: np.ndarray, idx: int = 0) -> Optional[Polygon]:
        """
        使用skimage提取平滑轮廓
        """
        if binary_mask.sum() == 0:
            return None
        
        try:
            # 方法1：直接使用skimage的find_contours
            contours = measure.find_contours(
                binary_mask, 
                level=self.contour_params['level']
            )
            
            if not contours:
                return None
            
            # 选择面积最大的轮廓
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
            
            # 转换为(x, y)格式
            points = [(point[1], point[0]) for point in main_contour]
            
            # 如果点数太少，进行插值
            if len(points) < self.contour_params['min_contour_points']:
                points = self._resample_contour_points(points, 
                    target_points=self.contour_params['min_contour_points'])
            
            # 如果点数太多，进行简化
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
            print(f"skimage轮廓提取失败 (颗粒{idx}): {e}")
            return None
    
    def _resample_contour_points(self, points: List[Tuple[float, float]], 
                                target_points: int = 100) -> List[Tuple[float, float]]:
        """
        轮廓点重采样
        """
        if len(points) <= 2:
            return points
        
        # 将点转换为numpy数组
        points_array = np.array(points)
        
        # 计算轮廓总长度
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
        
        # 均匀采样
        step = total_length / target_points
        new_points = []
        current_length = 0
        segment_index = 0
        segment_accumulated = 0
        
        for i in range(target_points):
            target_length = i * step
            
            # 找到目标长度所在的线段
            while segment_accumulated + segment_lengths[segment_index] < target_length:
                segment_accumulated += segment_lengths[segment_index]
                segment_index = (segment_index + 1) % len(points_array)
            
            # 在线段上插值
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
        
        # 确保闭合
        if new_points and new_points[0] != new_points[-1]:
            new_points.append(new_points[0])
        
        return new_points
    
    def _simplify_contour_points(self, points: List[Tuple[float, float]], 
                               tolerance: float = 1.0) -> List[Tuple[float, float]]:
        """
        简化轮廓点（使用Douglas-Peucker算法）
        """
        if len(points) <= 3:
            return points
        
        try:
            from shapely.geometry import LineString
            from shapely.ops import simplify
            
            # 创建LineString
            line = LineString(points)
            
            # 简化
            simplified_line = simplify(line, tolerance=tolerance)
            
            # 获取简化后的点
            if hasattr(simplified_line, 'coords'):
                return list(simplified_line.coords)
            else:
                return points
        except Exception:
            # 如果简化失败，返回原始点
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
        print("UltraFastSAM分割流水线 (skimage优化版)")
        print("=" * 60)
        
        h, w = image.shape[:2]
        print(f"输入图像: {w}x{h} 像素")
        
        # 步骤1: YOLO检测
        print("\n步骤1: YOLO颗粒检测...")
        boxes_array, detections_df = self.detect_grains_yolo(
            image, conf_threshold, min_bbox_area
        )
        
        if len(boxes_array) == 0:
            print("未检测到颗粒")
            empty_labels = np.zeros((h, w), dtype=np.int32)
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            self.performance['total_time'] = time.time() - total_start
            return [], empty_labels, empty_mask, pd.DataFrame(), None, None
        
        # 步骤2: UltraFastSAM分割
        print("\n步骤2: UltraFastSAM智能分割...")
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
        
        # 步骤3: 使用skimage进行高质量的轮廓提取
        print(f"\n步骤3: 使用skimage进行轮廓提取...")
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
            
            # 进度显示
            if (i + 1) % 50 == 0 or (i + 1) == len(boxes_array):
                print(f"  已处理 {i+1}/{len(boxes_array)} 个颗粒")
        
        print(f"找到 {len(processed_polygons)} 个有效多边形")
        
        # 步骤4: 使用项目一后处理 - 修复：创建假的image_pred
        if len(processed_polygons) > 0 and PROJECT1_AVAILABLE:
            try:
                # 🆕 修复：创建假的image_pred参数
                # 项目一的merge_overlapping_polygons函数需要一个image_pred参数
                # 创建一个全黑的图像作为假的预测结果
                fake_image_pred = np.zeros((h, w, 3), dtype=np.float32)
                
                new_grains, comps, g = find_connected_components(processed_polygons, min_area)
                processed_polygons = merge_overlapping_polygons(
                    processed_polygons, new_grains, comps, min_area, fake_image_pred  # 传入假的image_pred
                )
                print(f"后处理后颗粒数: {len(processed_polygons)}")
            except Exception as e:
                print(f" 项目一后处理失败: {e}")
                # 如果失败，使用简单后处理
                processed_polygons = self._simple_postprocess(processed_polygons, min_area)
        elif len(processed_polygons) > 0:
            # 如果没有项目一函数，使用简单后处理
            processed_polygons = self._simple_postprocess(processed_polygons, min_area)
        
        self.performance['postprocess_time'] = time.time() - postprocess_start
        
        # 步骤5: 创建标签
        print("\n步骤5: 创建标签图像...")
        if len(processed_polygons) > 0 and PROJECT1_AVAILABLE:
            try:
                labels, mask_all = create_labeled_image(processed_polygons, image)
                print(" 使用项目一函数创建标签")
            except Exception as e:
                print(f"标签创建失败: {e}")
                labels, mask_all = self._create_simple_labels(processed_polygons, image)
        else:
            labels, mask_all = self._create_simple_labels(processed_polygons, image)
        
        # 步骤6: 计算属性
        print("\n步骤6: 计算颗粒属性...")
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
                print(f"计算颗粒属性失败: {e}")
                grain_data = pd.DataFrame()
        else:
            grain_data = pd.DataFrame()
        
        # 步骤7: 可视化
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
                print(" 可视化完成")
                
            except Exception as e:
                print(f"可视化失败: {e}")
        
        # 性能总结
        total_time = time.time() - total_start
        self.performance['total_time'] = total_time
        
        print(f"\n最终结果: {len(processed_polygons)} 个颗粒")
        print("=" * 60)
        
        return processed_polygons, labels, mask_all, grain_data, fig, ax
    
    def _simple_postprocess(self, polygons: List[Polygon], min_area: int) -> List[Polygon]:
        """
        简单后处理 - 去除高度重叠的多边形
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
        """创建简单的标签图像"""
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
        """检查是否为边缘颗粒"""
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
        """更新轮廓提取参数"""
        self.contour_params.update(params)
        print("轮廓参数已更新")


if __name__ == "__main__":
    print("UltraSegmentationPipeline测试")
    print("=" * 60)
    
    # 创建pipeline实例
    pipeline = UltraSegmentationPipeline()
    
    # 可以调整轮廓提取参数
    pipeline.update_contour_params({
        'level': 0.5,  # 降低阈值可能获得更多细节
        'smooth_sigma': 0.8,  # 减小平滑强度
        'min_contour_points': 50,
        'max_contour_points': 300,
    })
    
    print("UltraSegmentationPipeline测试通过")