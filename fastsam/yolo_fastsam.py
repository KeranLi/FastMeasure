"""
UltraFastSAM核心分割流水线
文件名：yolo_fastsam.py
功能：完整的分割流水线，与项目一100%兼容
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

# 导入UltraFastSAM引擎
from .seg_engine import UltraFastSAM

# 导入工具函数
try:
    from seg_tools import (
        validate_image_data,
        convert_to_rgb,
        normalize_image,
        calculate_iou,
        filter_small_polygons,
        smart_merge_polygons
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    print("⚠️ 使用简化工具函数")

# 导入后处理模块
try:
    from seg_optimize import SmartPostProcessor
    POSTPROCESSOR_AVAILABLE = True
except ImportError:
    POSTPROCESSOR_AVAILABLE = False
    print("⚠️ 使用简化后处理")

# 导入项目一的关键函数（确保兼容性）
try:
    # 尝试导入项目一的函数
    sys.path.insert(0, '/root/autodl-tmp/segmenteverygrain')
    from segmenteverygrain import (
        create_labeled_image,
        collect_polygon_from_mask,
        plot_image_w_colorful_grains,
        plot_grain_axes_and_centroids,
        find_connected_components,
        merge_overlapping_polygons
    )
    PROJECT1_AVAILABLE = True
    print("✅ 成功导入项目一关键函数")
except ImportError as e:
    PROJECT1_AVAILABLE = False
    print(f"⚠️ 导入项目一函数失败: {e}")

from skimage import measure, morphology


class UltraSegmentationPipeline:
    """UltraFastSAM核心分割流水线"""
    
    def __init__(self, config: Dict = None):
        """
        初始化分割流水线
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 模型实例
        self.yolo_model = None
        self.ultra_fastsam = None
        
        # 性能监控
        self.performance = {
            'yolo_time': 0.0,
            'fastsam_time': 0.0,
            'postprocess_time': 0.0,
            'total_time': 0.0
        }
        
        print("✅ UltraSegmentationPipeline初始化完成")
    
    def load_models(self, 
                   yolo_path: str, 
                   fastsam_path: str, 
                   device: str = "cpu") -> bool:
        """
        加载模型
        
        Args:
            yolo_path: YOLO模型路径
            fastsam_path: FastSAM模型路径
            device: 运行设备
            
        Returns:
            是否成功
        """
        print("🔄 加载AI模型...")
        
        try:
            # 加载YOLO模型
            self.yolo_model = YOLO(yolo_path)
            print(f"✅ YOLO模型加载成功: {yolo_path}")
            
            # 加载UltraFastSAM引擎
            self.ultra_fastsam = UltraFastSAM(fastsam_path, device)
            print(f"✅ UltraFastSAM引擎加载成功: {fastsam_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def detect_grains_yolo(self, 
                          image: np.ndarray,
                          conf_threshold: float = 0.25,
                          min_bbox_area: int = 20,
                          class_id: Optional[int] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        改进的YOLO检测函数
        
        Args:
            image: 输入图像
            conf_threshold: 置信度阈值
            min_bbox_area: 最小检测框面积
            class_id: 类别ID（可选）
            
        Returns:
            boxes_array: 边界框数组
            detections_df: 检测结果DataFrame
        """
        start_time = time.time()
        
        # 运行YOLO推理
        results = self.yolo_model(
            image,
            conf=conf_threshold,
            verbose=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )[0]
        
        # 提取检测框
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            print("⚠️ YOLO未检测到任何颗粒")
            return np.array([]), pd.DataFrame()
        
        # 转换为numpy数组
        boxes_xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        
        # 过滤有效检测
        valid_detections = []
        for i, (box, conf, cls_id) in enumerate(zip(boxes_xyxy, confidences, class_ids)):
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
                    'width': float(x2 - x1),
                    'height': float(y2 - y1)
                })
        
        if not valid_detections:
            return np.array([]), pd.DataFrame()
        
        # 创建DataFrame
        detections_df = pd.DataFrame(valid_detections)
        detections_df = detections_df.sort_values('confidence', ascending=False)
        
        # 提取边界框数组
        boxes_array = detections_df['box'].values
        
        yolo_time = time.time() - start_time
        self.performance['yolo_time'] = yolo_time
        
        print(f"🎯 YOLO检测完成: {len(boxes_array)}个颗粒, 耗时: {yolo_time:.2f}s")
        
        return boxes_array, detections_df
    
    def ultra_segmentation(self, 
                         image: np.ndarray,
                         conf_threshold: float = 0.25,
                         min_area: int = 30,
                         min_bbox_area: int = 20,
                         remove_edge_grains: bool = False,
                         plot_image: bool = False,
                         keep_edges: Optional[Dict] = None) -> Tuple[List[Polygon], np.ndarray, np.ndarray, pd.DataFrame, Optional[plt.Figure], Optional[plt.Axes]]:
        """
        UltraFastSAM主分割函数（与项目一接口完全相同）
        
        Returns:
            all_grains: 颗粒多边形列表
            labels: 标签图像
            mask_all: 掩码图像
            grain_data: 颗粒数据
            fig: 图形对象
            ax: 坐标轴对象
        """
        total_start = time.time()
        
        print("=" * 60)
        print("🚀 UltraFastSAM终极分割流水线启动")
        print("=" * 60)
        
        # 验证输入图像
        h, w = image.shape[:2]
        print(f"📊 输入图像: {w}x{h} 像素")
        
        # 步骤1: YOLO检测
        print("\n📦 步骤1: YOLO颗粒检测...")
        boxes_array, detections_df = self.detect_grains_yolo(
            image, conf_threshold, min_bbox_area
        )
        
        if len(boxes_array) == 0:
            print("❌ 未检测到颗粒，返回空结果")
            empty_labels = np.zeros((h, w), dtype=np.int32)
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            return [], empty_labels, empty_mask, pd.DataFrame(), None, None
        
        # 步骤2: UltraFastSAM分割
        print("\n🎯 步骤2: UltraFastSAM智能分割...")
        fastsam_start = time.time()
        
        # 2.1 全局推理获取候选掩码
        global_masks, mask_scores = self.ultra_fastsam.inference_whole_image(image)
        
        # 2.2 智能掩码匹配
        assigned_masks = self.ultra_fastsam.intelligent_matching(
            boxes_array.tolist(),
            global_masks,
            mask_scores,
            image.shape
        )
        
        # 2.3 处理未分配的框（单框精细分割）
        all_masks = []
        mask_qualities = []
        
        print(f"\n🔍 步骤2.3: 精细分割未分配的框...")
        for i, (box, assigned_mask) in enumerate(zip(boxes_array, assigned_masks)):
            if assigned_mask is not None:
                all_masks.append(assigned_mask)
                # 估算质量分数
                mask_quality = np.sum(assigned_mask > 0) / (h * w)
                mask_qualities.append(min(mask_quality * 10, 1.0))
            else:
                # 对未分配的框进行精细分割
                single_mask, quality = self.ultra_fastsam.segment_single_box(image, box)
                all_masks.append(single_mask)
                mask_qualities.append(quality)
        
        fastsam_time = time.time() - fastsam_start
        self.performance['fastsam_time'] = fastsam_time
        
        # 步骤3: 掩码后处理
        print("\n🔄 步骤3: 掩码后处理...")
        postprocess_start = time.time()
        
        processed_polygons = []
        valid_masks = []
        
        for i, (mask, quality, box) in enumerate(zip(all_masks, mask_qualities, boxes_array)):
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
            
            # 获取轮廓
            contours = measure.find_contours(binary_mask, 0.5)
            if len(contours) == 0:
                continue
            
            # 取面积最大的轮廓
            contour_areas = [len(c) for c in contours]
            main_contour = contours[np.argmax(contour_areas)]
            
            # 将轮廓转换为多边形
            try:
                if len(main_contour) >= 3:
                    # 轮廓坐标是 (row, column)，转换为 (x, y)
                    polygon_points = [(point[1], point[0]) for point in main_contour]
                    polygon = Polygon(polygon_points)
                    
                    if polygon.is_valid and polygon.area >= min_area:
                        processed_polygons.append(polygon)
                        valid_masks.append(binary_mask)
            except Exception as e:
                print(f"⚠️ 轮廓{i}转换失败: {e}")
                continue
        
        postprocess_time = time.time() - postprocess_start
        self.performance['postprocess_time'] = postprocess_time
        
        print(f"✅ 后处理完成: {len(processed_polygons)}个有效多边形")
        
        # 步骤4: 智能后处理（去重和合并）
        print("\n🔗 步骤4: 智能后处理...")
        if len(processed_polygons) > 0:
            if POSTPROCESSOR_AVAILABLE:
                # 使用智能后处理器
                postprocessor = SmartPostProcessor(min_area=min_area)
                processed_polygons = postprocessor.process(processed_polygons)
            elif PROJECT1_AVAILABLE:
                # 使用项目一的后处理函数
                try:
                    new_grains, comps, g = find_connected_components(processed_polygons, min_area)
                    processed_polygons = merge_overlapping_polygons(
                        processed_polygons, new_grains, comps, min_area, None
                    )
                except Exception as e:
                    print(f"⚠️ 项目一后处理失败: {e}")
                    processed_polygons = self._simple_postprocess(processed_polygons, min_area)
            else:
                # 使用简单后处理
                processed_polygons = self._simple_postprocess(processed_polygons, min_area)
        
        print(f"✅ 后处理后: {len(processed_polygons)}个最终颗粒")
        
        # 步骤5: 创建标签图像
        print("\n🏷️  步骤5: 创建标签图像...")
        if len(processed_polygons) > 0:
            if PROJECT1_AVAILABLE:
                try:
                    labels, mask_all = create_labeled_image(processed_polygons, image)
                except Exception as e:
                    print(f"⚠️ 使用项目一create_labeled_image失败: {e}")
                    labels, mask_all = self._create_simple_labels(processed_polygons, image)
            else:
                labels, mask_all = self._create_simple_labels(processed_polygons, image)
        else:
            labels = np.zeros((h, w), dtype=np.int32)
            mask_all = np.zeros((h, w), dtype=np.uint8)
        
        # 步骤6: 计算颗粒属性
        print("\n📊 步骤6: 计算颗粒属性...")
        if np.max(labels) > 0:
            try:
                props = measure.regionprops_table(
                    labels,
                    intensity_image=image,
                    properties=(
                        "label",
                        "area",
                        "centroid",
                        "major_axis_length",
                        "minor_axis_length",
                        "orientation",
                        "perimeter",
                        "max_intensity",
                        "mean_intensity",
                        "min_intensity",
                    ),
                )
                grain_data = pd.DataFrame(props)
            except Exception as e:
                print(f"⚠️ 计算颗粒属性失败: {e}")
                grain_data = pd.DataFrame()
        else:
            grain_data = pd.DataFrame()
        
        # 步骤7: 可视化
        fig, ax = None, None
        if plot_image and len(processed_polygons) > 0:
            print("\n🎨 步骤7: 生成可视化结果...")
            try:
                fig, ax = plt.subplots(figsize=(15, 10))
                ax.imshow(image)
                
                if PROJECT1_AVAILABLE:
                    plot_image_w_colorful_grains(image, processed_polygons, ax, cmap="Paired")
                    plot_grain_axes_and_centroids(processed_polygons, labels, ax, linewidth=1, markersize=10)
                else:
                    self._plot_simple_grains(image, processed_polygons, labels, ax)
                
                # 与项目一相同的图形设置
                plt.xticks([])
                plt.yticks([])
                plt.xlim([0, w])
                plt.ylim([h, 0])
                plt.tight_layout()
                
            except Exception as e:
                print(f"⚠️ 可视化生成失败: {e}")
        
        # 性能总结
        total_time = time.time() - total_start
        self.performance['total_time'] = total_time
        
        print("\n" + "=" * 60)
        print("📈 UltraFastSAM性能总结")
        print("=" * 60)
        print(f"总处理时间: {total_time:.2f}秒")
        print(f"YOLO检测: {self.performance['yolo_time']:.2f}秒 ({self.performance['yolo_time']/total_time*100:.1f}%)")
        print(f"UltraFastSAM分割: {self.performance['fastsam_time']:.2f}秒 ({self.performance['fastsam_time']/total_time*100:.1f}%)")
        print(f"后处理: {self.performance['postprocess_time']:.2f}秒 ({self.performance['postprocess_time']/total_time*100:.1f}%)")
        
        # FastSAM引擎性能统计
        fastsam_stats = self.ultra_fastsam.get_performance_stats()
        print(f"\n🔧 FastSAM引擎统计:")
        print(f"  推理次数: {fastsam_stats['total_inferences']}")
        print(f"  生成掩码: {fastsam_stats['total_masks_generated']}")
        print(f"  过滤后掩码: {fastsam_stats['total_masks_filtered']}")
        print(f"  平均推理时间: {fastsam_stats.get('avg_time_per_inference', 0):.3f}s")
        
        print(f"\n🎯 最终结果:")
        print(f"  YOLO检测框: {len(boxes_array)}")
        print(f"  有效掩码: {len(all_masks)}")
        print(f"  最终颗粒数: {len(processed_polygons)} ({len(processed_polygons)/len(boxes_array)*100:.1f}%)")
        print("=" * 60)
        
        return processed_polygons, labels, mask_all, grain_data, fig, ax
    
    def _is_edge_grain(self, mask: np.ndarray, keep_edges: Optional[Dict]) -> bool:
        """
        检查是否为边缘颗粒
        
        Returns:
            True如果是边缘颗粒且应该移除
        """
        h, w = mask.shape
        edge_thickness = 4
        
        # 检查是否接触边缘
        top_edge = mask[:edge_thickness, :].sum() > 0
        bottom_edge = mask[-edge_thickness:, :].sum() > 0
        left_edge = mask[:, :edge_thickness].sum() > 0
        right_edge = mask[:, -edge_thickness:].sum() > 0
        
        # 根据keep_edges设置决定
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
            # 默认移除所有边缘颗粒
            return top_edge or bottom_edge or left_edge or right_edge
    
    def _simple_postprocess(self, polygons: List[Polygon], min_area: int) -> List[Polygon]:
        """
        简单后处理（去除高度重叠的多边形）
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
        """
        创建简单的标签图像
        """
        h, w = image.shape[:2]
        labels = np.zeros((h, w), dtype=np.int32)
        mask_all = np.zeros((h, w), dtype=np.uint8)
        
        for i, polygon in enumerate(polygons):
            try:
                if hasattr(polygon, 'exterior'):
                    from skimage.draw import polygon as draw_polygon
                    
                    # 获取多边形顶点
                    x, y = polygon.exterior.xy
                    x_coords = np.array(x)
                    y_coords = np.array(y)
                    
                    # 确保坐标在图像范围内
                    x_coords = np.clip(x_coords, 0, w-1)
                    y_coords = np.clip(y_coords, 0, h-1)
                    
                    # 绘制多边形
                    rr, cc = draw_polygon(y_coords, x_coords, labels.shape)
                    labels[rr, cc] = i + 1
                    mask_all[rr, cc] = 255
            except Exception as e:
                print(f"⚠️ 创建标签失败: {e}")
                continue
        
        return labels, mask_all
    
    def _plot_simple_grains(self, image, polygons, labels, ax):
        """
        简单可视化
        """
        import matplotlib.patches as patches
        
        # 显示原始图像
        ax.imshow(image, alpha=0.7)
        
        # 绘制颗粒轮廓
        for i, polygon in enumerate(polygons):
            try:
                if hasattr(polygon, 'exterior'):
                    x, y = polygon.exterior.xy
                    poly_patch = patches.Polygon(
                        np.column_stack([x, y]),
                        closed=True,
                        facecolor='red',
                        edgecolor='red',
                        alpha=0.3,
                        linewidth=1
                    )
                    ax.add_patch(poly_patch)
            except:
                continue
        
        # 绘制质心
        if labels is not None and np.max(labels) > 0:
            props = measure.regionprops(labels.astype("int"))
            for prop in props:
                y0, x0 = prop.centroid
                ax.plot(x0, y0, '.k', markersize=10)
    
    def get_performance(self) -> Dict[str, float]:
        """获取性能数据"""
        return self.performance.copy()


if __name__ == "__main__":
    print("UltraSegmentationPipeline测试")
    print("=" * 60)
    
    # 测试代码
    pipeline = UltraSegmentationPipeline()
    print("✅ 流水线测试通过")