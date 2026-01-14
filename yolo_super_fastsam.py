"""
SuperFastSAM主分割流水线
文件名：yolo_super_fastsam.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import Polygon
import warnings
from typing import List, Tuple, Optional, Dict
import sys
import time
import cv2
import torch

from ultralytics import YOLO
from fastsam_optimized import SuperFastSAM

# 导入项目一的关键函数
try:
    from segmenteverygrain import (
        create_labeled_image,
        collect_polygon_from_mask,
        plot_image_w_colorful_grains,
        plot_grain_axes_and_centroids,
        find_connected_components,
        merge_overlapping_polygons
    )
    SEGMENTEVERYGRAIN_AVAILABLE = True
    print("✅ 成功导入项目一关键函数")
except ImportError as e:
    print(f"⚠️ 导入项目一函数失败: {e}")
    SEGMENTEVERYGRAIN_AVAILABLE = False

from skimage import measure


def detect_grains_yolo_super(
    image: np.ndarray,
    yolo_model: YOLO,
    conf_threshold: float = 0.25,
    min_bbox_area: int = 20,
    class_id: Optional[int] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    生产级YOLO检测（与项目一完全相同）
    """
    # 运行YOLO推理
    results = yolo_model(
        image,
        conf=conf_threshold,
        verbose=False,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )[0]
    
    # 提取检测框
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
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
                'confidence': conf,
                'class_id': cls_id,
                'center_x': (x1 + x2) / 2,
                'center_y': (y1 + y2) / 2,
                'area': bbox_area,
                'width': x2 - x1,
                'height': y2 - y1
            })
    
    if not valid_detections:
        return np.array([]), pd.DataFrame()
    
    # 创建DataFrame并按置信度排序
    detections_df = pd.DataFrame(valid_detections)
    detections_df = detections_df.sort_values('confidence', ascending=False)
    
    # 提取边界框数组
    boxes_array = detections_df['box'].values
    print(f"🎯 YOLO检测到 {len(boxes_array)} 个颗粒，置信度 > {conf_threshold}")
    
    return boxes_array, detections_df


def yolo_super_fastsam_segmentation(
    image: np.ndarray,
    yolo_model: YOLO,
    super_fastsam: SuperFastSAM,
    conf_threshold: float = 0.25,
    min_area: int = 30,
    min_bbox_area: int = 20,
    remove_edge_grains: bool = False,
    class_id: Optional[int] = None,
    plot_image: bool = False,
    keep_edges: Optional[Dict] = None
) -> Tuple[List[Polygon], np.ndarray, np.ndarray, pd.DataFrame, Optional[plt.Figure], Optional[plt.Axes]]:
    """
    SuperFastSAM主分割函数（与项目一接口完全相同）
    """
    total_start = time.time()
    
    print("=" * 60)
    print("🚀 SuperFastSAM岩石颗粒分割流水线")
    print("=" * 60)
    
    # 步骤1: YOLO检测
    print("📦 步骤1: YOLO颗粒检测...")
    yolo_start = time.time()
    boxes_array, detections_df = detect_grains_yolo_super(
        image, yolo_model, conf_threshold, min_bbox_area, class_id
    )
    yolo_time = time.time() - yolo_start
    
    if len(boxes_array) == 0:
        print("❌ YOLO未检测到任何颗粒")
        return [], np.zeros_like(image[:,:,0]), np.zeros_like(image), pd.DataFrame(), None, None
    
    print(f"✅ YOLO检测完成: {len(boxes_array)}个颗粒, 耗时: {yolo_time:.2f}秒")
    
    # 步骤2: SuperFastSAM分割
    print("\n🎯 步骤2: SuperFastSAM智能分割...")
    fastsam_start = time.time()
    
    # 使用SuperFastSAM进行智能分割
    masks, fastsam_time = super_fastsam.segment_image(image, boxes_array.tolist())
    
    # 步骤3: 后处理（与项目一完全相同）
    print("\n🔄 步骤3: 后处理（项目一标准）...")
    post_start = time.time()
    
    all_grains = []
    valid_masks = []
    
    for i, mask in enumerate(masks):
        if mask is None or mask.sum() == 0:
            continue
        
        # 确保掩码是二值图像
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8) * 255
        
        # 检查边缘颗粒（与项目一逻辑相同）
        if remove_edge_grains:
            h, w = mask.shape
            edge_thickness = 4
            
            # 检查是否接触边缘
            top_edge = mask[:edge_thickness, :].sum() > 0
            bottom_edge = mask[-edge_thickness:, :].sum() > 0
            left_edge = mask[:, :edge_thickness].sum() > 0
            right_edge = mask[:, -edge_thickness:].sum() > 0
            
            # 根据keep_edges设置决定是否保留
            keep = True
            if keep_edges is not None:
                if not keep_edges.get('top', True) and top_edge:
                    keep = False
                if not keep_edges.get('bottom', True) and bottom_edge:
                    keep = False
                if not keep_edges.get('left', True) and left_edge:
                    keep = False
                if not keep_edges.get('right', True) and right_edge:
                    keep = False
            elif top_edge or bottom_edge or left_edge or right_edge:
                keep = False
            
            if not keep:
                continue
        
        # 计算掩码面积
        mask_area = mask.sum() / 255
        if mask_area < min_area:
            continue
        
        # 获取轮廓（与项目一相同）
        contours = measure.find_contours(mask, 0.5)
        if len(contours) == 0:
            continue
        
        # 取面积最大的轮廓
        contour_areas = [len(c) for c in contours]
        main_contour = contours[np.argmax(contour_areas)]
        
        # 将轮廓转换为多边形（与项目一相同）
        try:
            if len(main_contour) >= 3:
                # 注意：contour的坐标是 (row, column)，需要转换为 (x, y)
                polygon = Polygon([(point[1], point[0]) for point in main_contour])
                if polygon.is_valid and polygon.area >= min_area:
                    all_grains.append(polygon)
                    valid_masks.append(mask)
        except Exception as e:
            print(f"⚠️ 轮廓{i}转换为多边形失败: {e}")
            continue
    
    post_time = time.time() - post_start
    
    # 步骤4: 后处理去重和合并（使用项目一函数）
    print("\n🔗 步骤4: 去重和合并处理...")
    if len(all_grains) > 0 and SEGMENTEVERYGRAIN_AVAILABLE:
        try:
            print("  使用项目一find_connected_components和merge_overlapping_polygons...")
            new_grains, comps, g = find_connected_components(all_grains, min_area)
            all_grains = merge_overlapping_polygons(all_grains, new_grains, comps, min_area, None)
        except Exception as e:
            print(f"⚠️ 项目一后处理失败: {e}")
            # 简单去重：基于IoU去除高度重叠的多边形
            all_grains = _simple_overlap_removal(all_grains, min_area)
    
    print(f"✅ 后处理完成: 有效颗粒 {len(all_grains)}个, 耗时: {post_time:.2f}秒")
    
    # 步骤5: 创建标签图像
    print("\n🏷️  步骤5: 创建标签图像...")
    label_start = time.time()
    
    if len(all_grains) > 0 and SEGMENTEVERYGRAIN_AVAILABLE:
        try:
            labels, mask_all = create_labeled_image(all_grains, image)
        except Exception as e:
            print(f"⚠️ 使用项目一create_labeled_image失败: {e}")
            labels, mask_all = _create_simple_labeled_image(all_grains, image)
    else:
        labels, mask_all = _create_simple_labeled_image(all_grains, image)
    
    label_time = time.time() - label_start
    
    # 步骤6: 计算颗粒属性
    print("\n📊 步骤6: 计算颗粒属性...")
    prop_start = time.time()
    
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
    
    prop_time = time.time() - prop_start
    
    # 步骤7: 可视化（与项目一完全相同）
    fig, ax = None, None
    if plot_image and len(all_grains) > 0:
        print("\n🎨 步骤7: 生成可视化结果（项目一风格）...")
        try:
            fig, ax = plt.subplots(figsize=(15, 10))
            
            # 使用项目一完全相同的可视化函数
            ax.imshow(image)
            
            if SEGMENTEVERYGRAIN_AVAILABLE:
                plot_image_w_colorful_grains(image, all_grains, ax, cmap="Paired")
                plot_grain_axes_and_centroids(all_grains, labels, ax, linewidth=1, markersize=10)
            else:
                # 备用可视化
                _plot_simple_grains(image, all_grains, labels, ax)
            
            # 与项目一完全相同的图形设置
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0, image.shape[1]])
            plt.ylim([image.shape[0], 0])
            plt.tight_layout()
            
        except Exception as e:
            print(f"  ⚠️ 可视化生成失败: {e}")
    
    # 性能总结
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("📈 SuperFastSAM性能总结")
    print("=" * 60)
    print(f"总处理时间: {total_time:.2f}秒")
    print(f"YOLO检测: {yolo_time:.2f}秒 ({yolo_time/total_time*100:.1f}%)")
    print(f"SuperFastSAM分割: {fastsam_time:.2f}秒 ({fastsam_time/total_time*100:.1f}%)")
    print(f"后处理: {post_time:.2f}秒 ({post_time/total_time*100:.1f}%)")
    print(f"标签和属性: {label_time+prop_time:.2f}秒 ({(label_time+prop_time)/total_time*100:.1f}%)")
    print(f"最终颗粒数: {len(all_grains)}/{len(boxes_array)} ({len(all_grains)/len(boxes_array)*100:.1f}%)")
    print("=" * 60)
    
    return all_grains, labels, mask_all, grain_data, fig, ax


def _simple_overlap_removal(grains: List[Polygon], min_area: int) -> List[Polygon]:
    """简单重叠去除（备用）"""
    if len(grains) <= 1:
        return grains
    
    filtered_grains = []
    for i, poly1 in enumerate(grains):
        if not poly1.is_valid or poly1.area < min_area:
            continue
        
        # 检查是否与已选择的颗粒高度重叠
        highly_overlapped = False
        for poly2 in filtered_grains:
            if poly1.intersects(poly2):
                intersection = poly1.intersection(poly2).area
                if intersection / min(poly1.area, poly2.area) > 0.7:  # 70%重叠
                    highly_overlapped = True
                    break
        
        if not highly_overlapped:
            filtered_grains.append(poly1)
    
    return filtered_grains


def _create_simple_labeled_image(grains: List[Polygon], image: np.ndarray):
    """创建简单标签图像（备用）"""
    labels = np.zeros(image.shape[:2], dtype=np.int32)
    mask_all = np.zeros_like(image[:,:,0])
    
    for i, grain in enumerate(grains):
        try:
            if hasattr(grain, 'exterior'):
                # 创建多边形掩码
                from skimage.draw import polygon
                x, y = grain.exterior.xy
                rr, cc = polygon(np.array(y), np.array(x), labels.shape)
                
                # 确保坐标在图像范围内
                rr = np.clip(rr, 0, labels.shape[0]-1)
                cc = np.clip(cc, 0, labels.shape[1]-1)
                
                labels[rr, cc] = i + 1
                mask_all[rr, cc] = 255
        except Exception as e:
            continue
    
    return labels, mask_all


def _plot_simple_grains(image, grains, labels, ax):
    """简单可视化（备用）"""
    import matplotlib.patches as patches
    from skimage import measure
    
    # 显示原始图像
    ax.imshow(image, alpha=0.7)
    
    # 绘制颗粒轮廓
    for i, grain in enumerate(grains):
        try:
            if hasattr(grain, 'exterior'):
                x, y = grain.exterior.xy
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


if __name__ == "__main__":
    print("=" * 60)
    print("SuperFastSAM主分割流水线")
    print("=" * 60)
    print("使用方法:")
    print("1. 导入: from yolo_super_fastsam import yolo_super_fastsam_segmentation")
    print("2. 创建SuperFastSAM引擎: from fastsam_optimized import SuperFastSAM")
    print("3. 调用: result = yolo_super_fastsam_segmentation(image, yolo_model, super_fastsam)")
    print("=" * 60)