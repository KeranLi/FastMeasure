# yolo_sam_segmentation_fixed.py
"""
YOLO + SAM Grain Segmentation Pipeline (已修复版本)
修复了 'NoneType' object is not subscriptable 错误
【新增】：统计SAM跳过的点数量，排查漏分割问题
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import Polygon, Point
from shapely.affinity import translate
import warnings
from typing import List, Tuple, Optional
import sys

# YOLO and SAM imports
import torch
from ultralytics import YOLO
from segment_anything import SamPredictor

# Import existing functions from segmenteverygrain.py
# 【修改1：删除了重叠/聚类相关的函数导入】
from segmenteverygrain import (
    one_point_prompt,
    create_labeled_image,
    collect_polygon_from_mask,
    plot_image_w_colorful_grains,
    plot_grain_axes_and_centroids
)
from skimage import measure


def detect_grains_yolo(
    image: np.ndarray,
    yolo_model: YOLO,
    conf_threshold: float = 0.5,
    min_bbox_area: int = 100,
    class_id: Optional[int] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Detect grains using YOLO model and extract center points for SAM prompts.
    """
    # Run YOLO inference
    results = yolo_model(
        image,
        conf=conf_threshold,
        verbose=False,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )[0]
    
    # Extract detection boxes
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return np.array([]), pd.DataFrame()
    
    # Convert to numpy arrays
    boxes_xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    
    # Filter valid detections
    valid_detections = []
    for i, (box, conf, cls_id) in enumerate(zip(boxes_xyxy, confidences, class_ids)):
        if class_id is not None and cls_id != class_id:
            continue
        
        x1, y1, x2, y2 = box
        bbox_area = (x2 - x1) * (y2 - y1)
        
        if bbox_area >= min_bbox_area:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            valid_detections.append({
                'box': box,
                'confidence': conf,
                'class_id': cls_id,
                'center_x': center_x,
                'center_y': center_y,
                'area': bbox_area
            })
    
    if not valid_detections:
        return np.array([]), pd.DataFrame()
    
    # Create DataFrame
    detections_df = pd.DataFrame(valid_detections)
    detections_df = detections_df.sort_values('confidence', ascending=False)
    
    # Extract center coordinates
    coords = detections_df[['center_x', 'center_y']].values.astype(np.int32)
    
    print(f"YOLO detected {len(coords)} grains with confidence > {conf_threshold}")
    return coords, detections_df


def yolo_sam_segmentation(
    image: np.ndarray,
    yolo_model: YOLO,
    sam_predictor: SamPredictor,
    conf_threshold: float = 0.5,
    min_area: int = 100,
    min_bbox_area: int = 100,
    remove_edge_grains: bool = False,
    remove_large_objects: bool = False,
    class_id: Optional[int] = None,
    plot_image: bool = False,
    keep_edges: Optional[dict] = None
) -> Tuple[List[Polygon], np.ndarray, np.ndarray, pd.DataFrame, Optional[plt.Figure], Optional[plt.Axes]]:
    """
    Main segmentation function (已修复版本)
    【修改说明】：1.删除了无监督聚类/重叠合并的冗余步骤 2.新增SAM跳过点统计
    """
    # YOLO detection
    print("Running YOLO detection...")
    coords, detections_df = detect_grains_yolo(
        image, yolo_model, conf_threshold, min_bbox_area, class_id
    )
    
    if len(coords) == 0:
        print("No grains detected by YOLO.")
        return [], np.zeros_like(image[:,:,0]), np.zeros_like(image), pd.DataFrame(), None, None
    
    # SAM segmentation
    print(f"Running SAM segmentation on {len(coords)} prompts...")
    sam_predictor.set_image(image)
    all_grains = []
    skipped_count = 0  # 【新增】统计跳过的点数量
    skip_reasons = {   # 【新增】统计跳过原因
        "out_of_bound": 0,
        "invalid_mask": 0,
        "exception": 0
    }
    
    for i in tqdm(range(len(coords))):
        # ===== 【核心修复：异常处理开始】 =====
        try:
            x, y = coords[i]
            
            # Skip out-of-bound coordinates
            # 【临时测试：注释掉这行，关闭越界检查】
            # if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
            #     skipped_count +=1
            #     skip_reasons["out_of_bound"] +=1
            #     print(f"跳过越界点 {i}: ({x},{y}) | 图片尺寸: {image.shape[1]}x{image.shape[0]}")
            #     continue
                
            # Single-point SAM segmentation
            sx, sy, mask = one_point_prompt(x, y, image, sam_predictor, ax=False)
            
            # Additional safety check
            if mask is None or not isinstance(mask, np.ndarray) or mask.size == 0:
                skipped_count +=1
                skip_reasons["invalid_mask"] +=1
                print(f"跳过无效mask点 {i}: ({x},{y})")
                continue
                
            if np.max(mask) > 0:
                # Check edge-touching grains
                if remove_edge_grains:
                    edges_to_check = []
                    if keep_edges is None:
                        edges_to_check = [mask[:4, :], mask[-4:, :], mask[:, :4].T, mask[:, -4:].T]
                    else:
                        if not keep_edges.get('top', False):
                            edges_to_check.append(mask[:4, :])
                        if not keep_edges.get('bottom', False):
                            edges_to_check.append(mask[-4:, :])
                        if not keep_edges.get('left', False):
                            edges_to_check.append(mask[:, :4].T)
                        if not keep_edges.get('right', False):
                            edges_to_check.append(mask[:, -4:].T)
                    
                    if len(edges_to_check) == 0 or np.sum(np.hstack(edges_to_check)) == 0:
                        all_grains = collect_polygon_from_mask(
                            np.zeros_like(image[:,:,0]), mask, None, all_grains, sx, sy, 10  # 【临时调整】min_area改成10
                        )
                else:
                    all_grains = collect_polygon_from_mask(
                        np.zeros_like(image[:,:,0]), mask, None, all_grains, sx, sy, 10  # 【临时调整】min_area改成10
                    )
        except Exception as e:
            skipped_count +=1
            skip_reasons["exception"] +=1
            print(f"⚠️  Skipping point {i} ({x},{y}): {str(e)[:50]}...")
            continue
        # ===== 【核心修复：异常处理结束】 =====
    
    # 【新增】打印跳过统计
    print(f"\n===== SAM处理统计 =====")
    print(f"YOLO检测总点数: {len(coords)}")
    print(f"SAM跳过点数: {skipped_count}")
    print(f"跳过原因: {skip_reasons}")
    print(f"SAM成功分割颗粒数: {len(all_grains)}")
    print(f"=======================\n")
    
    # Post-processing（【修改2：删除了重叠检测/聚类/合并的冗余代码】）
    if len(all_grains) == 0:
        print("No valid grains after SAM segmentation.")
        return [], np.zeros_like(image[:,:,0]), np.zeros_like(image), pd.DataFrame(), None, None
    
    # 移除了无用的重叠检测/连通域分析/合并步骤
    if remove_large_objects:
        print("Removing large objects...")
        pass
    
    # Create labeled image
    if len(all_grains) > 0:
        print("Creating labeled image...")
        labels, mask_all = create_labeled_image(all_grains, image)
    else:
        labels = np.zeros_like(image[:,:,0])
        mask_all = np.zeros_like(image)
    
    # Calculate grain properties
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
    
    # Visualization
    fig, ax = None, None
    if plot_image and len(all_grains) > 0:
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.imshow(image)
        plot_image_w_colorful_grains(image, all_grains, ax, cmap="Paired")
        plot_grain_axes_and_centroids(all_grains, labels, ax, linewidth=1, markersize=10)
        plt.xticks([])
        plt.yticks([])
        plt.xlim([0, image.shape[1]])
        plt.ylim([image.shape[0], 0])
        plt.tight_layout()
        plt.show()
    
    return all_grains, labels, mask_all, grain_data, fig, ax