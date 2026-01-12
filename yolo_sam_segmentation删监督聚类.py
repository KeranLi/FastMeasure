import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import Polygon, Point
from shapely.affinity import translate
import warnings
from typing import List, Tuple, Optional
import sys

# YOLO和SAM导入
import torch
from ultralytics import YOLO
from segment_anything import SamPredictor

# 导入简化版函数库
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
    使用YOLO模型检测颗粒并提取中心点作为SAM提示
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
    
    # 创建DataFrame
    detections_df = pd.DataFrame(valid_detections)
    detections_df = detections_df.sort_values('confidence', ascending=False)
    
    # 提取中心坐标
    coords = detections_df[['center_x', 'center_y']].values.astype(np.int32)
    
    print(f"YOLO检测到 {len(coords)} 个颗粒，置信度 > {conf_threshold}")
    return coords, detections_df


def yolo_sam_segmentation(
    image: np.ndarray,
    yolo_model: YOLO,
    sam_predictor: SamPredictor,
    conf_threshold: float = 0.5,
    min_area: int = 100,
    min_bbox_area: int = 100,
    remove_edge_grains: bool = False,
    class_id: Optional[int] = None,
    plot_image: bool = False,
    keep_edges: Optional[dict] = None
) -> Tuple[List[Polygon], np.ndarray, np.ndarray, pd.DataFrame, Optional[plt.Figure], Optional[plt.Axes]]:
    """
    简化的主分割函数
    移除了所有无监督聚类、重叠检测和合并逻辑
    """
    # YOLO检测
    print("运行YOLO检测...")
    coords, detections_df = detect_grains_yolo(
        image, yolo_model, conf_threshold, min_bbox_area, class_id
    )
    
    if len(coords) == 0:
        print("YOLO未检测到任何颗粒。")
        return [], np.zeros_like(image[:,:,0]), np.zeros_like(image), pd.DataFrame(), None, None
    
    # SAM分割
    print(f"对 {len(coords)} 个提示点运行SAM分割...")
    sam_predictor.set_image(image)
    all_grains = []
    
    for i in tqdm(range(len(coords)), desc="SAM分割进度"):
        try:
            x, y = coords[i]
            
            # 跳过越界坐标
            if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
                continue
                
            # 单点SAM分割
            sx, sy, mask = one_point_prompt(x, y, image, sam_predictor, ax=False)
            
            # 安全检查
            if mask is None or not isinstance(mask, np.ndarray) or mask.size == 0:
                continue
                
            if np.max(mask) > 0:
                # 检查是否接触边缘
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
                    
                    # 只有不接触指定边缘的颗粒才保留
                    if len(edges_to_check) == 0 or np.sum(np.hstack(edges_to_check)) == 0:
                        all_grains = collect_polygon_from_mask(
                            np.zeros_like(image[:,:,0]), mask, None, all_grains, sx, sy, min_area
                        )
                else:
                    all_grains = collect_polygon_from_mask(
                        np.zeros_like(image[:,:,0]), mask, None, all_grains, sx, sy, min_area
                    )
        except Exception as e:
            # 跳过问题点，继续处理
            print(f"⚠️ 跳过点 {i} ({x},{y}): {str(e)[:50]}...")
            continue
    
    # 后处理：没有复杂的聚类/合并步骤
    if len(all_grains) == 0:
        print("SAM分割后没有有效的颗粒。")
        return [], np.zeros_like(image[:,:,0]), np.zeros_like(image), pd.DataFrame(), None, None
    
    print(f"成功分割 {len(all_grains)} 个颗粒")
    
    # 创建标签图像
    print("创建标签图像...")
    labels, mask_all = create_labeled_image(all_grains, image)
    
    # 计算颗粒属性
    if np.max(labels) > 0:
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
    else:
        grain_data = pd.DataFrame()
    
    # 可视化
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


def batch_process_images(
    image_paths: List[str],
    yolo_model: YOLO,
    sam_predictor: SamPredictor,
    conf_threshold: float = 0.5,
    min_area: int = 100,
    min_bbox_area: int = 100,
    output_dir: str = "results",
    plot_each: bool = False
) -> dict:
    """
    批量处理图像
    """
    import os
    from pathlib import Path
    
    results = {}
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for img_path in tqdm(image_paths, desc="批量处理"):
        try:
            # 加载图像
            from PIL import Image
            image = np.array(Image.open(img_path).convert('RGB'))
            
            # 处理图像
            all_grains, labels, mask_all, grain_data, fig, ax = yolo_sam_segmentation(
                image=image,
                yolo_model=yolo_model,
                sam_predictor=sam_predictor,
                conf_threshold=conf_threshold,
                min_area=min_area,
                min_bbox_area=min_bbox_area,
                plot_image=plot_each
            )
            
            # 保存结果
            img_name = Path(img_path).stem
            img_output_dir = output_path / img_name
            img_output_dir.mkdir(exist_ok=True)
            
            # 保存可视化结果
            if fig is not None:
                fig.savefig(img_output_dir / "segmentation_result.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
            
            # 保存掩码
            if mask_all is not None:
                from PIL import Image as PILImage
                mask_img = PILImage.fromarray((mask_all * 127).astype(np.uint8))
                mask_img.save(img_output_dir / "mask.png")
            
            # 保存统计数据
            if not grain_data.empty:
                grain_data.to_csv(img_output_dir / "grain_statistics.csv", index=False)
            
            # 记录结果
            results[img_path] = {
                'success': True,
                'grains_count': len(all_grains),
                'output_dir': str(img_output_dir)
            }
            
        except Exception as e:
            results[img_path] = {
                'success': False,
                'error': str(e),
                'grains_count': 0
            }
            print(f"处理 {img_path} 失败: {e}")
    
    # 保存汇总报告
    summary = {
        'total_images': len(image_paths),
        'successful': sum(1 for r in results.values() if r['success']),
        'failed': sum(1 for r in results.values() if not r['success']),
        'total_grains': sum(r['grains_count'] for r in results.values() if r['success']),
        'results': results
    }
    
    import json
    with open(output_path / "batch_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n批量处理完成！")
    print(f"总图像: {summary['total_images']}")
    print(f"成功: {summary['successful']}")
    print(f"失败: {summary['failed']}")
    print(f"总颗粒数: {summary['total_grains']}")
    
    return summary


# ==================== 示例使用代码 ====================

if __name__ == "__main__":
    # 示例：如何单独使用这个模块
    
    # 1. 加载模型
    yolo_model = YOLO("models/best.pt")
    
    # 2. 加载SAM模型
    import torch
    from segment_anything import sam_model_registry
    
    sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth")
    sam.to(device='cuda' if torch.cuda.is_available() else 'cpu')
    sam_predictor = SamPredictor(sam)
    
    # 3. 加载图像
    from PIL import Image
    image_path = "your_image.jpg"
    image = np.array(Image.open(image_path).convert('RGB'))
    
    # 4. 运行分割
    all_grains, labels, mask_all, grain_data, fig, ax = yolo_sam_segmentation(
        image=image,
        yolo_model=yolo_model,
        sam_predictor=sam_predictor,
        conf_threshold=0.25,
        min_area=30,
        min_bbox_area=20,
        plot_image=True
    )
    
    # 5. 输出结果
    print(f"分割完成！找到 {len(all_grains)} 个颗粒")
    if not grain_data.empty:
        print(f"颗粒统计数据已保存，平均面积: {grain_data['area'].mean():.2f} 像素")