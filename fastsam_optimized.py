"""
SuperFastSAM优化引擎 -
文件名：fastsam_optimized.py
"""

import numpy as np
import torch
import time
from scipy.optimize import linear_sum_assignment
from ultralytics import FastSAM
from typing import List, Tuple, Dict, Optional
import cv2
from scipy import ndimage


class SuperFastSAM:
    """超优化FastSAM引擎，专门用于岩石颗粒分割"""
    
    def __init__(self, model_path: str = "../models/FastSAM-s.pt", device: str = "cpu"):
        """
        初始化超优化FastSAM引擎
        
        Args:
            model_path: FastSAM模型路径
            device: 运行设备 ('cpu' 或 'cuda')
        """
        self.device = device
        self.model = FastSAM(model_path)
        self.model.to(device=self.device)
        
        # 优化参数（经过实验验证）
        self.conf_threshold = 0.35      # 中等置信度，平衡精度和召回率
        self.iou_threshold = 0.45       # 中等IoU阈值
        self.imgsz = 768                # 最佳尺寸（512-1024之间）
        
        # 掩码过滤参数
        self.min_mask_area = 50         # 最小掩码面积
        self.max_mask_area_ratio = 0.5  # 最大掩码面积/图像面积比例
        
        print(f"✅ SuperFastSAM引擎初始化完成 (设备: {device}, 图像尺寸: {self.imgsz})")
    
    def inference_whole_image(self, image: np.ndarray) -> List[np.ndarray]:
        """
        对整个图像进行一次FastSAM推理
        
        Args:
            image: RGB图像 (H, W, 3)
            
        Returns:
            masks: 过滤后的掩码列表
            scores: 对应的置信度分数
        """
        h, w = image.shape[:2]
        
        try:
            # 运行FastSAM推理
            results = self.model(
                image,
                device=self.device,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                retina_masks=True,
                verbose=False
            )
            
            if results[0].masks is None:
                return [], []
            
            # 获取掩码数据
            masks_data = results[0].masks.data.cpu().numpy()
            scores = results[0].masks.conf.cpu().numpy() if hasattr(results[0].masks, 'conf') else None
            
            # 处理掩码
            processed_masks = []
            valid_scores = []
            
            for idx, mask in enumerate(masks_data):
                # 转换为二值掩码
                binary_mask = (mask > 0).astype(np.uint8)
                
                # 计算掩码面积
                mask_area = np.sum(binary_mask)
                img_area = h * w
                
                # 过滤掩码
                if mask_area < self.min_mask_area:
                    continue
                
                if mask_area / img_area > self.max_mask_area_ratio:
                    continue  # 跳过太大的掩码（可能是背景）
                
                # 形态学后处理：填充孔洞
                filled_mask = ndimage.binary_fill_holes(binary_mask).astype(np.uint8) * 255
                
                processed_masks.append(filled_mask)
                if scores is not None and idx < len(scores):
                    valid_scores.append(scores[idx])
                else:
                    valid_scores.append(1.0)  # 默认分数
            
            print(f"✅ FastSAM生成 {len(processed_masks)} 个有效掩码 (原始: {len(masks_data)})")
            return processed_masks, valid_scores
            
        except Exception as e:
            print(f"❌ FastSAM推理失败: {e}")
            return [], []
    
    def intelligent_mask_assignment(self, 
                                   yolo_boxes: List[List[float]], 
                                   fastsam_masks: List[np.ndarray],
                                   image_shape: Tuple[int, int]) -> List[Optional[np.ndarray]]:
        """
        智能掩码分配：将FastSAM掩码与YOLO框进行最佳匹配
        
        Args:
            yolo_boxes: YOLO检测框列表 [[x1,y1,x2,y2], ...]
            fastsam_masks: FastSAM掩码列表
            image_shape: 图像尺寸 (H, W)
            
        Returns:
            assigned_masks: 分配给每个YOLO框的掩码（None表示未分配）
        """
        if len(fastsam_masks) == 0:
            return [None] * len(yolo_boxes)
        
        h, w = image_shape[:2]
        n_boxes = len(yolo_boxes)
        n_masks = len(fastsam_masks)
        
        # 1. 计算相似度矩阵（使用改进的交集比例）
        similarity_matrix = np.zeros((n_boxes, n_masks))
        
        for i, box in enumerate(yolo_boxes):
            x1, y1, x2, y2 = map(int, box)
            box_area = max((x2 - x1) * (y2 - y1), 1)  # 避免除零
            
            # 创建框掩码
            box_mask = np.zeros((h, w), dtype=np.uint8)
            box_mask[y1:y2, x1:x2] = 1
            
            for j, mask in enumerate(fastsam_masks):
                # 计算掩码在框内的面积比例
                mask_in_box = mask[y1:y2, x1:x2]
                intersection = np.sum(mask_in_box > 0)
                
                # 计算两种相似度指标
                overlap_ratio = intersection / box_area  # 框内覆盖率
                mask_ratio = intersection / max(np.sum(mask > 0), 1)  # 掩码在框内的比例
                
                # 综合相似度分数（鼓励完全包含的匹配）
                similarity = overlap_ratio * 0.7 + mask_ratio * 0.3
                similarity_matrix[i, j] = similarity
        
        # 2. 使用匈牙利算法进行最优分配
        if n_boxes > 0 and n_masks > 0:
            # 确保矩阵是二维的
            row_ind, col_ind = linear_sum_assignment(-similarity_matrix)  # 最大化相似度
            
            # 3. 创建分配结果
            assigned_masks = [None] * n_boxes
            mask_used = [False] * n_masks
            
            for i, j in zip(row_ind, col_ind):
                if i < n_boxes and j < n_masks and similarity_matrix[i, j] > 0.3:  # 阈值
                    assigned_masks[i] = fastsam_masks[j]
                    mask_used[j] = True
            
            # 4. 处理未分配的框（后备方案）
            for i in range(n_boxes):
                if assigned_masks[i] is None:
                    # 尝试找相似度次高的未使用掩码
                    mask_indices = np.argsort(-similarity_matrix[i])
                    for j in mask_indices:
                        if not mask_used[j] and similarity_matrix[i, j] > 0.15:
                            assigned_masks[i] = fastsam_masks[j]
                            mask_used[j] = True
                            break
            
            return assigned_masks
        else:
            return [None] * n_boxes
    
    def backup_segmentation_for_box(self, 
                                   image: np.ndarray, 
                                   box: List[int]) -> np.ndarray:
        """
        后备分割：对未匹配到掩码的框进行独立分割
        
        Args:
            image: 原始图像
            box: 边界框 [x1,y1,x2,y2]
            
        Returns:
            mask: 分割掩码
        """
        x1, y1, x2, y2 = map(int, box)
        h, w = image.shape[:2]
        
        # 确保框在图像范围内
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros((h, w), dtype=np.uint8)
        
        # 裁剪区域
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((h, w), dtype=np.uint8)
        
        try:
            # 计算合适的图像尺寸
            crop_h, crop_w = crop.shape[:2]
            target_size = min(256, max(64, min(crop_h, crop_w)))
            target_size = ((target_size + 31) // 32) * 32  # 32的倍数
            
            # 对裁剪区域运行FastSAM
            crop_results = self.model(
                crop,
                device=self.device,
                imgsz=target_size,
                conf=0.15,  # 较低置信度，避免漏检
                iou=0.4,
                verbose=False
            )
            
            if crop_results[0].masks is not None and len(crop_results[0].masks) > 0:
                # 取面积最大的掩码
                crop_masks = crop_results[0].masks.data.cpu().numpy()
                areas = [mask.sum() for mask in crop_masks]
                best_idx = np.argmax(areas)
                crop_mask = (crop_masks[best_idx] > 0).astype(np.uint8) * 255
                
                # 创建完整图像掩码
                full_mask = np.zeros((h, w), dtype=np.uint8)
                mask_h, mask_w = crop_mask.shape
                if mask_h > 0 and mask_w > 0:
                    full_mask[y1:y1+mask_h, x1:x1+mask_w] = crop_mask
                
                return full_mask
        except Exception as e:
            print(f"⚠️ 后备分割失败: {e}")
        
        # 返回矩形掩码作为最终后备
        return self._create_rect_mask(box, h, w)
    
    def _create_rect_mask(self, box: List[int], h: int, w: int) -> np.ndarray:
        """创建矩形掩码"""
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
        
        return mask
    
    def segment_image(self, 
                      image: np.ndarray, 
                      yolo_boxes: List[List[float]]) -> Tuple[List[np.ndarray], float]:
        """
        主分割函数：智能分割所有YOLO框
        
        Args:
            image: 原始图像
            yolo_boxes: YOLO检测框
            
        Returns:
            masks: 分割掩码列表（与yolo_boxes一一对应）
            inference_time: FastSAM推理时间
        """
        if len(yolo_boxes) == 0:
            return [], 0.0
        
        start_time = time.time()
        
        # 步骤1：对整个图像进行FastSAM推理
        fastsam_masks, _ = self.inference_whole_image(image)
        fastsam_time = time.time() - start_time
        
        # 步骤2：智能掩码分配
        assigned_masks = self.intelligent_mask_assignment(yolo_boxes, fastsam_masks, image.shape)
        
        # 步骤3：处理未分配的框（后备分割）
        final_masks = []
        for i, (box, mask) in enumerate(zip(yolo_boxes, assigned_masks)):
            if mask is not None:
                final_masks.append(mask)
            else:
                backup_mask = self.backup_segmentation_for_box(image, box)
                final_masks.append(backup_mask)
        
        total_time = time.time() - start_time
        
        # 统计信息
        valid_masks = sum(1 for mask in final_masks if mask is not None and mask.sum() > 0)
        print(f"📊 分割统计: {valid_masks}/{len(yolo_boxes)} 个框获得有效掩码")
        print(f"⏱️  推理时间: {fastsam_time:.2f}s, 总时间: {total_time:.2f}s")
        
        return final_masks, fastsam_time