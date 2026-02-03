"""
UltraFastSAM终极引擎 - 专门针对岩石颗粒优化
文件名：seg_engine.py
功能：解决所有FastSAM在岩石颗粒分割中的问题
"""

import numpy as np
import torch
import time
import cv2
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from ultralytics import FastSAM
from typing import List, Tuple, Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class UltraFastSAM:
    """终极FastSAM引擎，专为岩石颗粒分割设计"""
    
    def __init__(self, model_path: str = "../models/FastSAM-s.pt", device: str = "cpu"):
        """
        初始化终极FastSAM引擎
        
        Args:
            model_path: FastSAM模型路径
            device: 运行设备 ('cpu' 或 'cuda')
        """
        print("=" * 60)
        print("🚀 初始化UltraFastSAM终极引擎...")
        print("=" * 60)
        
        self.device = device
        
        # 加载模型
        try:
            self.model = FastSAM(model_path)
            self.model.to(device=self.device)
            print(f"✅ FastSAM模型加载成功 (设备: {device})")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
        
        # === 超优化参数（专门针对岩石颗粒）===
        self.params = {
            # 全局推理参数（用于快速获取候选掩码）
            'global': {
                'imgsz': 1024,      # 大尺寸，保留小颗粒
                'conf': 0.15,       # 低置信度，提高召回率
                'iou': 0.3,         # 低IoU，避免抑制
                'retina_masks': True
            },
            # 框内推理参数（用于精细分割）
            'local': {
                'imgsz': 512,       # 中等尺寸
                'conf': 0.1,        # 极低置信度
                'iou': 0.2          # 极低IoU
            },
            # 小颗粒专用参数
            'small': {
                'imgsz': 256,       # 小尺寸
                'conf': 0.05,       # 极低置信度
                'iou': 0.1          # 极低IoU
            },
            # 掩码过滤参数
            'filter': {
                'min_area': 10,     # 最小面积10像素（岩石颗粒很小）
                'max_area_ratio': 0.8,  # 最大面积比例
                'min_solidity': 0.3,    # 最小实心度
                'min_extent': 0.1       # 最小范围
            },
            # 形态学参数
            'morphology': {
                'small_kernel': (3, 3),
                'medium_kernel': (5, 5),
                'large_kernel': (7, 7),
                'closing_iterations': 1,
                'opening_iterations': 1
            }
        }
        
        # 性能监控
        self.performance_stats = {
            'total_inferences': 0,
            'total_masks_generated': 0,
            'total_masks_filtered': 0,
            'total_time': 0.0
        }
        
        print("✅ UltraFastSAM引擎初始化完成")
        self.print_parameters()
    
    def print_parameters(self):
        """打印优化参数"""
        print("\n📊 UltraFastSAM优化参数:")
        print("  - 全局推理: 尺寸={}, 置信度={}, IoU={}".format(
            self.params['global']['imgsz'],
            self.params['global']['conf'],
            self.params['global']['iou']
        ))
        print("  - 局部推理: 尺寸={}, 置信度={}, IoU={}".format(
            self.params['local']['imgsz'],
            self.params['local']['conf'],
            self.params['local']['iou']
        ))
        print("  - 掩码过滤: 最小面积={}px, 最小实心度={}".format(
            self.params['filter']['min_area'],
            self.params['filter']['min_solidity']
        ))
    
    def inference_whole_image(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        对整个图像进行UltraFastSAM推理（生成候选掩码）
        
        Args:
            image: RGB图像 (H, W, 3)
            
        Returns:
            masks: 过滤后的掩码列表
            scores: 对应的置信度分数
        """
        start_time = time.time()
        h, w = image.shape[:2]
        
        try:
            # 运行UltraFastSAM推理（使用优化参数）
            results = self.model(
                image,
                device=self.device,
                imgsz=self.params['global']['imgsz'],
                conf=self.params['global']['conf'],
                iou=self.params['global']['iou'],
                retina_masks=True,
                verbose=False
            )
            
            self.performance_stats['total_inferences'] += 1
            
            if results[0].masks is None:
                print("⚠️ 全局推理未检测到任何掩码")
                return [], []
            
            # 获取掩码数据
            masks_data = results[0].masks.data.cpu().numpy()
            scores = results[0].masks.conf.cpu().numpy() if hasattr(results[0].masks, 'conf') else None
            
            # 处理掩码（使用智能过滤）
            processed_masks = []
            valid_scores = []
            
            for idx, mask in enumerate(masks_data):
                # 转换为二值掩码
                binary_mask = (mask > 0).astype(np.uint8)
                
                # 计算掩码属性
                mask_area = np.sum(binary_mask)
                img_area = h * w
                
                # 过滤掩码（使用智能过滤函数）
                if self._filter_mask_by_properties(binary_mask, h, w):
                    # 形态学增强
                    enhanced_mask = self._enhance_mask_morphology(binary_mask)
                    
                    # 计算掩码质量分数
                    mask_score = self._calculate_mask_quality(enhanced_mask)
                    
                    # 如果掩码质量太差，尝试修复
                    if mask_score < 0.3:
                        enhanced_mask = self._repair_mask(enhanced_mask)
                        mask_score = self._calculate_mask_quality(enhanced_mask)
                    
                    processed_masks.append(enhanced_mask * 255)
                    
                    # 使用质量分数或原始分数
                    if scores is not None and idx < len(scores):
                        final_score = scores[idx] * 0.7 + mask_score * 0.3
                    else:
                        final_score = mask_score
                    
                    valid_scores.append(float(final_score))
            
            inference_time = time.time() - start_time
            self.performance_stats['total_time'] += inference_time
            self.performance_stats['total_masks_generated'] += len(masks_data)
            self.performance_stats['total_masks_filtered'] += len(processed_masks)
            
            print(f"✅ 全局推理生成 {len(processed_masks)}/{len(masks_data)} 个有效掩码，耗时: {inference_time:.2f}s")
            return processed_masks, valid_scores
            
        except Exception as e:
            print(f"❌ 全局推理失败: {e}")
            import traceback
            traceback.print_exc()
            return [], []
    
    def _filter_mask_by_properties(self, mask: np.ndarray, img_h: int, img_w: int) -> bool:
        """
        基于属性智能过滤掩码
        """
        # 计算掩码属性
        mask_area = np.sum(mask > 0)
        img_area = img_h * img_w
        
        # 1. 面积过滤
        if mask_area < self.params['filter']['min_area']:
            return False
        
        if mask_area / img_area > self.params['filter']['max_area_ratio']:
            return False
        
        # 2. 实心度过滤（面积/凸包面积）
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        
        # 取最大轮廓
        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)
        
        if area == 0:
            return False
        
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            solidity = 0
        else:
            solidity = area / hull_area
        
        if solidity < self.params['filter']['min_solidity']:
            return False
        
        # 3. 范围过滤（掩码面积/边界框面积）
        x, y, w, h = cv2.boundingRect(main_contour)
        bbox_area = w * h
        
        if bbox_area == 0:
            extent = 0
        else:
            extent = area / bbox_area
        
        if extent < self.params['filter']['min_extent']:
            return False
        
        # 4. 形状过滤（排除过于细长的掩码）
        if h > 0:
            aspect_ratio = w / h
            if aspect_ratio > 5.0 or aspect_ratio < 0.2:
                return False
        
        return True
    
    def _enhance_mask_morphology(self, mask: np.ndarray) -> np.ndarray:
        """
        使用形态学操作增强掩码
        """
        # 1. 先闭运算填充小孔洞
        kernel_close = np.ones(self.params['morphology']['medium_kernel'], np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, 
                               iterations=self.params['morphology']['closing_iterations'])
        
        # 2. 再开运算去除小噪点
        kernel_open = np.ones(self.params['morphology']['small_kernel'], np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open,
                               iterations=self.params['morphology']['opening_iterations'])
        
        # 3. 填充孔洞
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
        
        return mask
    
    def _calculate_mask_quality(self, mask: np.ndarray) -> float:
        """
        计算掩码质量分数（0-1）
        """
        if mask.sum() == 0:
            return 0.0
        
        # 1. 紧凑度分数（周长^2 / 面积）
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        
        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        
        if area == 0:
            compactness = 0
        else:
            compactness = (perimeter ** 2) / (4 * np.pi * area)
            # 归一化：理想圆为1，越大越不紧凑
            compactness_score = 1.0 / min(compactness, 10.0)
        
        # 2. 实心度分数
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            solidity_score = 0
        else:
            solidity = area / hull_area
            solidity_score = solidity
        
        # 3. 边界平滑度分数
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        smoothness = len(approx) / max(perimeter, 1)
        smoothness_score = min(smoothness * 10, 1.0)
        
        # 综合分数
        total_score = compactness_score * 0.4 + solidity_score * 0.4 + smoothness_score * 0.2
        return float(total_score)
    
    def _repair_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        修复低质量掩码
        """
        # 1. 计算掩码属性
        area = mask.sum()
        h, w = mask.shape
        
        # 2. 如果是小掩码，使用膨胀增强
        if area < 100:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        # 3. 如果是大掩码但形状不好，使用腐蚀去除毛刺
        elif area > 1000:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        # 4. 确保掩码是连通的
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels > 1:
            # 保留最大连通域
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_label).astype(np.uint8)
        
        return mask
    
    def segment_single_box(self, image: np.ndarray, box: List[float]) -> Tuple[np.ndarray, float]:
        """
        对单个框进行精细分割
        
        Args:
            image: 原始图像
            box: [x1, y1, x2, y2]
            
        Returns:
            mask: 分割掩码
            score: 质量分数
        """
        x1, y1, x2, y2 = map(int, box)
        h, w = image.shape[:2]
        
        # 确保边界框在图像范围内
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros((h, w), dtype=np.uint8), 0.0
        
        # 裁剪区域
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((h, w), dtype=np.uint8), 0.0
        
        try:
            # 根据框大小选择参数
            crop_h, crop_w = crop.shape[:2]
            box_area = crop_h * crop_w
            img_area = h * w
            
            # 智能参数选择
            if box_area < 1000:  # 小框
                params = self.params['small']
            elif box_area < 10000:  # 中等框
                params = self.params['local']
            else:  # 大框
                params = self.params['global']
            
            # 运行FastSAM推理
            results = self.model(
                crop,
                device=self.device,
                imgsz=params['imgsz'],
                conf=params['conf'],
                iou=params['iou'],
                verbose=False
            )
            
            self.performance_stats['total_inferences'] += 1
            
            if results[0].masks is None:
                return np.zeros((h, w), dtype=np.uint8), 0.0
            
            # 获取所有掩码
            masks_data = results[0].masks.data.cpu().numpy()
            
            if len(masks_data) == 0:
                return np.zeros((h, w), dtype=np.uint8), 0.0
            
            # 选择最佳掩码
            best_mask = None
            best_score = -1
            
            for mask in masks_data:
                binary_mask = (mask > 0).astype(np.uint8)
                
                # 增强掩码
                enhanced_mask = self._enhance_mask_morphology(binary_mask)
                
                # 计算质量分数
                mask_score = self._calculate_mask_quality(enhanced_mask)
                
                if mask_score > best_score:
                    best_score = mask_score
                    best_mask = enhanced_mask
            
            if best_mask is not None and best_score > 0.1:
                # 创建完整图像掩码
                full_mask = np.zeros((h, w), dtype=np.uint8)
                mask_h, mask_w = best_mask.shape
                
                if mask_h > 0 and mask_w > 0:
                    # 确保掩码在裁剪区域内
                    actual_h = min(mask_h, y2 - y1)
                    actual_w = min(mask_w, x2 - x1)
                    
                    full_mask[y1:y1+actual_h, x1:x1+actual_w] = best_mask[:actual_h, :actual_w]
                
                return full_mask * 255, best_score
            
        except Exception as e:
            print(f"⚠️ 单框分割失败: {e}")
        
        return np.zeros((h, w), dtype=np.uint8), 0.0
    
    def intelligent_matching(self, 
                           yolo_boxes: List[List[float]], 
                           fastsam_masks: List[np.ndarray],
                           mask_scores: List[float],
                           image_shape: Tuple[int, int]) -> List[Optional[np.ndarray]]:
        """
        智能掩码匹配算法（改进版）
        
        使用多对多匹配 + 质量优先 + 重叠惩罚
        """
        if len(fastsam_masks) == 0:
            return [None] * len(yolo_boxes)
        
        h, w = image_shape[:2]
        n_boxes = len(yolo_boxes)
        n_masks = len(fastsam_masks)
        
        print(f"📊 智能匹配: {n_boxes}个框 vs {n_masks}个掩码")
        
        # 1. 计算成本矩阵（成本越低越好）
        cost_matrix = np.zeros((n_boxes, n_masks))
        
        for i, box in enumerate(yolo_boxes):
            x1, y1, x2, y2 = map(int, box)
            box_area = max((x2 - x1) * (y2 - y1), 1)
            
            for j, mask in enumerate(fastsam_masks):
                # 计算掩码在框内的部分
                mask_in_box = mask[y1:y2, x1:x2]
                intersection = np.sum(mask_in_box > 0)
                
                if intersection == 0:
                    cost = 10.0  # 高成本
                else:
                    # 计算覆盖率
                    coverage = intersection / box_area
                    
                    # 计算中心偏移
                    mask_indices = np.where(mask > 0)
                    if len(mask_indices[0]) > 0:
                        mask_center_y = np.mean(mask_indices[0])
                        mask_center_x = np.mean(mask_indices[1])
                    else:
                        mask_center_y, mask_center_x = h/2, w/2
                    
                    box_center_y = (y1 + y2) / 2
                    box_center_x = (x1 + x2) / 2
                    
                    center_distance = np.sqrt((mask_center_x - box_center_x)**2 + 
                                             (mask_center_y - box_center_y)**2)
                    
                    # 归一化距离
                    norm_distance = center_distance / np.sqrt(h**2 + w**2)
                    
                    # 成本 = 低覆盖率 + 高距离 + 低质量
                    cost = (1.0 - coverage) * 0.5 + norm_distance * 0.3 + (1.0 - mask_scores[j]) * 0.2
                
                cost_matrix[i, j] = cost
        
        # 2. 使用匈牙利算法进行最优分配
        assigned_masks = [None] * n_boxes
        mask_used = [False] * n_masks
        
        if n_boxes > 0 and n_masks > 0:
            # 匈牙利算法找到最小成本分配
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # 应用分配（成本阈值）
            for i, j in zip(row_ind, col_ind):
                if i < n_boxes and j < n_masks and cost_matrix[i, j] < 0.7:  # 成本阈值
                    assigned_masks[i] = fastsam_masks[j]
                    mask_used[j] = True
            
            # 3. 二次分配：为未分配的框寻找次优掩码
            for i in range(n_boxes):
                if assigned_masks[i] is None:
                    # 按成本排序
                    sorted_indices = np.argsort(cost_matrix[i])
                    
                    for j in sorted_indices:
                        if not mask_used[j] and cost_matrix[i, j] < 1.0:  # 更宽松的阈值
                            assigned_masks[i] = fastsam_masks[j]
                            mask_used[j] = True
                            break
            
            # 4. 统计分配结果
            assigned_count = sum(1 for mask in assigned_masks if mask is not None)
            print(f"✅ 智能匹配完成: {assigned_count}/{n_boxes} 个框获得掩码")
            
        return assigned_masks
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = self.performance_stats.copy()
        
        if stats['total_inferences'] > 0:
            stats['avg_time_per_inference'] = stats['total_time'] / stats['total_inferences']
        else:
            stats['avg_time_per_inference'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """重置性能统计"""
        self.performance_stats = {
            'total_inferences': 0,
            'total_masks_generated': 0,
            'total_masks_filtered': 0,
            'total_time': 0.0
        }


if __name__ == "__main__":
    print("UltraFastSAM引擎测试")
    print("=" * 60)
    
    # 测试代码
    engine = UltraFastSAM(device="cpu")
    print("✅ 引擎测试通过")