"""
MobileSAM引擎针对岩石颗粒优化
文件名：mobile_sam_engine.py
功能：MobileSAM引擎，专为岩石颗粒分割设计，解决MobileSAM在岩石颗粒上的问题
"""

import numpy as np
import torch
import time
import cv2
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# MobileSAM导入
try:
    from mobile_sam import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    MOBILESAM_AVAILABLE = True
except ImportError:
    MOBILESAM_AVAILABLE = False
    print("MobileSAM库未安装，请安装")

# Segment Anything导入（作为备选）
try:
    from segment_anything import sam_model_registry as sam_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


class MobileSAMEngine:
    """MobileSAM引擎，为岩石颗粒分割设计"""
    
    def __init__(self, model_path: str = "models/mobile_sam.pt", 
                 device: str = "cuda", model_type: str = "vit_t"):
        """
        初始化MobileSAM
        
        Args:
            model_path: MobileSAM模型路径
            device: 运行设备 ('cpu' 或 'cuda')
            model_type: 模型类型 ('vit_t' for MobileSAM)
        """
        print("=" * 60)
        print("初始化MobileSAM终极引擎")
        print("=" * 60)
        
        self.device = device
        self.model_type = model_type
        
        # 检查MobileSAM可用性
        if not MOBILESAM_AVAILABLE:
            print("MobileSAM库未安装")
            print("请安装")
            raise ImportError("MobileSAM库未安装")
        
        # 加载模型
        try:
            print(f"加载MobileSAM模型: {model_path}")
            self.sam = sam_model_registry[model_type](checkpoint=model_path)
            self.sam.to(device=self.device)
            
            # 创建预测器
            self.predictor = SamPredictor(self.sam)
            
            # 创建自动掩码生成器（用于自动分割）
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=32,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                box_nms_thresh=0.7,
                crop_n_layers=0,
                crop_n_points_downscale_factor=1,
                min_mask_region_area=10,
            )
            
            print(f"✅ MobileSAM模型加载成功 (设备: {device})")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
        
        # === MobileSAM优化参数（针对岩石颗粒）===
        self.params = {
            # 自动掩码生成参数
            'auto_mask': {
                'points_per_side': 32,
                'pred_iou_thresh': 0.88,
                'stability_score_thresh': 0.95,
                'box_nms_thresh': 0.7,
                'crop_n_layers': 0,
                'min_mask_region_area': 10,
            },
            # 框提示参数
            'box_prompt': {
                'box_expansion': 1.15,      # 框扩展系数
                'multimask_output': False,  # 单掩码输出
            },
            # 点提示参数
            'point_prompt': {
                'multimask_output': True,   # 多掩码输出
                'mask_threshold': 0.0,      # 掩码阈值
            },
            # 掩码过滤参数
            'filter': {
                'min_area': 10,
                'max_area_ratio': 0.8,
                'min_solidity': 0.3,
                'min_extent': 0.1,
                'min_confidence': 0.3,
            },
            # 形态学参数
            'morphology': {
                'small_kernel': (3, 3),
                'medium_kernel': (5, 5),
                'large_kernel': (7, 7),
                'closing_iterations': 1,
                'opening_iterations': 1,
                'dilation_iterations': 1,
            },
            # 多尺度推理
            'multi_scale': {
                'enabled': True,
                'scales': [0.8, 1.0, 1.2],
                'merge_strategy': 'union',
            }
        }
        
        # 性能监控
        self.performance_stats = {
            'total_inferences': 0,
            'total_masks_generated': 0,
            'total_masks_filtered': 0,
            'total_time': 0.0,
            'box_prompts': 0,
            'point_prompts': 0,
            'auto_masks': 0,
        }
        
        print("✅ MobileSAM引擎初始化完成")
        self.print_parameters()
    
    def print_parameters(self):
        """打印优化参数"""
        print("\n MobileSAM优化参数:")
        print("  - 自动掩码生成: points_per_side={}, iou_thresh={}".format(
            self.params['auto_mask']['points_per_side'],
            self.params['auto_mask']['pred_iou_thresh']
        ))
        print("  - 框提示: box_expansion={}, multimask={}".format(
            self.params['box_prompt']['box_expansion'],
            self.params['box_prompt']['multimask_output']
        ))
        print("  - 掩码过滤: 最小面积={}px, 最小置信度={}".format(
            self.params['filter']['min_area'],
            self.params['filter']['min_confidence']
        ))
        print("  - 多尺度推理: {}".format(
            "启用" if self.params['multi_scale']['enabled'] else "禁用"
        ))
    
    def set_image(self, image: np.ndarray):
        """设置当前图像（必须先调用）"""
        self.predictor.set_image(image)
        self.current_image = image
        self.image_shape = image.shape[:2]
    
    def segment_with_box(self, box: List[float]) -> Tuple[np.ndarray, float]:
        """
        使用边界框进行分割（主要方法）
        
        Args:
            box: [x1, y1, x2, y2] 边界框
            
        Returns:
            mask: 分割掩码 (0/1)
            score: 置信度分数
        """
        start_time = time.time()
        
        try:
            # 扩展边界框
            expanded_box = self._expand_box(box, self.params['box_prompt']['box_expansion'])
            
            # 确保框在图像范围内
            h, w = self.image_shape
            x1, y1, x2, y2 = expanded_box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return np.zeros(self.image_shape, dtype=np.uint8), 0.0
            
            # 转换为numpy数组
            input_box = np.array([x1, y1, x2, y2])
            
            # 预测掩码
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=self.params['box_prompt']['multimask_output'],
            )
            
            self.performance_stats['box_prompts'] += 1
            self.performance_stats['total_inferences'] += 1
            
            if len(masks) == 0:
                return np.zeros(self.image_shape, dtype=np.uint8), 0.0
            
            # 选择最佳掩码
            if self.params['box_prompt']['multimask_output']:
                best_idx = np.argmax(scores)
                mask = masks[best_idx]
                score = float(scores[best_idx])
            else:
                mask = masks[0]
                score = 0.8  # 如果没有分数，使用默认值
            
            # 转换为二值掩码
            binary_mask = (mask > 0).astype(np.uint8)
            
            # 过滤掩码
            if not self._filter_mask_by_properties(binary_mask):
                return np.zeros(self.image_shape, dtype=np.uint8), 0.0
            
            # 形态学增强
            enhanced_mask = self._enhance_mask_morphology(binary_mask)
            
            # 计算掩码质量
            mask_quality = self._calculate_mask_quality(enhanced_mask)
            final_score = score * 0.7 + mask_quality * 0.3
            
            inference_time = time.time() - start_time
            self.performance_stats['total_time'] += inference_time
            self.performance_stats['total_masks_generated'] += 1
            
            if final_score >= self.params['filter']['min_confidence']:
                self.performance_stats['total_masks_filtered'] += 1
                return enhanced_mask, final_score
            else:
                return np.zeros(self.image_shape, dtype=np.uint8), 0.0
            
        except Exception as e:
            print(f"框分割失败: {e}")
            return np.zeros(self.image_shape, dtype=np.uint8), 0.0
    
    def segment_with_box_and_point(self, box: List[float], point: Tuple[float, float]) -> Tuple[np.ndarray, float]:
        """
        使用边界框+中心点进行分割（更稳定）
        
        Args:
            box: [x1, y1, x2, y2] 边界框
            point: (x, y) 中心点
            
        Returns:
            mask: 分割掩码
            score: 置信度分数
        """
        try:
            # 扩展边界框
            expanded_box = self._expand_box(box, self.params['box_prompt']['box_expansion'])
            
            # 确保框和点在图像范围内
            h, w = self.image_shape
            x1, y1, x2, y2 = expanded_box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            px, py = point
            px, py = max(0, px), max(0, py)
            px, py = min(w-1, px), min(h-1, py)
            
            if x2 <= x1 or y2 <= y1:
                return np.zeros(self.image_shape, dtype=np.uint8), 0.0
            
            # 准备输入
            input_box = np.array([x1, y1, x2, y2])
            input_point = np.array([[px, py]])
            input_label = np.array([1])  # 前景点
            
            # 预测掩码
            masks, scores, _ = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box[None, :],
                multimask_output=True,
            )
            
            self.performance_stats['box_prompts'] += 1
            self.performance_stats['point_prompts'] += 1
            self.performance_stats['total_inferences'] += 1
            
            if len(masks) == 0:
                return np.zeros(self.image_shape, dtype=np.uint8), 0.0
            
            # 选择最佳掩码
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = float(scores[best_idx])
            
            # 转换为二值掩码
            binary_mask = (mask > 0).astype(np.uint8)
            
            # 过滤掩码
            if not self._filter_mask_by_properties(binary_mask):
                return np.zeros(self.image_shape, dtype=np.uint8), 0.0
            
            # 形态学增强
            enhanced_mask = self._enhance_mask_morphology(binary_mask)
            
            # 计算掩码质量
            mask_quality = self._calculate_mask_quality(enhanced_mask)
            final_score = score * 0.7 + mask_quality * 0.3
            
            self.performance_stats['total_masks_generated'] += 1
            
            if final_score >= self.params['filter']['min_confidence']:
                self.performance_stats['total_masks_filtered'] += 1
                return enhanced_mask, final_score
            else:
                return np.zeros(self.image_shape, dtype=np.uint8), 0.0
            
        except Exception as e:
            print(f"框+点分割失败: {e}")
            return np.zeros(self.image_shape, dtype=np.uint8), 0.0
    
    def generate_auto_masks(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        自动生成掩码（无需提示）
        
        Args:
            image: 输入图像
            
        Returns:
            masks: 掩码列表
            scores: 分数列表
        """
        start_time = time.time()
        
        try:
            # 生成自动掩码
            anns = self.mask_generator.generate(image)
            
            self.performance_stats['auto_masks'] += 1
            self.performance_stats['total_inferences'] += 1
            
            masks = []
            scores = []
            
            for ann in anns:
                mask = ann['segmentation']
                score = ann.get('predicted_iou', 0.5) * ann.get('stability_score', 0.5)
                
                # 转换为二值掩码
                binary_mask = mask.astype(np.uint8)
                
                # 过滤掩码
                if self._filter_mask_by_properties(binary_mask):
                    # 形态学增强
                    enhanced_mask = self._enhance_mask_morphology(binary_mask)
                    
                    # 计算掩码质量
                    mask_quality = self._calculate_mask_quality(enhanced_mask)
                    final_score = score * 0.7 + mask_quality * 0.3
                    
                    if final_score >= self.params['filter']['min_confidence']:
                        masks.append(enhanced_mask)
                        scores.append(final_score)
            
            inference_time = time.time() - start_time
            self.performance_stats['total_time'] += inference_time
            self.performance_stats['total_masks_generated'] += len(masks)
            self.performance_stats['total_masks_filtered'] += len(masks)
            
            print(f"自动掩码生成: {len(masks)}个有效掩码，耗时: {inference_time:.2f}s")
            return masks, scores
            
        except Exception as e:
            print(f"自动掩码生成失败: {e}")
            return [], []
    
    def _expand_box(self, box: List[float], expansion: float) -> List[float]:
        """扩展边界框"""
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # 计算扩展量
        dx = width * (expansion - 1) / 2
        dy = height * (expansion - 1) / 2
        
        return [x1 - dx, y1 - dy, x2 + dx, y2 + dy]
    
    def _filter_mask_by_properties(self, mask: np.ndarray) -> bool:
        """基于属性智能过滤掩码"""
        if mask.sum() == 0:
            return False
        
        # 计算掩码属性
        mask_area = mask.sum()
        img_area = mask.shape[0] * mask.shape[1]
        
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
        """使用形态学操作增强掩码"""
        # 1. 先闭运算填充小孔洞
        kernel_close = np.ones(self.params['morphology']['medium_kernel'], np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, 
                               iterations=self.params['morphology']['closing_iterations'])
        
        # 2. 再开运算去除小噪点
        kernel_open = np.ones(self.params['morphology']['small_kernel'], np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open,
                               iterations=self.params['morphology']['opening_iterations'])
        
        # 3. 轻微膨胀以增强边界
        kernel_dilate = np.ones(self.params['morphology']['small_kernel'], np.uint8)
        mask = cv2.dilate(mask, kernel_dilate, 
                         iterations=self.params['morphology']['dilation_iterations'])
        
        # 4. 填充孔洞
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
        
        return mask
    
    def _calculate_mask_quality(self, mask: np.ndarray) -> float:
        """计算掩码质量分数（0-1）"""
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
    
    def multi_scale_segmentation(self, image: np.ndarray, boxes: List[List[float]]) -> List[np.ndarray]:
        """
        多尺度分割（提高小颗粒检测率）
        
        Args:
            image: 输入图像
            boxes: 边界框列表
            
        Returns:
            掩码列表
        """
        if not self.params['multi_scale']['enabled']:
            # 单尺度分割
            self.set_image(image)
            masks = []
            for box in boxes:
                mask, _ = self.segment_with_box(box)
                if mask.sum() > 0:
                    masks.append(mask)
            return masks
        
        # 多尺度分割
        all_masks = []
        
        for scale in self.params['multi_scale']['scales']:
            # 调整图像大小
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 调整边界框
            scaled_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                scaled_box = [
                    int(x1 * scale), int(y1 * scale),
                    int(x2 * scale), int(y2 * scale)
                ]
                scaled_boxes.append(scaled_box)
            
            # 在当前尺度下分割
            self.set_image(scaled_image)
            
            for i, box in enumerate(scaled_boxes):
                mask, _ = self.segment_with_box(box)
                
                if mask.sum() > 0:
                    # 调整掩码大小回原始尺寸
                    if scale != 1.0:
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        mask = (mask > 0.5).astype(np.uint8)
                    
                    all_masks.append(mask)
        
        # 合并多尺度结果
        if len(all_masks) > 0:
            merged_masks = self._merge_multi_scale_masks(all_masks)
            return merged_masks
        else:
            return []
    
    def _merge_multi_scale_masks(self, masks: List[np.ndarray]) -> List[np.ndarray]:
        """合并多尺度掩码"""
        if len(masks) == 0:
            return []
        
        # 简单去重：合并高度重叠的掩码
        merged_masks = []
        used = [False] * len(masks)
        
        for i in range(len(masks)):
            if used[i]:
                continue
            
            current_mask = masks[i]
            
            # 查找重叠掩码
            for j in range(i + 1, len(masks)):
                if used[j]:
                    continue
                
                # 计算IoU
                intersection = np.logical_and(current_mask, masks[j]).sum()
                union = np.logical_or(current_mask, masks[j]).sum()
                
                if union > 0 and intersection / union > 0.5:  # IoU > 0.5
                    # 合并掩码
                    current_mask = np.logical_or(current_mask, masks[j]).astype(np.uint8)
                    used[j] = True
            
            if current_mask.sum() > 0:
                merged_masks.append(current_mask)
                used[i] = True
        
        return merged_masks
    
    def intelligent_matching(self, 
                           yolo_boxes: List[List[float]], 
                           mobilesam_masks: List[np.ndarray],
                           mask_scores: List[float],
                           image_shape: Tuple[int, int]) -> List[Optional[np.ndarray]]:
        """
        智能掩码匹配算法
        
        使用多对多匹配 + 质量优先 + 重叠惩罚
        """
        if len(mobilesam_masks) == 0:
            return [None] * len(yolo_boxes)
        
        h, w = image_shape[:2]
        n_boxes = len(yolo_boxes)
        n_masks = len(mobilesam_masks)
        
        print(f"智能匹配: {n_boxes}个框 vs {n_masks}个掩码")
        
        # 1. 计算成本矩阵（成本越低越好）
        cost_matrix = np.zeros((n_boxes, n_masks))
        
        for i, box in enumerate(yolo_boxes):
            x1, y1, x2, y2 = map(int, box)
            box_area = max((x2 - x1) * (y2 - y1), 1)
            box_center_x = (x1 + x2) / 2
            box_center_y = (y1 + y2) / 2
            
            for j, mask in enumerate(mobilesam_masks):
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
                    
                    center_distance = np.sqrt((mask_center_x - box_center_x)**2 + 
                                             (mask_center_y - box_center_y)**2)
                    
                    # 归一化距离
                    norm_distance = center_distance / np.sqrt(h**2 + w**2)
                    
                    # 成本 = 低覆盖率 + 高距离 + 低质量
                    cost = (1.0 - coverage) * 0.4 + norm_distance * 0.3 + (1.0 - mask_scores[j]) * 0.3
                
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
                    assigned_masks[i] = mobilesam_masks[j]
                    mask_used[j] = True
            
            # 3. 二次分配：为未分配的框寻找次优掩码
            for i in range(n_boxes):
                if assigned_masks[i] is None:
                    # 按成本排序
                    sorted_indices = np.argsort(cost_matrix[i])
                    
                    for j in sorted_indices:
                        if not mask_used[j] and cost_matrix[i, j] < 1.0:  # 更宽松的阈值
                            assigned_masks[i] = mobilesam_masks[j]
                            mask_used[j] = True
                            break
            
            # 4. 统计分配结果
            assigned_count = sum(1 for mask in assigned_masks if mask is not None)
            print(f"智能匹配完成: {assigned_count}/{n_boxes} 个框获得掩码")
            
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
            'total_time': 0.0,
            'box_prompts': 0,
            'point_prompts': 0,
            'auto_masks': 0,
        }


if __name__ == "__main__":
    print("MobileSAM引擎测试")
    print("=" * 60)
    
    # 测试代码
    try:
        engine = MobileSAMEngine(device="cpu")
        print("引擎测试通过")
    except Exception as e:
        print(f"引擎测试失败: {e}")