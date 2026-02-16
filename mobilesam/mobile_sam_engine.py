"""
MobileSAM Engine Optimized for Rock Grains
Filename: mobile_sam_engine.py
Function: MobileSAM engine designed for rock grain segmentation, solving issues with MobileSAM on rock grains
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

# MobileSAM import
try:
    from mobile_sam import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    MOBILESAM_AVAILABLE = True
except ImportError:
    MOBILESAM_AVAILABLE = False
    print(" MobileSAM library not installed, please run: pip install git+https://github.com/ChaoningZhang/MobileSAM.git")

# Segment Anything import (as fallback)
try:
    from segment_anything import sam_model_registry as sam_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


class MobileSAMEngine:
    """MobileSAM engine designed for rock grain segmentation"""
    
    def __init__(self, model_path: str = "models/mobile_sam.pt", 
                 device: str = "cuda", model_type: str = "vit_t"):
        """
        Initialize MobileSAM
        
        Args:
            model_path: MobileSAM model path
            device: Device to run on ('cpu' or 'cuda')
            model_type: Model type ('vit_t' for MobileSAM)
        """
        print("=" * 60)
        print("Initializing MobileSAM Ultimate Engine")
        print("=" * 60)
        
        self.device = device
        self.model_type = model_type
        
        # Check MobileSAM availability
        if not MOBILESAM_AVAILABLE:
            raise ImportError(
                "MobileSAM library not installed! Please run:\n"
                "pip install git+https://github.com/ChaoningZhang/MobileSAM.git"
            )
        
        # Auto-detect device (fallback to CPU if CUDA unavailable)
        if self.device == "cuda" and not torch.cuda.is_available():
            print(" CUDA unavailable, automatically switching to CPU")
            self.device = "cpu"
        
        # Load model
        try:
            print(f"Loading MobileSAM model: {model_path}")
            self.sam = sam_model_registry[model_type](checkpoint=model_path)
            self.sam.to(device=self.device)
            
            # Create predictor
            self.predictor = SamPredictor(self.sam)
            
            # Create automatic mask generator (for auto segmentation)
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
            
            print(f" MobileSAM model loaded successfully (device: {self.device})")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}\nPlease check if the model path is correct")
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")
        
        # === MobileSAM optimization parameters (for rock grains) ===
        self.params = {
            # Auto mask generation parameters
            'auto_mask': {
                'points_per_side': 32,
                'pred_iou_thresh': 0.88,
                'stability_score_thresh': 0.95,
                'box_nms_thresh': 0.7,
                'crop_n_layers': 0,
                'min_mask_region_area': 10,
            },
            # Box prompt parameters
            'box_prompt': {
                'box_expansion': 1.15,      # Box expansion factor
                'multimask_output': False,  # Single mask output
            },
            # Point prompt parameters
            'point_prompt': {
                'multimask_output': True,   # Multi-mask output
                'mask_threshold': 0.0,      # Mask threshold
            },
            # Mask filtering parameters
            'filter': {
                'min_area': 10,
                'max_area_ratio': 0.8,
                'min_solidity': 0.3,
                'min_extent': 0.1,
                'min_confidence': 0.3,
            },
            # Morphological parameters
            'morphology': {
                'small_kernel': (3, 3),
                'medium_kernel': (5, 5),
                'large_kernel': (7, 7),
                'closing_iterations': 1,
                'opening_iterations': 1,
                'dilation_iterations': 1,
            },
            # Multi-scale inference
            'multi_scale': {
                'enabled': True,
                'scales': [0.8, 1.0, 1.2],
                'merge_strategy': 'union',
            }
        }
        
        # Performance monitoring
        self.performance_stats = {
            'total_inferences': 0,
            'total_masks_generated': 0,
            'total_masks_filtered': 0,
            'total_time': 0.0,
            'box_prompts': 0,
            'point_prompts': 0,
            'auto_masks': 0,
        }
        
        print(" MobileSAM engine initialization complete")
        self.print_parameters()
    
    def print_parameters(self):
        """Print optimization parameters"""
        print("\n MobileSAM optimization parameters:")
        print("  - Auto mask generation: points_per_side={}, iou_thresh={}".format(
            self.params['auto_mask']['points_per_side'],
            self.params['auto_mask']['pred_iou_thresh']
        ))
        print("  - Box prompt: box_expansion={}, multimask={}".format(
            self.params['box_prompt']['box_expansion'],
            self.params['box_prompt']['multimask_output']
        ))
        print("  - Mask filtering: min_area={}px, min_confidence={}".format(
            self.params['filter']['min_area'],
            self.params['filter']['min_confidence']
        ))
        print("  - Multi-scale inference: {}".format(
            "enabled" if self.params['multi_scale']['enabled'] else "disabled"
        ))
    
    def set_image(self, image: np.ndarray):
        """Set current image (must be called first)"""
        self.predictor.set_image(image)
        self.current_image = image
        self.image_shape = image.shape[:2]
    
    def segment_with_box(self, box: List[float]) -> Tuple[np.ndarray, float]:
        """
        Segment using bounding box (main method)
        
        Args:
            box: [x1, y1, x2, y2] bounding box
            
        Returns:
            mask: segmentation mask (0/1)
            score: confidence score
        """
        start_time = time.time()
        
        try:
            # Expand bounding box
            expanded_box = self._expand_box(box, self.params['box_prompt']['box_expansion'])
            
            # Ensure box is within image bounds
            h, w = self.image_shape
            x1, y1, x2, y2 = expanded_box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return np.zeros(self.image_shape, dtype=np.uint8), 0.0
            
            # Convert to numpy array
            input_box = np.array([x1, y1, x2, y2])
            
            # Predict mask
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
            
            # Select best mask
            if self.params['box_prompt']['multimask_output']:
                best_idx = np.argmax(scores)
                mask = masks[best_idx]
                score = float(scores[best_idx])
            else:
                mask = masks[0]
                score = 0.8  # Use default value if no score available
            
            # Convert to binary mask
            binary_mask = (mask > 0).astype(np.uint8)
            
            # Filter mask
            if not self._filter_mask_by_properties(binary_mask):
                return np.zeros(self.image_shape, dtype=np.uint8), 0.0
            
            # Morphological enhancement
            enhanced_mask = self._enhance_mask_morphology(binary_mask)
            
            # Calculate mask quality
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
            print(f" Box segmentation failed: {e}")
            return np.zeros(self.image_shape, dtype=np.uint8), 0.0
    
    def segment_with_box_and_point(self, box: List[float], point: Tuple[float, float]) -> Tuple[np.ndarray, float]:
        """
        Segment using bounding box + center point (more stable)
        
        Args:
            box: [x1, y1, x2, y2] bounding box
            point: (x, y) center point
            
        Returns:
            mask: segmentation mask
            score: confidence score
        """
        try:
            # Expand bounding box
            expanded_box = self._expand_box(box, self.params['box_prompt']['box_expansion'])
            
            # Ensure box and point are within image bounds
            h, w = self.image_shape
            x1, y1, x2, y2 = expanded_box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            px, py = point
            px, py = max(0, px), max(0, py)
            px, py = min(w-1, px), min(h-1, py)
            
            if x2 <= x1 or y2 <= y1:
                return np.zeros(self.image_shape, dtype=np.uint8), 0.0
            
            # Prepare inputs
            input_box = np.array([x1, y1, x2, y2])
            input_point = np.array([[px, py]])
            input_label = np.array([1])  # Foreground point
            
            # Predict mask
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
            
            # Select best mask
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = float(scores[best_idx])
            
            # Convert to binary mask
            binary_mask = (mask > 0).astype(np.uint8)
            
            # Filter mask
            if not self._filter_mask_by_properties(binary_mask):
                return np.zeros(self.image_shape, dtype=np.uint8), 0.0
            
            # Morphological enhancement
            enhanced_mask = self._enhance_mask_morphology(binary_mask)
            
            # Calculate mask quality
            mask_quality = self._calculate_mask_quality(enhanced_mask)
            final_score = score * 0.7 + mask_quality * 0.3
            
            self.performance_stats['total_masks_generated'] += 1
            
            if final_score >= self.params['filter']['min_confidence']:
                self.performance_stats['total_masks_filtered'] += 1
                return enhanced_mask, final_score
            else:
                return np.zeros(self.image_shape, dtype=np.uint8), 0.0
            
        except Exception as e:
            print(f" Box+point segmentation failed: {e}")
            return np.zeros(self.image_shape, dtype=np.uint8), 0.0
    
    def generate_auto_masks(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        Generate masks automatically (no prompts needed)
        
        Args:
            image: input image
            
        Returns:
            masks: list of masks
            scores: list of scores
        """
        start_time = time.time()
        
        try:
            # Generate automatic masks
            anns = self.mask_generator.generate(image)
            
            self.performance_stats['auto_masks'] += 1
            self.performance_stats['total_inferences'] += 1
            
            masks = []
            scores = []
            
            for ann in anns:
                mask = ann['segmentation']
                score = ann.get('predicted_iou', 0.5) * ann.get('stability_score', 0.5)
                
                # Convert to binary mask
                binary_mask = mask.astype(np.uint8)
                
                # Filter mask
                if self._filter_mask_by_properties(binary_mask):
                    # Morphological enhancement
                    enhanced_mask = self._enhance_mask_morphology(binary_mask)
                    
                    # Calculate mask quality
                    mask_quality = self._calculate_mask_quality(enhanced_mask)
                    final_score = score * 0.7 + mask_quality * 0.3
                    
                    if final_score >= self.params['filter']['min_confidence']:
                        masks.append(enhanced_mask)
                        scores.append(final_score)
            
            inference_time = time.time() - start_time
            self.performance_stats['total_time'] += inference_time
            self.performance_stats['total_masks_generated'] += len(masks)
            self.performance_stats['total_masks_filtered'] += len(masks)
            
            print(f" Auto mask generation: {len(masks)} valid masks, time: {inference_time:.2f}s")
            return masks, scores
            
        except Exception as e:
            print(f" Auto mask generation failed: {e}")
            return [], []
    
    def _expand_box(self, box: List[float], expansion: float) -> List[float]:
        """Expand bounding box"""
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Calculate expansion amount
        dx = width * (expansion - 1) / 2
        dy = height * (expansion - 1) / 2
        
        return [x1 - dx, y1 - dy, x2 + dx, y2 + dy]
    
    def _filter_mask_by_properties(self, mask: np.ndarray) -> bool:
        """Smart mask filtering based on properties"""
        if mask.sum() == 0:
            return False
        
        # Calculate mask properties
        mask_area = mask.sum()
        img_area = mask.shape[0] * mask.shape[1]
        
        # 1. Area filtering
        if mask_area < self.params['filter']['min_area']:
            return False
        
        if mask_area / img_area > self.params['filter']['max_area_ratio']:
            return False
        
        # 2. Solidity filtering (area / convex hull area)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        
        # Take largest contour
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
        
        # 3. Extent filtering (mask area / bounding box area)
        x, y, w, h = cv2.boundingRect(main_contour)
        bbox_area = w * h
        
        if bbox_area == 0:
            extent = 0
        else:
            extent = area / bbox_area
        
        if extent < self.params['filter']['min_extent']:
            return False
        
        # 4. Shape filtering (exclude overly elongated masks)
        if h > 0:
            aspect_ratio = w / h
            if aspect_ratio > 5.0 or aspect_ratio < 0.2:
                return False
        
        return True
    
    def _enhance_mask_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Enhance mask using morphological operations"""
        # 1. Close operation to fill small holes
        kernel_close = np.ones(self.params['morphology']['medium_kernel'], np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, 
                               iterations=self.params['morphology']['closing_iterations'])
        
        # 2. Open operation to remove small noise
        kernel_open = np.ones(self.params['morphology']['small_kernel'], np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open,
                               iterations=self.params['morphology']['opening_iterations'])
        
        # 3. Slight dilation to enhance boundaries
        kernel_dilate = np.ones(self.params['morphology']['small_kernel'], np.uint8)
        mask = cv2.dilate(mask, kernel_dilate, 
                         iterations=self.params['morphology']['dilation_iterations'])
        
        # 4. Fill holes
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
        
        return mask
    
    def _calculate_mask_quality(self, mask: np.ndarray) -> float:
        """Calculate mask quality score (0-1)"""
        if mask.sum() == 0:
            return 0.0
        
        # 1. Compactness score (perimeter^2 / area)
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
        
        # 2. Solidity score
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            solidity_score = 0
        else:
            solidity = area / hull_area
            solidity_score = solidity
        
        # 3. Boundary smoothness score
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        smoothness = len(approx) / max(perimeter, 1)
        smoothness_score = min(smoothness * 10, 1.0)
        
        # Combined score
        total_score = compactness_score * 0.4 + solidity_score * 0.4 + smoothness_score * 0.2
        return float(total_score)
    
    def multi_scale_segmentation(self, image: np.ndarray, boxes: List[List[float]]) -> List[np.ndarray]:
        """
        Multi-scale segmentation (improves small grain detection)
        
        Args:
            image: input image
            boxes: list of bounding boxes
            
        Returns:
            list of masks
        """
        if not self.params['multi_scale']['enabled']:
            # Single-scale segmentation
            self.set_image(image)
            masks = []
            for box in boxes:
                mask, _ = self.segment_with_box(box)
                if mask.sum() > 0:
                    masks.append(mask)
            return masks
        
        # Multi-scale segmentation
        all_masks = []
        
        for scale in self.params['multi_scale']['scales']:
            # Resize image
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Scale bounding boxes
            scaled_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                scaled_box = [
                    int(x1 * scale), int(y1 * scale),
                    int(x2 * scale), int(y2 * scale)
                ]
                scaled_boxes.append(scaled_box)
            
            # Segment at current scale
            self.set_image(scaled_image)
            
            for i, box in enumerate(scaled_boxes):
                mask, _ = self.segment_with_box(box)
                
                if mask.sum() > 0:
                    # Resize mask back to original size
                    if scale != 1.0:
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        mask = (mask > 0.5).astype(np.uint8)
                    
                    all_masks.append(mask)
        
        # Merge multi-scale results
        if len(all_masks) > 0:
            merged_masks = self._merge_multi_scale_masks(all_masks)
            return merged_masks
        else:
            return []
    
    def _merge_multi_scale_masks(self, masks: List[np.ndarray]) -> List[np.ndarray]:
        """Merge multi-scale masks"""
        if len(masks) == 0:
            return []
        
        # Simple deduplication: merge highly overlapping masks
        merged_masks = []
        used = [False] * len(masks)
        
        for i in range(len(masks)):
            if used[i]:
                continue
            
            current_mask = masks[i]
            
            # Find overlapping masks
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
        Intelligent mask matching algorithm
        
        Uses many-to-many matching + quality priority + overlap penalty
        """
        if len(mobilesam_masks) == 0:
            return [None] * len(yolo_boxes)
        
        h, w = image_shape[:2]
        n_boxes = len(yolo_boxes)
        n_masks = len(mobilesam_masks)
        
        print(f"🔍 Intelligent matching: {n_boxes} boxes vs {n_masks} masks")
        
        # 1. Calculate cost matrix (lower is better)
        cost_matrix = np.zeros((n_boxes, n_masks))
        
        for i, box in enumerate(yolo_boxes):
            x1, y1, x2, y2 = map(int, box)
            box_area = max((x2 - x1) * (y2 - y1), 1)
            box_center_x = (x1 + x2) / 2
            box_center_y = (y1 + y2) / 2
            
            for j, mask in enumerate(mobilesam_masks):
                # Calculate mask portion inside box
                mask_in_box = mask[y1:y2, x1:x2]
                intersection = np.sum(mask_in_box > 0)
                
                if intersection == 0:
                    cost = 10.0  # High cost
                else:
                    # Calculate coverage
                    coverage = intersection / box_area
                    
                    # Calculate center offset
                    mask_indices = np.where(mask > 0)
                    if len(mask_indices[0]) > 0:
                        mask_center_y = np.mean(mask_indices[0])
                        mask_center_x = np.mean(mask_indices[1])
                    else:
                        mask_center_y, mask_center_x = h/2, w/2
                    
                    center_distance = np.sqrt((mask_center_x - box_center_x)**2 + 
                                             (mask_center_y - box_center_y)**2)
                    
                    # Normalize distance
                    norm_distance = center_distance / np.sqrt(h**2 + w**2)
                    
                    # Cost = low coverage + high distance + low quality
                    cost = (1.0 - coverage) * 0.4 + norm_distance * 0.3 + (1.0 - mask_scores[j]) * 0.3
                
                cost_matrix[i, j] = cost
        
        # 2. Use Hungarian algorithm for optimal assignment
        assigned_masks = [None] * n_boxes
        mask_used = [False] * n_masks
        
        if n_boxes > 0 and n_masks > 0:
            # Hungarian algorithm finds minimum cost assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Apply assignment (cost threshold)
            for i, j in zip(row_ind, col_ind):
                if i < n_boxes and j < n_masks and cost_matrix[i, j] < 0.7:  # 成本阈值
                    assigned_masks[i] = mobilesam_masks[j]
                    mask_used[j] = True
            
            # 3. Secondary assignment: find suboptimal masks for unassigned boxes
            for i in range(n_boxes):
                if assigned_masks[i] is None:
                    # 按成本排序
                    sorted_indices = np.argsort(cost_matrix[i])
                    
                    for j in sorted_indices:
                        if not mask_used[j] and cost_matrix[i, j] < 1.0:  # 更宽松的阈值
                            assigned_masks[i] = mobilesam_masks[j]
                            mask_used[j] = True
                            break
            
            # 4. Count assignment results
            assigned_count = sum(1 for mask in assigned_masks if mask is not None)
            print(f" Intelligent matching complete: {assigned_count}/{n_boxes} boxes got masks")
            
        return assigned_masks
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.performance_stats.copy()
        
        if stats['total_inferences'] > 0:
            stats['avg_time_per_inference'] = stats['total_time'] / stats['total_inferences']
        else:
            stats['avg_time_per_inference'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics"""
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
    print(" MobileSAM Engine Test")
    print("=" * 60)
    
    # Test code
    try:
        engine = MobileSAMEngine(device="cpu")
        print(" Engine test passed")
    except Exception as e:
        print(f" Engine test failed: {e}")