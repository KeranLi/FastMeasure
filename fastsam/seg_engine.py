"""
UltraFastSAM Ultimate Engine - Specially optimized for rock grains
Filename: seg_engine.py
Function: Solves all FastSAM issues in rock grain segmentation
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
    """Ultimate FastSAM engine, designed for rock grain segmentation"""
    
    def __init__(self, model_path: str = "../models/FastSAM-s.pt", device: str = "cpu"):
        """
        Initialize Ultimate FastSAM engine
        
        Args:
            model_path: FastSAM model path
            device: Device to run on ('cpu' or 'cuda')
        """
        print("=" * 60)
        print("🚀 Initializing UltraFastSAM Ultimate Engine...")
        print("=" * 60)
        
        self.device = device
        
        # Load model
        try:
            self.model = FastSAM(model_path)
            self.model.to(device=self.device)
            print(f"✅ FastSAM model loaded successfully (device: {device})")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
        
        # === Hyper-optimized parameters (specially for rock grains) ===
        self.params = {
            # Global inference parameters (for quickly obtaining candidate masks)
            'global': {
                'imgsz': 1024,      # Large size, preserve small grains
                'conf': 0.15,       # Low confidence, improve recall
                'iou': 0.3,         # Low IoU, avoid suppression
                'retina_masks': True
            },
            # Box interior inference parameters (for fine segmentation)
            'local': {
                'imgsz': 512,       # 中等尺寸
                'conf': 0.1,        # 极低置信度
                'iou': 0.2          # 极低IoU
            },
            # Small grain specific parameters
            'small': {
                'imgsz': 256,       # 小尺寸
                'conf': 0.05,       # 极低置信度
                'iou': 0.1          # 极低IoU
            },
            # Mask filtering parameters
            'filter': {
                'min_area': 10,     # Minimum area 10 pixels (rock grains are small)
                'max_area_ratio': 0.8,  # Maximum area ratio
                'min_solidity': 0.3,    # Minimum solidity
                'min_extent': 0.1       # Minimum extent
            },
            # Morphological parameters
            'morphology': {
                'small_kernel': (3, 3),
                'medium_kernel': (5, 5),
                'large_kernel': (7, 7),
                'closing_iterations': 1,
                'opening_iterations': 1
            }
        }
        
        # Performance monitoring
        self.performance_stats = {
            'total_inferences': 0,
            'total_masks_generated': 0,
            'total_masks_filtered': 0,
            'total_time': 0.0
        }
        
        print("✅ UltraFastSAM engine initialization complete")
        self.print_parameters()
    
    def print_parameters(self):
        """Print optimization parameters"""
        print("\n📊 UltraFastSAM optimization parameters:")
        print("  - Global inference: size={}, confidence={}, IoU={}".format(
            self.params['global']['imgsz'],
            self.params['global']['conf'],
            self.params['global']['iou']
        ))
        print("  - Local inference: size={}, confidence={}, IoU={}".format(
            self.params['local']['imgsz'],
            self.params['local']['conf'],
            self.params['local']['iou']
        ))
        print("  - Mask filtering: min_area={}px, min_solidity={}".format(
            self.params['filter']['min_area'],
            self.params['filter']['min_solidity']
        ))
    
    def inference_whole_image(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        Run UltraFastSAM inference on whole image (generate candidate masks)
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            masks: filtered list of masks
            scores: corresponding confidence scores
        """
        start_time = time.time()
        h, w = image.shape[:2]
        
        try:
            # Run UltraFastSAM inference (using optimized parameters)
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
                print("⚠️ Global inference did not detect any masks")
                return [], []
            
            # Get mask data
            masks_data = results[0].masks.data.cpu().numpy()
            scores = results[0].masks.conf.cpu().numpy() if hasattr(results[0].masks, 'conf') else None
            
            # Process masks (using smart filtering)
            processed_masks = []
            valid_scores = []
            
            for idx, mask in enumerate(masks_data):
                # Convert to binary mask
                binary_mask = (mask > 0).astype(np.uint8)
                
                # Calculate mask properties
                mask_area = np.sum(binary_mask)
                img_area = h * w
                
                # Filter mask (using smart filtering function)
                if self._filter_mask_by_properties(binary_mask, h, w):
                    # Morphological enhancement
                    enhanced_mask = self._enhance_mask_morphology(binary_mask)
                    
                    # Calculate mask quality score
                    mask_score = self._calculate_mask_quality(enhanced_mask)
                    
                    # If mask quality is too poor, try to repair
                    if mask_score < 0.3:
                        enhanced_mask = self._repair_mask(enhanced_mask)
                        mask_score = self._calculate_mask_quality(enhanced_mask)
                    
                    processed_masks.append(enhanced_mask * 255)
                    
                    # Use quality score or original score
                    if scores is not None and idx < len(scores):
                        final_score = scores[idx] * 0.7 + mask_score * 0.3
                    else:
                        final_score = mask_score
                    
                    valid_scores.append(float(final_score))
            
            inference_time = time.time() - start_time
            self.performance_stats['total_time'] += inference_time
            self.performance_stats['total_masks_generated'] += len(masks_data)
            self.performance_stats['total_masks_filtered'] += len(processed_masks)
            
            print(f"✅ Global inference generated {len(processed_masks)}/{len(masks_data)} valid masks, time: {inference_time:.2f}s")
            return processed_masks, valid_scores
            
        except Exception as e:
            print(f"❌ Global inference failed: {e}")
            import traceback
            traceback.print_exc()
            return [], []
    
    def _filter_mask_by_properties(self, mask: np.ndarray, img_h: int, img_w: int) -> bool:
        """
        Smart mask filtering based on properties
        """
        # Calculate mask properties
        mask_area = np.sum(mask > 0)
        img_area = img_h * img_w
        
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
        """
        Enhance mask using morphological operations
        """
        # 1. Close operation to fill small holes
        kernel_close = np.ones(self.params['morphology']['medium_kernel'], np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, 
                               iterations=self.params['morphology']['closing_iterations'])
        
        # 2. Open operation to remove small noise
        kernel_open = np.ones(self.params['morphology']['small_kernel'], np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open,
                               iterations=self.params['morphology']['opening_iterations'])
        
        # 3. Fill holes
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
        
        return mask
    
    def _calculate_mask_quality(self, mask: np.ndarray) -> float:
        """
        Calculate mask quality score (0-1)
        """
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
            # Normalization: ideal circle is 1, larger is less compact
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
    
    def _repair_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Repair low-quality mask
        """
        # 1. Calculate mask properties
        area = mask.sum()
        h, w = mask.shape
        
        # 2. If small mask, use dilation to enhance
        if area < 100:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        # 3. If large mask but poor shape, use erosion to remove burrs
        elif area > 1000:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        # 4. Ensure mask is connected
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels > 1:
            # Keep largest connected component
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_label).astype(np.uint8)
        
        return mask
    
    def segment_single_box(self, image: np.ndarray, box: List[float]) -> Tuple[np.ndarray, float]:
        """
        Fine segmentation for single box
        
        Args:
            image: original image
            box: [x1, y1, x2, y2]
            
        Returns:
            mask: segmentation mask
            score: quality score
        """
        x1, y1, x2, y2 = map(int, box)
        h, w = image.shape[:2]
        
        # Ensure bounding box is within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros((h, w), dtype=np.uint8), 0.0
        
        # Crop region
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((h, w), dtype=np.uint8), 0.0
        
        try:
            # Select parameters based on box size
            crop_h, crop_w = crop.shape[:2]
            box_area = crop_h * crop_w
            img_area = h * w
            
            # Smart parameter selection
            if box_area < 1000:  # Small box
                params = self.params['small']
            elif box_area < 10000:  # Medium box
                params = self.params['local']
            else:  # Large box
                params = self.params['global']
            
            # Run FastSAM inference
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
            
            # Get all masks
            masks_data = results[0].masks.data.cpu().numpy()
            
            if len(masks_data) == 0:
                return np.zeros((h, w), dtype=np.uint8), 0.0
            
            # Select best mask
            best_mask = None
            best_score = -1
            
            for mask in masks_data:
                binary_mask = (mask > 0).astype(np.uint8)
                
                # Enhance mask
                enhanced_mask = self._enhance_mask_morphology(binary_mask)
                
                # Calculate quality score
                mask_score = self._calculate_mask_quality(enhanced_mask)
                
                if mask_score > best_score:
                    best_score = mask_score
                    best_mask = enhanced_mask
            
            if best_mask is not None and best_score > 0.1:
                # Create full image mask
                full_mask = np.zeros((h, w), dtype=np.uint8)
                mask_h, mask_w = best_mask.shape
                
                if mask_h > 0 and mask_w > 0:
                    # Ensure mask is within crop region
                    actual_h = min(mask_h, y2 - y1)
                    actual_w = min(mask_w, x2 - x1)
                    
                    full_mask[y1:y1+actual_h, x1:x1+actual_w] = best_mask[:actual_h, :actual_w]
                
                return full_mask * 255, best_score
            
        except Exception as e:
            print(f"⚠️ Single box segmentation failed: {e}")
        
        return np.zeros((h, w), dtype=np.uint8), 0.0
    
    def intelligent_matching(self, 
                           yolo_boxes: List[List[float]], 
                           fastsam_masks: List[np.ndarray],
                           mask_scores: List[float],
                           image_shape: Tuple[int, int]) -> List[Optional[np.ndarray]]:
        """
        Intelligent mask matching algorithm (improved)
        
        Uses many-to-many matching + quality priority + overlap penalty
        """
        if len(fastsam_masks) == 0:
            return [None] * len(yolo_boxes)
        
        h, w = image_shape[:2]
        n_boxes = len(yolo_boxes)
        n_masks = len(fastsam_masks)
        
        print(f"📊 Intelligent matching: {n_boxes} boxes vs {n_masks} masks")
        
        # 1. Calculate cost matrix (lower is better)
        cost_matrix = np.zeros((n_boxes, n_masks))
        
        for i, box in enumerate(yolo_boxes):
            x1, y1, x2, y2 = map(int, box)
            box_area = max((x2 - x1) * (y2 - y1), 1)
            
            for j, mask in enumerate(fastsam_masks):
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
                    
                    box_center_y = (y1 + y2) / 2
                    box_center_x = (x1 + x2) / 2
                    
                    center_distance = np.sqrt((mask_center_x - box_center_x)**2 + 
                                             (mask_center_y - box_center_y)**2)
                    
                    # 归一化距离
                    norm_distance = center_distance / np.sqrt(h**2 + w**2)
                    
                    # Cost = low coverage + high distance + low quality
                    cost = (1.0 - coverage) * 0.5 + norm_distance * 0.3 + (1.0 - mask_scores[j]) * 0.2
                
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
                    assigned_masks[i] = fastsam_masks[j]
                    mask_used[j] = True
            
            # 3. Secondary assignment: find suboptimal masks for unassigned boxes
            for i in range(n_boxes):
                if assigned_masks[i] is None:
                    # 按成本排序
                    sorted_indices = np.argsort(cost_matrix[i])
                    
                    for j in sorted_indices:
                        if not mask_used[j] and cost_matrix[i, j] < 1.0:  # 更宽松的阈值
                            assigned_masks[i] = fastsam_masks[j]
                            mask_used[j] = True
                            break
            
            # 4. Count assignment results
            assigned_count = sum(1 for mask in assigned_masks if mask is not None)
            print(f"✅ Intelligent matching complete: {assigned_count}/{n_boxes} boxes got masks")
            
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
            'total_time': 0.0
        }


if __name__ == "__main__":
    print("UltraFastSAM Engine Test")
    print("=" * 60)
    
    # Test code
    engine = UltraFastSAM(device="cpu")
    print("✅ Engine test passed")