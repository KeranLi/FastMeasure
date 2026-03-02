"""
FastSAM Enhanced Interactive Interface Module - Fixed Version
File: fastsam_interactive.py
Function: Interactive segmentation based on FastSAM features, fixing point selection, lag, and offset issues
"""

import os
import sys
import numpy as np
import matplotlib

def setup_backend():
    """Smart backend setup, prioritize GUI backends"""
    try:
        import tkinter
        matplotlib.use('TkAgg')
        return 'TkAgg'
    except ImportError:
        pass
    
    if os.getenv('DISPLAY') and not os.getenv('SSH_CONNECTION'):
        for backend in ['Qt5Agg', 'WXAgg']:
            try:
                matplotlib.use(backend)
                return backend
            except:
                continue
    
    matplotlib.use('Agg')
    return 'Agg'

backend = setup_backend()

import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from PIL import Image
import pandas as pd
from shapely.geometry import Polygon as ShapelyPolygon, Point, box
import json
from skimage import measure
import traceback
import time
import threading
import cv2
from typing import List, Dict, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import FastSAM
try:
    from ultralytics import FastSAM
    FASTSAM_AVAILABLE = True
    print("Using Ultralytics FastSAM")
except ImportError as e:
    FASTSAM_AVAILABLE = False
    print(f"FastSAM library not installed: {e}")

# Force import project one functions
print("=" * 60)
print("Config: Force using project one functions")
print("=" * 60)

import sys
from pathlib import Path

# Get project root path
project_root = Path(__file__).parent

# Import core segmentation functions (migrated from segmenteverygrain)
try:
    from core.segment_core import (
        create_labeled_image,
        plot_image_w_colorful_grains,
        plot_grain_axes_and_centroids,
        find_connected_components,
        merge_overlapping_polygons
    )
    PROJECT1_AVAILABLE = True
    print("Successfully imported core segmentation functions")
except ImportError as e:
    print(f"Failed to import core segmentation functions: {e}")
    print("Program requires core segmentation functions to run")
    sys.exit(1)

# Import geometry calculation modules
try:
    geometry_dir = project_root / "geometry"
    if str(geometry_dir) not in sys.path:
        sys.path.insert(0, str(geometry_dir))
    
    from geometry.grain_metric import GrainShapeMetrics
    from geometry.config_loader import load_geometry_config
    from geometry.export_csv import select_columns_for_grain_statistics_csv
    
    GEOMETRY_AVAILABLE = True
    print("geometry module loaded successfully")
except ImportError as e:
    GEOMETRY_AVAILABLE = False
    print(f"geometry module unavailable: {e}")

try:
    from core.scale_calibration import InteractiveScaleCalibrator, quick_scale_calibration
    SCALE_CALIBRATION_AVAILABLE = True
    print("Scale calibration module loaded successfully")
except ImportError as e:
    SCALE_CALIBRATION_AVAILABLE = False
    print(f"Scale calibration module unavailable: {e}")

print("=" * 60)


class PureFastSAMInteractiveEnhanced:
    """Enhanced Interactive FastSAM - Fixed Version"""
    
    def __init__(self, model_path: str = "models/FastSAM-s.pt", 
                 device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        
        self.image = None
        self.image_path = None
        self.fastsam_model = None
        self.model_loaded = False
        
        # Full image inference result cache
        self.global_results = None
        self.all_masks_cache = []
        self.all_masks_scores = []
        
        # Interactive state
        self.grains = []
        self.current_grain_id = 0
        self.drawing_box = False
        self.box_start = None
        self.box_end = None
        self.current_box = None
        
        # Optimization: reduce redraw
        self.last_draw_time = 0
        self.draw_interval = 0.05  # 50ms，Avoid frequent redraw
        
        # Result storage
        self.polygons = []
        self.labels = None
        self.mask_all = None
        self.grain_data = None
        
        # Display related
        self.fig = None
        self.ax = None
        self.grain_patches = {}
        self.box_artist = None
        self.grain_texts = {}
        
        # Output directory
        # Unified output directory: results/fastsam/interactive/
        self.output_dir = Path("results") / "fastsam" / "interactive"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Geometry configuration
        self.geometry_config = None
        if GEOMETRY_AVAILABLE:
            try:
                # Modification: adjust config file path
                config_path = Path(__file__).parent / "configs" / "geometry.yaml"
                if config_path.exists():
                    self.geometry_config = load_geometry_config(str(config_path))
                    print("Geometry configuration loaded successfully")
                else:
                    # Try to find in fastsam subdirectory
                    alt_config_path = Path(__file__).parent / "configs" / "geometry.yaml"
                    if alt_config_path.exists():
                        self.geometry_config = load_geometry_config(str(alt_config_path))
                        print("Successfully loaded geometry configuration from fastsam subdirectory")
            except Exception as e:
                print(f"Failed to load geometry configuration: {e}")
        
        # Performance statistics
        self.start_time = None
        self.total_grains = 0
        self.total_interactions = 0
        
        self.gui_running = False
        
        # Scale calibration
        self.scale_calibrator = None
        self.is_scale_calibration_mode = False
        self.scale_factor = None
        if SCALE_CALIBRATION_AVAILABLE:
            self.scale_calibrator = InteractiveScaleCalibrator()
        
        print("=" * 70)
        print("FastSAM Interactive System (Fixed Version)")
        print("=" * 70)
        
        self._load_fastsam_model()
    
    def _load_fastsam_model(self) -> bool:
        """Load FastSAM model"""
        if not FASTSAM_AVAILABLE:
            print("FastSAM library not available")
            return False
        
        try:
            print(f"Loading FastSAM model: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                print(f"Model file does not exist: {self.model_path}")
                return False
            
            # Load FastSAM model
            self.fastsam_model = FastSAM(self.model_path)
            self.fastsam_model.to(self.device)
            self.model_loaded = True
            
            print(f"FastSAM model loaded successfully (device: {self.device})")
            return True
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            traceback.print_exc()
            return False
    
    def _run_global_inference(self):
        """Run full image inference and cache results"""
        if self.image is None or not self.model_loaded:
            return False
        
        try:
            print("Running full image inference...")
            start_time = time.time()
            
            # RunFastSAMFull image inference
            results = self.fastsam_model(
                self.image,
                device=self.device,
                imgsz=1024,
                conf=0.25,
                iou=0.3,
                verbose=False
            )
            
            if len(results) == 0 or results[0].masks is None:
                print("Full image inference did not generate masks")
                return False
            
            # Cache results
            self.global_results = results[0]
            masks_data = results[0].masks.data.cpu().numpy()
            
            # Process and cache all masks
            self.all_masks_cache = []
            self.all_masks_scores = []
            
            h, w = self.image.shape[:2]
            
            for idx, mask in enumerate(masks_data):
                binary_mask = (mask > 0).astype(np.uint8)
                
                # Filter small masks
                if np.sum(binary_mask) < 10:
                    continue
                
                # Ensure mask size is correct
                if binary_mask.shape[0] != h or binary_mask.shape[1] != w:
                    binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # Morphological enhancement
                enhanced_mask = self._enhance_mask_morphology(binary_mask)
                
                # Calculate quality score
                score = self._calculate_mask_quality(enhanced_mask)
                
                self.all_masks_cache.append(enhanced_mask)
                self.all_masks_scores.append(score)
            
            inference_time = time.time() - start_time
            print(f"Full image inference completed: {len(self.all_masks_cache)} candidate masks, time: {inference_time:.2f}s")
            return True
            
        except Exception as e:
            print(f"Full image inference failed: {e}")
            return False
    
    def _enhance_mask_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Enhance mask using morphological operations"""
        # Close operation to fill small holes
        kernel_close = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        
        # Open operation to remove small noise
        kernel_open = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # Fill holes
        from scipy import ndimage
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
        
        return mask
    
    def _calculate_mask_quality(self, mask: np.ndarray) -> float:
        """Calculate mask quality score (0-1)"""
        if mask.sum() == 0:
            return 0.0
        
        # Calculate solidity
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        
        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)
        
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            return 0.0
        
        solidity = area / hull_area
        return float(solidity)
    
    def set_image(self, image: np.ndarray):
        """Set current image andRunFull image inference"""
        self.image = image
        print(f"Image set: {image.shape}")
        
        # Run full image inference
        self._run_global_inference()
    
    def _safe_file_dialog(self):
        """Safe file selection dialog (macOS compatible)"""
        self.selected_file = None
        
        try:
            # For macOS, run in main thread to avoid NSWindow threading issues
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            file_path = filedialog.askopenfilename(
                title="Select Rock Microscopic Image",
                filetypes=[
                    ("Image Files", "*.tif *.tiff *.jpg *.jpeg *.png *.bmp"),
                    ("All Files", "*.*")
                ]
            )
            
            if file_path:
                self.selected_file = file_path
            
            root.destroy()
        except Exception as e:
            print(f"File dialog error: {e}")
        
        return self.selected_file
    
    def load_image_with_gui(self) -> bool:
        """Load image through GUI file dialog"""
        if not self.model_loaded or self.fastsam_model is None:
            print("Model not loaded, cannot process image")
            return False
        
        try:
            print("Please select rock microscopic image...")
            
            file_path = self._safe_file_dialog()
            
            if not file_path:
                print("No file selected, exiting interactive mode")
                return False
            
            if not os.path.exists(file_path):
                print(f"Image file does not exist: {file_path}")
                return False
            
            print(f"Loading image: {file_path}")
            pil_image = Image.open(file_path).convert('RGB')
            self.image = np.array(pil_image)
            self.image_path = file_path
            
            self.set_image(self.image)
            
            # Reset state
            self.grains = []
            self.current_grain_id = 0
            self.grain_patches = {}
            self.grain_texts = {}
            self.polygons = []
            self.labels = None
            self.mask_all = None
            self.grain_data = None
            
            self.start_time = time.time()
            
            print(f"Image loaded successfully: {self.image.shape}")
            return True
            
        except Exception as e:
            print(f"Image loading failed: {e}")
            traceback.print_exc()
            return False
    
    def load_image_from_path(self, image_path: str) -> bool:
        """Load image directly from path"""
        if not self.model_loaded or self.fastsam_model is None:
            print("Model not loaded, cannot process image")
            return False
        
        try:
            if not os.path.exists(image_path):
                print(f"Image file does not exist: {image_path}")
                return False
            
            print(f"Loading image: {image_path}")
            pil_image = Image.open(image_path).convert('RGB')
            self.image = np.array(pil_image)
            self.image_path = image_path
            
            self.set_image(self.image)
            
            # Reset state
            self.grains = []
            self.current_grain_id = 0
            self.grain_patches = {}
            self.grain_texts = {}
            self.polygons = []
            self.labels = None
            self.mask_all = None
            self.grain_data = None
            
            self.start_time = time.time()
            
            print(f"Image loaded successfully: {self.image.shape}")
            return True
            
        except Exception as e:
            print(f"Image loading failed: {e}")
            traceback.print_exc()
            return False
    
    def _find_mask_at_point(self, x: float, y: float) -> Optional[Dict]:
        """Find mask at clicked position - Fixed version"""
        ix, iy = int(x), int(y)
        h, w = self.image.shape[:2]
        
        if not (0 <= ix < w and 0 <= iy < h):
            return None
        
        # Search from cached masks
        for idx, mask in enumerate(self.all_masks_cache):
            # Ensure mask size correct
            if mask.shape[0] != h or mask.shape[1] != w:
                continue
                
            if mask[iy, ix] > 0:
                return {
                    'mask': mask,
                    'score': self.all_masks_scores[idx] if idx < len(self.all_masks_scores) else 0.5,
                    'index': idx
                }
        
        # If not found，Try small range search around point
        search_radius = 5
        for idx, mask in enumerate(self.all_masks_cache):
            # Check area around point
            x_min = max(0, ix - search_radius)
            x_max = min(w, ix + search_radius)
            y_min = max(0, iy - search_radius)
            y_max = min(h, iy + search_radius)
            
            if np.any(mask[y_min:y_max, x_min:x_max] > 0):
                return {
                    'mask': mask,
                    'score': self.all_masks_scores[idx] if idx < len(self.all_masks_scores) else 0.5,
                    'index': idx
                }
        
        return None
    
    def _find_masks_in_box(self, box: Tuple) -> List[Dict]:
        """Find all masks in box"""
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        masks_in_box = []
        
        for idx, mask in enumerate(self.all_masks_cache):
            # Calculate mask area in box
            mask_in_box = mask[y1:y2, x1:x2]
            intersection = np.sum(mask_in_box > 0)
            
            if intersection > 0:
                # Calculate coverage
                box_area = (x2 - x1) * (y2 - y1)
                coverage = intersection / box_area if box_area > 0 else 0
                
                # Calculate mask center
                mask_indices = np.where(mask > 0)
                if len(mask_indices[0]) > 0:
                    mask_center_y = np.mean(mask_indices[0])
                    mask_center_x = np.mean(mask_indices[1])
                    
                    # Calculate center distance
                    box_center_x = (x1 + x2) / 2
                    box_center_y = (y1 + y2) / 2
                    center_distance = np.sqrt((mask_center_x - box_center_x)**2 + 
                                             (mask_center_y - box_center_y)**2)
                else:
                    center_distance = 1000
                
                masks_in_box.append({
                    'mask': mask,
                    'score': self.all_masks_scores[idx] if idx < len(self.all_masks_scores) else 0.5,
                    'coverage': coverage,
                    'center_distance': center_distance,
                    'index': idx
                })
        
        # Sort by coverage and center distance
        masks_in_box.sort(key=lambda x: (x['coverage'], -x['center_distance']), reverse=True)
        return masks_in_box
    
    def _run_local_inference(self, box: Tuple) -> List[Dict]:
        """Run inference on local region - fixed version (solves offset issue)"""
        x1, y1, x2, y2 = map(int, box)
        h, w = self.image.shape[:2]
        
        # Ensure bounding box is within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return []
        
        # Expand bounding box to ensure complete grain is included
        expand_pixels = 20
        x1_exp = max(0, x1 - expand_pixels)
        y1_exp = max(0, y1 - expand_pixels)
        x2_exp = min(w, x2 + expand_pixels)
        y2_exp = min(h, y2 + expand_pixels)
        
        # Crop region
        crop = self.image[y1_exp:y2_exp, x1_exp:x2_exp]
        if crop.size == 0:
            return []
        
        try:
            # Choose appropriate inference size based on crop size
            crop_h, crop_w = crop.shape[:2]
            if crop_h * crop_w < 10000:  # Small region
                imgsz = 256
            elif crop_h * crop_w < 40000:  # Medium region
                imgsz = 512
            else:  # Large region
                imgsz = 1024
            
            # Run local inference
            results = self.fastsam_model(
                crop,
                device=self.device,
                imgsz=imgsz,
                conf=0.1,
                iou=0.2,
                verbose=False
            )
            
            if len(results) == 0 or results[0].masks is None:
                return []
            
            masks_data = results[0].masks.data.cpu().numpy()
            local_masks = []
            
            for mask in masks_data:
                binary_mask = (mask > 0).astype(np.uint8)
                
                if np.sum(binary_mask) < 10:
                    continue
                
                # Resize mask to match crop region
                if binary_mask.shape[0] != crop_h or binary_mask.shape[1] != crop_w:
                    binary_mask = cv2.resize(binary_mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                
                # Calculate mask position in original image
                mask_h, mask_w = binary_mask.shape
                
                # Find mask bounding box
                rows = np.any(binary_mask, axis=1)
                cols = np.any(binary_mask, axis=0)
                
                if np.any(rows) and np.any(cols):
                    y_min_local, y_max_local = np.where(rows)[0][[0, -1]]
                    x_min_local, x_max_local = np.where(cols)[0][[0, -1]]
                    
                    # Convert to original image coordinates
                    x_min_global = x1_exp + x_min_local
                    y_min_global = y1_exp + y_min_local
                    x_max_global = x1_exp + x_max_local
                    y_max_global = y1_exp + y_max_local
                    
                    # Ensure mask has sufficient coverage in original box
                    overlap_x = max(0, min(x_max_global, x2) - max(x_min_global, x1))
                    overlap_y = max(0, min(y_max_global, y2) - max(y_min_global, y1))
                    overlap_area = overlap_x * overlap_y
                    original_box_area = (x2 - x1) * (y2 - y1)
                    
                    if original_box_area > 0 and overlap_area / original_box_area < 0.3:
                        continue  # Skip masks with too little overlap with original box
                
                # Create full image mask
                full_mask = np.zeros((h, w), dtype=np.uint8)
                full_mask[y1_exp:y1_exp+mask_h, x1_exp:x1_exp+mask_w] = binary_mask
                
                # Enhance mask
                enhanced_mask = self._enhance_mask_morphology(full_mask)
                score = self._calculate_mask_quality(enhanced_mask)
                
                local_masks.append({
                    'mask': enhanced_mask,
                    'score': score,
                    'box': (x1_exp, y1_exp, x1_exp+mask_w, y1_exp+mask_h)
                })
            
            return local_masks
            
        except Exception as e:
            print(f"Local inference failed: {e}")
            return []
    
    def _create_grain_from_mask(self, mask_data: Dict) -> int:
        """Create grain from mask"""
        self.current_grain_id += 1
        
        new_grain = {
            'id': self.current_grain_id,
            'mask': mask_data['mask'],
            'score': mask_data.get('score', 0.5),
            'color': np.random.rand(3,),
            'bbox': None
        }
        
        # Calculate bounding box
        if mask_data['mask'] is not None and np.any(mask_data['mask']):
            rows = np.any(mask_data['mask'], axis=1)
            cols = np.any(mask_data['mask'], axis=0)
            
            if np.any(rows) and np.any(cols):
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                new_grain['bbox'] = (xmin, ymin, xmax, ymax)
        
        self.grains.append(new_grain)
        self.total_grains += 1
        
        print(f"Creating new grain #{self.current_grain_id}, quality: {new_grain['score']:.3f}")
        
        return self.current_grain_id
    
    def _draw_grain_with_text(self, grain):
        """Draw single grain and its text label"""
        try:
            grain_id = grain['id']
            mask = grain['mask']
            
            if mask is None or not np.any(mask):
                return
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Simplify contour
                epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                if len(approx) >= 3:
                    # Extract coordinates
                    sx = approx[:, 0, 0]
                    sy = approx[:, 0, 1]
                    
                    # Draw filled polygon
                    patch = self.ax.fill(sx, sy, 
                                       facecolor=grain['color'], 
                                       edgecolor='black',
                                       alpha=0.4, 
                                       linewidth=1.5)
                    self.grain_patches[grain_id] = patch[0]
                    
                    # Add text label
                    if grain['bbox']:
                        xmin, ymin, xmax, ymax = grain['bbox']
                        center_x = (xmin + xmax) / 2
                        center_y = (ymin + ymax) / 2
                        
                        text_obj = self.ax.text(center_x, center_y, str(grain_id),
                                              fontsize=10, fontweight='bold',
                                              color='white',
                                              ha='center', va='center',
                                              bbox=dict(boxstyle='round,pad=0.3',
                                                      facecolor=grain['color'],
                                                      edgecolor='black',
                                                      alpha=0.8))
                        
                        self.grain_texts[grain_id] = text_obj
        
        except Exception as e:
            print(f"Failed to draw grain #{grain.get('id', 'unknown')}: {e}")
    
    def _refresh_grain_display(self):
        """Refresh display of all grains"""
        try:
            # Clear existing display
            for patch in self.grain_patches.values():
                patch.remove()
            self.grain_patches.clear()
            
            for text in self.grain_texts.values():
                text.remove()
            self.grain_texts.clear()
            
            # Redraw all grains
            for grain in self.grains:
                if grain['mask'] is not None:
                    self._draw_grain_with_text(grain)
            
            self.fig.canvas.draw()
            print(f"Display refreshed, current grain count: {len(self.grains)}")
        
        except Exception as e:
            print(f"Failed to refresh display: {e}")
    
    def _on_mouse_press(self, event):
        """Handle mouse press events"""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        
        # Check if in scale calibration mode
        if self.is_scale_calibration_mode and self.scale_calibrator:
            calibration_complete = self.scale_calibrator.on_click(event)
            if calibration_complete:
                # Calibration finished, get result
                scale_factor = self.scale_calibrator.get_result()
                if scale_factor:
                    self.scale_factor = scale_factor
                    print(f"Scale calibration complete! Factor: {scale_factor:.4f} um/px")
                self.is_scale_calibration_mode = False
                # Restore title
                image_name = Path(self.image_path).name if self.image_path else "Unnamed"
                self.ax.set_title(f"FastSAM Interactive - {image_name}", fontsize=16)
                self.fig.canvas.draw()
            return
        
        current_time = time.time()
        if current_time - self.last_draw_time < self.draw_interval:
            return  # Avoid too fast response
        
        if event.button == 1:  # Left button: start drawing box or point selection
            self.drawing_box = True
            self.box_start = (event.xdata, event.ydata)
            self.box_end = (event.xdata, event.ydata)
            
            # Clear previous box
            if self.box_artist:
                self.box_artist.remove()
                self.box_artist = None
            
            self.last_draw_time = current_time
    
    def _on_mouse_move(self, event):
        """Mouse move event - optimized (reduce redraw)"""
        if not self.drawing_box or event.inaxes != self.ax:
            return
        
        current_time = time.time()
        if current_time - self.last_draw_time < self.draw_interval:
            return  # Limit redraw frequency
        
        if event.xdata is not None and event.ydata is not None:
            self.box_end = (event.xdata, event.ydata)
            self._draw_current_box()
            self.last_draw_time = current_time
    
    def _on_mouse_release(self, event):
        """Mouse release event"""
        if not self.drawing_box or event.inaxes != self.ax:
            return
        
        if event.button == 1:  # Left button release
            self.drawing_box = False
            
            if self.box_start and self.box_end:
                # Calculate box coordinates
                x1, y1 = self.box_start
                x2, y2 = self.box_end
                
                # Ensure coordinates are correct
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                box = (x1, y1, x2, y2)
                box_area = (x2 - x1) * (y2 - y1)
                
                if box_area < 100:  # If box is too small，Treat as point selection
                    print(f"Point select: ({x1:.1f}, {y1:.1f})")
                    self._handle_point_click(x1, y1)
                else:
                    print(f"Box select: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                    self._handle_box_selection(box)
            
            # Clear box display
            if self.box_artist:
                self.box_artist.remove()
                self.box_artist = None
                self.fig.canvas.draw()
            
            self.box_start = None
            self.box_end = None
    
    def _draw_current_box(self):
        """Draw current box - optimized"""
        if not self.box_start or not self.box_end:
            return
        
        # Clear previous box
        if self.box_artist:
            self.box_artist.remove()
        
        # Draw new box
        x1, y1 = self.box_start
        x2, y2 = self.box_end
        
        rect = plt.Rectangle((min(x1, x2), min(y1, y2)), 
                           abs(x2 - x1), abs(y2 - y1),
                           fill=False, edgecolor='cyan', 
                           linewidth=2, linestyle='--', alpha=0.7)
        
        self.box_artist = self.ax.add_patch(rect)
        
        # Only update box part instead of entire figure
        self.fig.canvas.draw_idle()
    
    def _handle_point_click(self, x: float, y: float):
        """Handle point click event - fixed version"""
        # Find mask at click position
        mask_data = self._find_mask_at_point(x, y)
        
        if mask_data:
            # Check if this mask is already selected
            for grain in self.grains:
                if np.array_equal(grain['mask'], mask_data['mask']):
                    print(f"Grain #{grain['id']} already selected")
                    return
            
            # Create new grain
            grain_id = self._create_grain_from_mask(mask_data)
            self._refresh_grain_display()
        else:
            # Try local inference around the point
            print("Mask not found, running local inference...")
            box_size = 50
            box = (x - box_size, y - box_size, x + box_size, y + box_size)
            local_masks = self._run_local_inference(box)
            
            if local_masks:
                # Select highest quality mask
                best_local_mask = max(local_masks, key=lambda x: x['score'])
                grain_id = self._create_grain_from_mask(best_local_mask)
                self._refresh_grain_display()
                print(f"Local inference successful, generated new mask")
            else:
                print("No mask found at this location")
    
    def _handle_box_selection(self, box: Tuple):
        """Handle box selection event - fixed version"""
        # Find masks in box
        masks_in_box = self._find_masks_in_box(box)
        
        if masks_in_box:
            # Select best mask (highest coverage, closest center)
            best_mask = masks_in_box[0]
            
            # Check if mask already selected
            for grain in self.grains:
                if np.array_equal(grain['mask'], best_mask['mask']):
                    print(f"grain #{grain['id']} already selected")
                    return
            
            # Create new grain
            grain_id = self._create_grain_from_mask(best_mask)
            self._refresh_grain_display()
            
            print(f"Selected best mask in box, coverage: {best_mask['coverage']:.3f}")
        else:
            # If no mask found, run local inference
            print("No mask found in box, running local inference...")
            local_masks = self._run_local_inference(box)
            
            if local_masks:
                # Select highest quality mask
                best_local_mask = max(local_masks, key=lambda x: x['score'])
                grain_id = self._create_grain_from_mask(best_local_mask)
                self._refresh_grain_display()
                print(f"Local inference successful, generated new mask")
            else:
                print("Local inference also did not generate mask")
    
    def _start_scale_calibration(self):
        """Start scale calibration mode"""
        if not SCALE_CALIBRATION_AVAILABLE:
            messagebox.showerror("Error", "Scale calibration module not available")
            return
            
        if self.is_scale_calibration_mode:
            print("Already in scale calibration mode")
            return
        
        self.is_scale_calibration_mode = True
        
        print("\n" + "="*60)
        print("SCALE CALIBRATION MODE")
        print("="*60)
        print("1. Click the START point of a known-length line")
        print("2. Click the END point of the line")
        print("3. Enter the actual length in microns when prompted")
        print("Press 'Escape' to cancel")
        print("="*60)
        
        self.scale_calibrator.calibrate_scale(self.image, self.ax, self.fig)
        
        # Update title
        image_name = Path(self.image_path).name if self.image_path else "Unnamed"
        self.ax.set_title(f"FastSAM Interactive - {image_name} [SCALE CALIBRATION - Click two points]", 
                         fontsize=14, color='red', fontweight='bold')
        self.fig.canvas.draw()
    
    def _on_key_press(self, event):
        """Handle keyboard press events"""
        if event.key == 'x':  # Delete last grain
            self._delete_last_grain()
        elif event.key == 'd':  # Delete all grains
            self._delete_all_grains()
        elif event.key == 's':  # Save results
            self._show_save_options()
        elif event.key == 'r':  # Reset
            self._reset_interface()
        elif event.key == 'q':  # Quit
            print("Exiting interactive interface")
            self.gui_running = False
            plt.close(self.fig)
        elif event.key == 'h':  # Show help
            self._show_help()
        elif event.key == 'm':  # Manual scale calibration
            self._start_scale_calibration()
        elif event.key == 'S':  # Shift+S: Quick save complete results
            print("Quick saving complete results...")
            self._generate_complete_outputs()
    
    def _delete_last_grain(self):
        """Delete last grain"""
        if self.grains:
            last_grain = self.grains[-1]
            grain_id = last_grain['id']
            
            if grain_id in self.grain_patches:
                self.grain_patches[grain_id].remove()
                del self.grain_patches[grain_id]
            
            if grain_id in self.grain_texts:
                self.grain_texts[grain_id].remove()
                del self.grain_texts[grain_id]
            
            self.grains.pop()
            
            if self.grains:
                max_id = max(grain['id'] for grain in self.grains)
                self.current_grain_id = max_id
            else:
                self.current_grain_id = 0
            
            self.fig.canvas.draw()
            print(f"Deleted grain #{grain_id}")
    
    def _delete_all_grains(self):
        """Delete all grains"""
        for patch in self.grain_patches.values():
            patch.remove()
        self.grain_patches.clear()
        
        for text in self.grain_texts.values():
            text.remove()
        self.grain_texts.clear()
        
        self.grains = []
        self.current_grain_id = 0
        
        self.fig.canvas.draw()
        print("All grains deleted")
    
    def _show_help(self):
        """Show help information"""
        help_text = (
            "FastSAM Interactive Segmentation Guide:\n\n"
            "Mouse Actions:\n"
            "• Left drag: Draw selection box\n"
            "• Left click: Point select grain (small area)\n\n"
            "Keyboard Shortcuts:\n"
            "• 's': Show save options\n"
            "• 'S' (Shift+s): Quick save complete results\n"
            "• 'x': Delete last grain\n"
            "• 'd': Delete all grains\n"
            "• 'r': Reset interface\n"
            "• 'm': Manual scale calibration\n"
            "• 'q': Quit program\n"
            "• 'h': Show this help\n"
        )
        
        messagebox.showinfo("FastSAM Interactive Help", help_text)
    
    def _show_help_text_fixed(self):
        """Show help text on interface"""
        help_text = (
            "FastSAM Interactive Guide:\n"
            "• Left drag: Draw selection box\n"
            "• Left click: Point select grain\n"
            "• 's' key: Show save options\n"
            "• 'S' key (Shift+s): Quick save\n"
            "• 'x' key: Delete last grain\n"
            "• 'd' key: Delete all grains\n"
            "• 'r' key: Reset interface\n"
            "• 'm' key: Manual scale calibration\n"
            "• 'q' key: Exit program\n"
        )
        
        try:
            plt.figtext(
                0.02, 0.98, help_text, 
                fontsize=11, 
                fontproperties='Microsoft YaHei',
                verticalalignment='top',
                bbox=dict(
                    boxstyle="round,pad=0.5", 
                    facecolor="white", 
                    alpha=0.9,
                    edgecolor="gray"
                )
            )
        except:
            plt.figtext(
                0.02, 0.98, help_text, 
                fontsize=11, 
                verticalalignment='top',
                bbox=dict(
                    boxstyle="round,pad=0.5", 
                    facecolor="white", 
                    alpha=0.9,
                    edgecolor="gray"
                )
            )
    
    def _reset_interface(self):
        """Reset entire interface"""
        self._delete_all_grains()
        
        # Rerun full image inference
        if self.image is not None:
            self._run_global_inference()
        
        self.ax.clear()
        self.ax.imshow(self.image)
        
        image_name = Path(self.image_path).name if self.image_path else "Unnamed image"
        title_text = f"FastSAM Enhanced Interactive Segmentation - {image_name}"
        self.ax.set_title(title_text, fontsize=16, fontproperties='Microsoft YaHei')
        
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        self._show_help_text_fixed()
        self.fig.canvas.draw()
        
        print("Interface completely reset")
    
    def show_interactive_interface(self):
        """Show interactive interface"""
        if self.image is None:
            print("Please load image first")
            return
        
        if self.fastsam_model is None:
            print("FastSAM model not initialized")
            return
        
        print("Creating interactive interface...")
        
        try:
            # Set Chinese font
            try:
                import matplotlib
                from matplotlib import rcParams
                
                import platform
                system = platform.system()
                
                if system == 'Windows':
                    rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
                    rcParams['axes.unicode_minus'] = False
                elif system == 'Darwin':
                    rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB']
                    rcParams['axes.unicode_minus'] = False
                else:
                    rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei']
                    rcParams['axes.unicode_minus'] = False
                    
            except Exception as e:
                print(f"Failed to set Chinese font: {e}")
            
            # Create figure
            self.fig, self.ax = plt.subplots(figsize=(14, 10))
            self.ax.imshow(self.image)
            
            image_name = Path(self.image_path).name if self.image_path else "Unnamed image"
            title_text = f"FastSAMEnhanced interactive segmentation - {image_name}"
            self.ax.set_title(title_text, fontsize=16, fontproperties='Microsoft YaHei')
            
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            
            # Connect events
            self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
            self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
            self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
            self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
            
            self._show_help_text_fixed()
            
            plt.tight_layout()
            
            print("FastSAM interactive interface started")
            print("Tips:")
            print("  1. Left drag to draw selection box")
            print("  2. Left click to point select small grains")
            print("  3. Press 'Shift+S' to quickly save results")
            
            self.gui_running = True
            
            if backend == 'Agg':
                print("No GUI environment detected, will save result image")
                output_path = self.output_dir / "interactive_result.png"
                self.fig.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Results saved to: {output_path}")
                plt.close(self.fig)
                return
            
            plt.show(block=True)
            
            print("Interactive window closed")
            
        except Exception as e:
            print(f"Failed to display interactive interface: {e}")
            traceback.print_exc()
    
    def run_interactive_mode(self, image_path: str = None):
        """Run complete interactive mode"""
        try:
            if not self.model_loaded:
                print("Model not loaded, cannot run interactive mode")
                return
            
            if image_path:
                print(f"Loading specified image: {image_path}")
                if not self.load_image_from_path(image_path):
                    print("Image loading failed, exiting interactive mode")
                    return
            else:
                print("Please select image through file dialog...")
                if not self.load_image_with_gui():
                    print("Image selection failed, exiting interactive mode")
                    return
            
            print("Starting interactive interface...")
            self.show_interactive_interface()
            
        except Exception as e:
            print(f"Interactive mode failed: {e}")
            traceback.print_exc()
    
    def _masks_to_polygons(self) -> List[ShapelyPolygon]:
        """Convert masks to polygon list"""
        polygons = []
        
        for grain in self.grains:
            if grain['mask'] is not None and np.any(grain['mask']):
                try:
                    mask = grain['mask']
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    
                    contours, _ = cv2.findContours(
                        mask_uint8, 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        
                        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                        
                        if len(approx) >= 3:
                            polygon_points = [(point[0][0], point[0][1]) for point in approx]
                            polygon = ShapelyPolygon(polygon_points)
                            
                            if polygon.is_valid and polygon.area > 0:
                                polygons.append(polygon)
                    
                except Exception as e:
                    print(f"Failed to convert mask to polygon (grain#{grain['id']}): {e}")
        
        return polygons
    
    def _generate_grain_dataframe(self) -> pd.DataFrame:
        """Generate grain DataFrame - fix geometry calculation issues"""
        if len(self.grains) == 0:
            return pd.DataFrame()
        
        try:
            basic_data = []
            for i, grain in enumerate(self.grains):
                if grain['mask'] is not None and np.any(grain['mask']):
                    mask = grain['mask']
                    
                    area = np.sum(mask)
                    y_indices, x_indices = np.where(mask)
                    
                    if len(y_indices) > 0 and len(x_indices) > 0:
                        centroid_y = np.mean(y_indices)
                        centroid_x = np.mean(x_indices)
                        
                        y_min, y_max = np.min(y_indices), np.max(y_indices)
                        x_min, x_max = np.min(x_indices), np.max(x_indices)
                        bbox_width = x_max - x_min
                        bbox_height = y_max - y_min
                        
                        # Calculate perimeter
                        perimeter = 0
                        try:
                            mask_uint8 = (mask * 255).astype(np.uint8)
                            contours, _ = cv2.findContours(
                                mask_uint8, 
                                cv2.RETR_EXTERNAL, 
                                cv2.CHAIN_APPROX_SIMPLE
                            )
                            if contours:
                                largest_contour = max(contours, key=cv2.contourArea)
                                perimeter = cv2.arcLength(largest_contour, True)
                        except Exception:
                            # Approximate calculation
                            perimeter = 4 * np.sqrt(area) * 0.9
                        
                        # Calculate basic geometry parameters
                        circularity = 0
                        if perimeter > 0:
                            circularity = (4 * np.pi * area) / (perimeter ** 2)
                        
                        aspect_ratio = 0
                        if bbox_height > 0:
                            aspect_ratio = bbox_width / bbox_height
                        
                        compactness = 0
                        if bbox_width > 0 and bbox_height > 0:
                            compactness = area / (bbox_width * bbox_height)
                        
                        basic_data.append({
                            'label': grain['id'],
                            'area': float(area),
                            'centroid_x': float(centroid_x),
                            'centroid_y': float(centroid_y),
                            'bbox_width': float(bbox_width),
                            'bbox_height': float(bbox_height),
                            'perimeter': float(perimeter),
                            'circularity': float(circularity),
                            'aspect_ratio': float(aspect_ratio),
                            'compactness': float(compactness),
                            'confidence': float(grain.get('score', 0.5))
                        })
            
            if not basic_data:
                return pd.DataFrame()
            
            basic_df = pd.DataFrame(basic_data)
            
            # Add missing columns to matchGrainShapeMetricsexpectation
            if 'major_axis_length' not in basic_df.columns:
                # Estimate major axis length
                basic_df['major_axis_length'] = basic_df['bbox_width']
            
            if 'minor_axis_length' not in basic_df.columns:
                # Estimate minor axis length
                basic_df['minor_axis_length'] = basic_df['bbox_height']
            
            if 'orientation' not in basic_df.columns:
                # Default orientation
                basic_df['orientation'] = 0.0
            
            return basic_df
                
        except Exception as e:
            print(f"Failed to generate grain data: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    
    def _generate_complete_outputs(self, output_dir: Optional[Path] = None) -> Path:
        """Generate complete output files"""
        if len(self.grains) == 0:
            print("No segmented grains, cannot generate output files")
            return None
        
        if output_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_name = Path(self.image_path).stem if self.image_path else "interactive"
            output_dir = self.output_dir / f"{image_name}_{timestamp}"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating complete output to: {output_dir}")
        
        try:
            # 1. Generate polygons
            self.polygons = self._masks_to_polygons()
            
            if len(self.polygons) == 0:
                print("Cannot generate valid polygons")
                return None
            
            print(f"Generated {len(self.polygons)} polygons")
            
            # 2. Generate grain data
            self.grain_data = self._generate_grain_dataframe()
            
            if self.grain_data.empty:
                print("Cannot generate grain data")
                return None
            
            print(f"Grain data contains {len(self.grain_data.columns)} parameters")
            
            # 3. Save interactive interface screenshot
            if self.fig is not None:
                vis_path = output_dir / "interactive_visualization.png"
                try:
                    self.fig.savefig(vis_path, dpi=300, bbox_inches='tight')
                    print(f"Interactive interface screenshot saved to: {vis_path}")
                except Exception as e:
                    print(f"Failed to save interactive interface screenshot: {e}")
            
            # 4. GenerateYOLOstyle visualization
            if self.image is not None:
                fig, axes = plt.subplots(1, 2, figsize=(20, 10))
                
                # Left: contour map
                axes[0].imshow(self.image)
                axes[0].set_title(f'FastSAM Grain Segmentation (n={len(self.polygons)})', fontsize=16)
                axes[0].axis('off')
                
                for poly in self.polygons:
                    if poly.is_valid:
                        x, y = poly.exterior.xy
                        axes[0].plot(x, y, color='red', linewidth=1, alpha=0.8)
                
                # Right: colored fill map
                axes[1].imshow(self.image)
                axes[1].set_title('Colored Grain Annotation', fontsize=16)
                axes[1].axis('off')
                
                colors = plt.cm.tab20(np.linspace(0, 1, len(self.polygons)))
                for i, poly in enumerate(self.polygons):
                    if poly.is_valid and i < len(colors):
                        x, y = poly.exterior.xy
                        axes[1].fill(x, y, color=colors[i], alpha=0.3)
                
                plt.tight_layout()
                
                plot_path = output_dir / "segmentation_result.png"
                fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                
                print(f"YOLO-style visualization saved to: {plot_path}")
            
            # 5. SaveCSVfile
            if not self.grain_data.empty:
                csv_path = output_dir / "grain_statistics.csv"
                
                if GEOMETRY_AVAILABLE and self.geometry_config:
                    try:
                        grain_data_to_save = select_columns_for_grain_statistics_csv(
                            self.grain_data,
                            self.geometry_config,
                            strict=False
                        )
                        
                        if grain_data_to_save is not None and not grain_data_to_save.empty:
                            grain_data_to_save.to_csv(csv_path, index=False, encoding='utf-8')
                        else:
                            self.grain_data.to_csv(csv_path, index=False, encoding='utf-8')
                    except Exception as e:
                        print(f"Configuration filtering failed: {e}")
                        self.grain_data.to_csv(csv_path, index=False, encoding='utf-8')
                else:
                    self.grain_data.to_csv(csv_path, index=False, encoding='utf-8')
                
                print(f"Grain statistics table saved to: {csv_path}")
            
            # 6. CreateJSONSummary
            summary = {
                'image_path': str(self.image_path) if self.image_path else "GUI_selected",
                'image_name': Path(self.image_path).name if self.image_path else "interactive",
                'success': True,
                'grains_count': len(self.polygons),
                'processing_time': time.time() - self.start_time if self.start_time else 0,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'processing_mode': 'fastsam_interactive'
            }
            
            if self.image is not None:
                summary['image_size'] = {
                    'height': self.image.shape[0],
                    'width': self.image.shape[1],
                    'channels': self.image.shape[2]
                }
            
            json_path = output_dir / "summary.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"JSON summary saved to: {json_path}")
            
            print(f"\nAll results saved to: {output_dir}")
            
            return output_dir
            
        except Exception as e:
            print(f"Failed to generate complete output: {e}")
            traceback.print_exc()
            return None
    
    def _show_save_options(self):
        """Show save options"""
        if len(self.grains) == 0:
            print("No grains to save")
            return
        
        save_choice = input("\nSave options:\n1. Quick save complete results\n2. Custom save path\n3. Cancel\nSelect (1-3): ").strip()
        
        if save_choice == '1':
            output_dir = self._generate_complete_outputs()
            if output_dir:
                print(f"Results saved to: {output_dir}")
        elif save_choice == '2':
            try:
                root = tk.Tk()
                root.withdraw()
                folder_path = filedialog.askdirectory(title="Select save directory")
                root.destroy()
                
                if folder_path:
                    output_dir = self._generate_complete_outputs(Path(folder_path))
                    if output_dir:
                        print(f"Results saved to: {output_dir}")
            except Exception as e:
                print(f"Save failed: {e}")


def main():
    """Main function：DirectRunEnhanced interactiveFastSAM"""
    print("=" * 70)
    print("FastSAM Enhanced Interactive Segmentation System (Fixed Version)")
    print("=" * 70)
    
    model_path = "models/FastSAM-s.pt"
    device = "cpu"
    
    interactive_system = PureFastSAMInteractiveEnhanced(
        model_path=model_path,
        device=device
    )
    
    interactive_system.run_interactive_mode()


if __name__ == "__main__":
    main()