"""
MobileSAM Fixed Interactive Interface Module (Enhanced Version)
Filename: mobilesam_interactive.py
Function: Point-based pure interactive MobileSAM segmentation, fixing multi-grain segmentation issues
Enhanced: Supports complete result output including statistics, CSV, JSON, etc., fully consistent with YOLO workflow output
"""

import os
import sys
import numpy as np
import matplotlib

# Helper function to get resource path (works for both dev and PyInstaller)
def get_resource_path(relative_path):
    """Get absolute path to resource, works for both dev and PyInstaller"""
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle
        base_path = Path(sys._MEIPASS)
    else:
        # Running in normal Python environment
        base_path = Path(__file__).parent
    return base_path / relative_path

def setup_backend():
    """Intelligent backend setup, GUI backend priority"""
    try:
        import tkinter
        matplotlib.use('TkAgg')
        return 'TkAgg'
    except ImportError:
        pass
    except Exception:
        pass
    
    if os.getenv('DISPLAY') and not os.getenv('SSH_CONNECTION'):
        for backend in ['TkAgg', 'Qt5Agg', 'WXAgg']:
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
from typing import List, Dict, Optional, Tuple, Any
import warnings
import cv2
import numpy as np
warnings.filterwarnings('ignore')

try:
    from mobile_sam import sam_model_registry, SamPredictor
    MOBILESAM_AVAILABLE = True
except ImportError:
    MOBILESAM_AVAILABLE = False
    print("MobileSAM library not installed")

try:
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
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

# Import core segmentation functions (migrated from segmenteverygrain)
try:
    from core.segment_core import (
        create_labeled_image,
        collect_polygon_from_mask,
        plot_image_w_colorful_grains,
        plot_grain_axes_and_centroids,
        find_connected_components,
        merge_overlapping_polygons
    )
    PROJECT1_AVAILABLE = True
except ImportError as e:
    PROJECT1_AVAILABLE = False
    print(f"Core segmentation functions unavailable: {e}")

# Import grain marker module for labeled visualization
try:
    from utils.grain_marker import add_grain_labels, add_labels_with_config
    GRAIN_MARKER_AVAILABLE = True
    print("Grain marker module loaded successfully")
except ImportError as e:
    GRAIN_MARKER_AVAILABLE = False
    print(f"Grain marker module unavailable: {e}")


class PureMobileSAMInteractiveEnhanced:
    """Enhanced pure interactive MobileSAM (output fully consistent with YOLO workflow)"""
    
    def __init__(self, model_path: str = "models/mobile_sam.pt", 
                 device: str = "cpu", model_type: str = "vit_t"):
        self.model_path = model_path
        self.device = device
        self.model_type = model_type
        
        self.image = None
        self.image_path = None
        self.predictor = None
        self.model_loaded = False
        
        self.grains = []
        self.current_grain_id = 0
        
        self.polygons = []
        self.labels = None
        self.mask_all = None
        self.grain_data = None
        
        self.fig = None
        self.ax = None
        self.grain_patches = {}
        self.point_markers = []
        self.grain_texts = {}
        
        # Unified output directory: results/mobilesam/interactive/
        self.output_dir = Path("results") / "mobilesam" / "interactive"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.geometry_config = None
        if GEOMETRY_AVAILABLE:
            try:
                # First, ensure configs are available in working directory
                configs_dir = Path("configs")
                config_path = configs_dir / "geometry.yaml"
                
                # If not in working directory, try to copy from bundle
                if not config_path.exists() and getattr(sys, 'frozen', False):
                    import shutil
                    bundle_configs = Path(sys._MEIPASS) / "configs"
                    if bundle_configs.exists():
                        configs_dir.mkdir(exist_ok=True)
                        for yaml_file in bundle_configs.glob("*.yaml"):
                            shutil.copy2(yaml_file, configs_dir / yaml_file.name)
                        print(f"Copied configs from bundle to: {configs_dir}")
                
                # Now try to load geometry.yaml
                if config_path.exists():
                    self.geometry_config = load_geometry_config(str(config_path))
                    print(f"Geometry configuration loaded from: {config_path}")
                else:
                    # Try alternative paths
                    alt_paths = [
                        get_resource_path("configs/geometry.yaml"),
                        get_resource_path("geometry.yaml"),
                        Path(__file__).parent / "configs" / "geometry.yaml",
                    ]
                    for alt_path in alt_paths:
                        if alt_path.exists():
                            self.geometry_config = load_geometry_config(str(alt_path))
                            print(f"Geometry configuration loaded from: {alt_path}")
                            break
                    else:
                        print("Warning: Could not find geometry.yaml, using default CSV export")
            except Exception as e:
                print(f"Failed to load geometry configuration: {e}")
        
        self.scale_factor = None
        self.scale_detection_success = False
        
        # Scale calibration
        self.scale_calibrator = None
        self.is_scale_calibration_mode = False
        if SCALE_CALIBRATION_AVAILABLE:
            self.scale_calibrator = InteractiveScaleCalibrator()
        
        self.start_time = None
        self.total_grains = 0
        self.total_interactions = 0
        
        self.gui_running = False
        
        print("MobileSAM Interactive System (Enhanced Version)")
        print("Output fully consistent with YOLO workflow")
        
        self._load_sam_model()
    
    def _load_sam_model(self) -> bool:
        """Loading MobileSAM model"""
        if not MOBILESAM_AVAILABLE:
            print("MobileSAM library not available")
            return False
        
        try:
            print(f"Loading MobileSAM model: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                print(f"Model file does not exist: {self.model_path}")
                return False
            
            sam = sam_model_registry[self.model_type](checkpoint=self.model_path)
            sam.to(device=self.device)
            
            self.predictor = SamPredictor(sam)
            self.model_loaded = True
            
            print(f"MobileSAM model loaded successfully (device: {self.device}, type: {self.model_type})")
            return True
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            traceback.print_exc()
            return False
    
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
        """Load image through GUI file selection dialog"""
        if not self.model_loaded or self.predictor is None:
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
            
            print("Setting image to SAM predictor...")
            self.predictor.set_image(self.image)
            
            self.grains = []
            self.current_grain_id = 0
            self.grain_patches = {}
            self.point_markers = []
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
        if not self.model_loaded or self.predictor is None:
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
            
            print("Setting image to SAM predictor...")
            self.predictor.set_image(self.image)
            
            self.grains = []
            self.current_grain_id = 0
            self.grain_patches = {}
            self.point_markers = []
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
                                continue
                    
                    print(f"Grain #{grain['id']} OpenCV conversion failed, trying skimage")
                    contours = measure.find_contours(mask, 0.5)
                    
                    if len(contours) > 0:
                        main_contour = max(contours, key=lambda x: len(x))
                        
                        if len(main_contour) >= 3:
                            polygon_points = [(point[1], point[0]) for point in main_contour]
                            polygon = ShapelyPolygon(polygon_points)
                            
                            if polygon.is_valid and polygon.area > 0:
                                polygons.append(polygon)
                    
                except Exception as e:
                    print(f"Failed to convert mask to polygon (grain #{grain['id']}): {e}")
        
        return polygons
    
    def _generate_unified_grain_dataframe(self) -> pd.DataFrame:
        """
        Generate grain DataFrame fully consistent with YOLO workflow
        """
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
                        width = x_max - x_min
                        height = y_max - y_min
                        
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
                            perimeter = 4 * np.sqrt(area) * 0.9
                        
                        confidence = float(grain.get('confidence', 0.5))
                        
                        # Calculate major_axis_length and minor_axis_length for geometry metrics
                        try:
                            from skimage.measure import regionprops
                            regions = regionprops(mask.astype(np.uint8))
                            if regions:
                                major_axis_length = regions[0].major_axis_length
                                minor_axis_length = regions[0].minor_axis_length
                            else:
                                major_axis_length = max(width, height)
                                minor_axis_length = min(width, height)
                        except Exception:
                            # Fallback if skimage not available
                            major_axis_length = max(width, height)
                            minor_axis_length = min(width, height)
                        
                        # Extract contour coordinates for advanced geometry calculations
                        coordinates = None
                        try:
                            mask_uint8 = (mask * 255).astype(np.uint8)
                            contours, _ = cv2.findContours(
                                mask_uint8, 
                                cv2.RETR_EXTERNAL, 
                                cv2.CHAIN_APPROX_SIMPLE
                            )
                            if contours:
                                largest_contour = max(contours, key=cv2.contourArea)
                                # Convert to [[x, y], ...] format
                                coordinates = largest_contour.reshape(-1, 2).tolist()
                        except Exception:
                            pass
                        
                        basic_data.append({
                            'grain_id': grain['id'],  # Uniformly use grain_id
                            'area': float(area),
                            'centroid_x': float(centroid_x),
                            'centroid_y': float(centroid_y),
                            'width': float(width),     # Uniformly use width
                            'height': float(height),   # Uniformly use height
                            'perimeter': float(perimeter),
                            'confidence': float(confidence),
                            'mask_area_pixels': int(area),  # Extra info
                            'major_axis_length': float(major_axis_length),
                            'minor_axis_length': float(minor_axis_length),
                            'coordinates': coordinates   # For fractal dimension and Fourier descriptors
                        })
            
            if not basic_data:
                return pd.DataFrame()
            
            basic_df = pd.DataFrame(basic_data)
            
            if GEOMETRY_AVAILABLE and len(self.polygons) > 0:
                try:
                    shape_calculator = GrainShapeMetrics(basic_df)
                    geometry_df = shape_calculator.compute_all_metrics()
                    
                    # Remove coordinates column (too large for CSV) but keep all other geometry parameters
                    if 'coordinates' in geometry_df.columns:
                        geometry_df = geometry_df.drop(columns=['coordinates'])
                    
                    print(f"Advanced geometry parameters calculated, total {len(geometry_df.columns)} parameters")
                    print(f"Column names: {list(geometry_df.columns)}")
                    
                    return geometry_df
                    
                except Exception as e:
                    print(f"GrainShapeMetrics calculation failed: {e}")
                    
                    basic_df = basic_df.rename(columns={'grain_id': 'grain_id'})
                    return basic_df
            else:
                print("Geometry module unavailable, using basic geometry parameters")
                basic_df = basic_df.rename(columns={'grain_id': 'grain_id'})
                return basic_df
                
        except Exception as e:
            print(f"Cannot generate grain data: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    
    def _generate_complete_outputs(self, output_dir: Optional[Path] = None) -> Path:
        """
        Generate complete output files (fully consistent with YOLO workflow)
        """
        if len(self.grains) == 0:
            print("No segmented grains, cannot generate output files")
            return None
        
        if output_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_name = Path(self.image_path).stem if self.image_path else "interactive"
            output_dir = self.output_dir / f"{image_name}_{timestamp}"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating complete output to: {output_dir}")
        
        try:
            self.polygons = self._masks_to_polygons()
            
            if len(self.polygons) == 0:
                print("Cannot generate valid polygons")
                return None
            
            print(f"Generated {len(self.polygons)} polygons")
            
            self.grain_data = self._generate_unified_grain_dataframe()
            
            if self.grain_data.empty:
                print("Cannot generate grain data")
                return None
            
            print(f"Grain data contains {len(self.grain_data.columns)} parameters")
            print(f"Data columns: {list(self.grain_data.columns)}")
            print(f"Scale detection success: {self.scale_detection_success}, scale factor: {self.scale_factor}")
            
            if self.scale_detection_success and self.scale_factor:
                if 'area' in self.grain_data.columns:
                    self.grain_data['area'] = pd.to_numeric(self.grain_data['area'], errors='coerce')
                    
                    # Calculate area_um2 and diameter_um for all rows
                    self.grain_data['area_um2'] = self.grain_data['area'] * (self.scale_factor ** 2)
                    self.grain_data['diameter_um'] = 2 * np.sqrt(self.grain_data['area_um2'] / np.pi)
                    
                    # Check for NaN values
                    valid_count = self.grain_data['area_um2'].notna().sum()
                    print(f"Real dimensions calculated for {valid_count} grains, scale factor: {self.scale_factor:.4f} μm/px")
                    print(f"Sample area_um2: {self.grain_data['area_um2'].iloc[0] if valid_count > 0 else 'N/A'}")
                else:
                    print("Warning: 'area' column not found in grain_data")
            else:
                print("Warning: Scale calibration not performed. area_um2 and diameter_um not calculated.")
            
            if self.fig is not None:
                vis_path = output_dir / "segmentation_result.png"
                self.fig.savefig(vis_path, dpi=300, bbox_inches='tight')
                print(f"Interactive interface screenshot saved to: {vis_path}")
                
                self._generate_yolo_style_visualization(output_dir)
                
                # Save labeled image with grain numbers
                self._generate_labeled_visualization(output_dir)
            
            self._generate_simple_masks(output_dir)
            
            if not self.grain_data.empty:
                csv_path = output_dir / "grain_statistics.csv"
                
                print(f"Before CSV export, grain_data columns: {list(self.grain_data.columns)}")
                print(f"area_um2 in columns: {'area_um2' in self.grain_data.columns}")
                print(f"diameter_um in columns: {'diameter_um' in self.grain_data.columns}")
                
                if GEOMETRY_AVAILABLE and self.geometry_config:
                    try:
                        print(f"Using geometry_config to filter columns")
                        grain_data_to_save = select_columns_for_grain_statistics_csv(
                            self.grain_data,
                            self.geometry_config,
                            strict=False
                        )
                        
                        if grain_data_to_save is not None and not grain_data_to_save.empty:
                            print(f"Config-driven output, keeping {len(grain_data_to_save.columns)} columns")
                            print(f"Final columns: {list(grain_data_to_save.columns)}")
                        else:
                            grain_data_to_save = self.grain_data
                    except Exception as e:
                        print(f"Config filtering failed: {e}")
                        grain_data_to_save = self.grain_data
                else:
                    print(f"No geometry_config, saving all columns")
                    grain_data_to_save = self.grain_data
                
                grain_data_to_save.to_csv(csv_path, index=False, encoding='utf-8')
                print(f"Grain statistics table saved to: {csv_path}")
                
                self._print_statistics_summary(grain_data_to_save)
            
            summary = self._create_yolo_style_summary()
            json_path = output_dir / "summary.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"JSON summary saved to: {json_path}")
            
            self._save_performance_info(output_dir)
            
            print(f"All results saved to: {output_dir}")
            print("Output files fully consistent with YOLO workflow")
            return output_dir
            
        except Exception as e:
            print(f"Failed to generate complete output: {e}")
            traceback.print_exc()
            return None
    
    def _generate_yolo_style_visualization(self, output_dir: Path):
        """Generate YOLO-style visualization"""
        if self.image is None:
            return
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            
            axes[0].imshow(self.image)
            axes[0].set_title(f'Rock Grain Segmentation (n={len(self.polygons)})', fontsize=16)
            axes[0].axis('off')
            
            for poly in self.polygons:
                if poly.is_valid:
                    x, y = poly.exterior.xy
                    axes[0].plot(x, y, color='red', linewidth=1, alpha=0.8)
            
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
            
        except Exception as e:
            print(f"Failed to generate YOLO-style visualization: {e}")
    
    def _generate_labeled_visualization(self, output_dir: Path):
        """Generate labeled visualization with grain numbers"""
        try:
            if not GRAIN_MARKER_AVAILABLE or self.grain_data is None or self.grain_data.empty:
                print("Grain marker not available or no grain data, skipping labeled visualization")
                return
            
            # Prepare grain data with required columns for add_labels_with_config
            # It expects: 'label', 'centroid-0', 'centroid-1', 'area'
            label_data = self.grain_data.copy()
            
            # Map column names to what add_labels_with_config expects
            if 'grain_id' in label_data.columns:
                label_data['label'] = label_data['grain_id']
            if 'centroid_y' in label_data.columns:
                label_data['centroid-0'] = label_data['centroid_y']  # y coordinate (row)
            if 'centroid_x' in label_data.columns:
                label_data['centroid-1'] = label_data['centroid_x']  # x coordinate (column)
            
            print(f"Label data columns: {list(label_data.columns)}")
            print(f"Sample label data: {label_data[['label', 'centroid-0', 'centroid-1', 'area']].head() if all(c in label_data.columns for c in ['label', 'centroid-0', 'centroid-1', 'area']) else 'Missing columns'}")
            
            # Create labeled image
            fig_labeled, ax_labeled = plt.subplots(figsize=(15, 10))
            ax_labeled.imshow(self.image)
            ax_labeled.axis('off')
            
            # Add grain labels
            grain_label_config = {
                'enabled': True,
                'font_size': 11,
                'font_color': 'white',
                'bbox_color': 'black',
                'bbox_alpha': 0.6,
                'bbox_style': 'round,pad=0.3',
                'label_format': 'id',
                'placement_strategy': 'auto',
                'seed': 42,
                'avoid_overlap': True,
                'min_distance': 10
            }
            
            ax_labeled = add_labels_with_config(
                ax=ax_labeled,
                grain_data=label_data,
                image_shape=self.image.shape,
                config=grain_label_config
            )
            
            # Hide axes and borders
            ax_labeled.set_xticks([])
            ax_labeled.set_yticks([])
            ax_labeled.set_xlim([0, self.image.shape[1]])
            ax_labeled.set_ylim([self.image.shape[0], 0])
            plt.tight_layout()
            
            # Save labeled image
            labeled_path = output_dir / "segmentation_labeled.png"
            fig_labeled.savefig(labeled_path, dpi=300, bbox_inches='tight', 
                               pad_inches=0, facecolor='white')
            plt.close(fig_labeled)
            
            print(f"Labeled visualization saved to: {labeled_path}")
            
        except Exception as e:
            print(f"Failed to generate labeled visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_simple_masks(self, output_dir: Path):
        """Generate segmentation mask"""
        if self.image is None:
            return
        
        h, w = self.image.shape[:2]
        mask_all = np.zeros((h, w), dtype=np.uint8)
        
        for grain in self.grains:
            if grain['mask'] is not None:
                mask_all = np.maximum(mask_all, grain['mask'].astype(np.uint8))
        
        mask_path = output_dir / "segmentation_mask.png"
        mask_uint8 = mask_all * 255
        Image.fromarray(mask_uint8).save(mask_path)
        print(f"Segmentation mask saved to: {mask_path}")
        
        self.mask_all = mask_all
    
    def _print_statistics_summary(self, grain_data: pd.DataFrame = None):
        """Print statistics summary"""
        if grain_data is None:
            grain_data = self.grain_data
            
        if grain_data is None or grain_data.empty:
            return
        
        print(f"Grain statistics summary:")
        print(f"  Total grain count: {len(grain_data)}")
        
        if 'area' in grain_data.columns:
            area_sum = grain_data['area'].sum()
            area_mean = grain_data['area'].mean()
            area_min = grain_data['area'].min()
            area_max = grain_data['area'].max()
            
            print(f"  Total pixel area: {area_sum:.0f}")
            print(f"  Average pixel area: {area_mean:.1f}")
            print(f"  Minimum pixel area: {area_min:.1f}")
            print(f"  Maximum pixel area: {area_max:.1f}")
        
        if self.scale_detection_success and 'area_um2' in grain_data.columns:
            area_um2_sum = grain_data['area_um2'].sum()
            print(f"  Total real area: {area_um2_sum:.0f} μm²")
    
    def _create_yolo_style_summary(self) -> Dict[str, Any]:
        """Create JSON summary consistent with YOLO workflow"""
        summary = {
            'image_path': str(self.image_path) if self.image_path else "GUI_selected",
            'image_name': Path(self.image_path).name if self.image_path else "interactive",
            'success': True,
            'grains_count': len(self.polygons),
            'error_message': None,
            'output_files': [],
            'processing_time': time.time() - self.start_time if self.start_time else 0,
            'performance_metrics': {},
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'scale_factor': float(self.scale_factor) if self.scale_factor else None,
            'scale_detection_success': self.scale_detection_success,
            'system_version': 'MobileSAM Interactive v2.0',
            'processing_mode': 'interactive'
        }
        
        if self.image is not None:
            summary['image_size'] = {
                'height': self.image.shape[0],
                'width': self.image.shape[1],
                'channels': self.image.shape[2]
            }
        
        if self.grain_data is not None and not self.grain_data.empty:
            if 'area' in self.grain_data.columns:
                summary['area_statistics_pixels'] = {
                    'total': float(self.grain_data['area'].sum()),
                    'average': float(self.grain_data['area'].mean()),
                    'min': float(self.grain_data['area'].min()),
                    'max': float(self.grain_data['area'].max()),
                    'std': float(self.grain_data['area'].std())
                }
            
            if self.scale_detection_success and 'area_um2' in self.grain_data.columns:
                summary['area_statistics_um2'] = {
                    'total': float(self.grain_data['area_um2'].sum()),
                    'average': float(self.grain_data['area_um2'].mean()),
                    'min': float(self.grain_data['area_um2'].min()),
                    'max': float(self.grain_data['area_um2'].max())
                }
        
        return summary
    
    def _save_performance_info(self, output_dir: Path):
        """Save performance info"""
        processing_time = time.time() - self.start_time if self.start_time else 0
        
        performance = {
            "processing_time_seconds": float(processing_time),
            "total_grains": len(self.grains),
            "total_interactions": self.total_interactions,
            "average_time_per_grain": float(processing_time / max(len(self.grains), 1)),
            "average_points_per_grain": float(self.total_interactions / max(len(self.grains), 1)),
            "average_confidence": float(self.grain_data['confidence'].mean()) if not self.grain_data.empty and 'confidence' in self.grain_data.columns else 0,
            "version": "interactive_yolo_consistent_v1.0"
        }
        
        perf_path = output_dir / "performance.json"
        with open(perf_path, 'w', encoding='utf-8') as f:
            json.dump(performance, f, indent=2, ensure_ascii=False)
        print(f"Performance info saved to: {perf_path}")
    
    def _get_grain_at_point(self, x: float, y: float) -> Optional[int]:
        """Get grain ID at clicked position"""
        for grain in self.grains:
            if 'mask' in grain and grain['mask'] is not None:
                h, w = grain['mask'].shape
                ix, iy = int(x), int(y)
                
                if 0 <= ix < w and 0 <= iy < h:
                    if grain['mask'][iy, ix]:
                        return grain['id']
        return None
    
    def _create_new_grain(self, x: float, y: float, is_foreground: bool = True) -> int:
        """Create new grain"""
        self.current_grain_id += 1
        new_grain = {
            'id': self.current_grain_id,
            'points': [{'x': x, 'y': y, 'is_foreground': is_foreground}],
            'mask': None,
            'bbox': None,
            'color': np.random.rand(3,),
            'confidence': 0.5
        }
        
        self.grains.append(new_grain)
        self.total_grains += 1
        print(f"Creating new grain #{self.current_grain_id}")
        
        return self.current_grain_id
    
    def _add_point_to_current_grain(self, x: float, y: float, is_foreground: bool = True):
        """Add point to current grain"""
        if not self.grains:
            print("No current grain, create new grain first")
            return
        
        current_grain = self.grains[-1]
        current_grain['points'].append({
            'x': x, 
            'y': y, 
            'is_foreground': is_foreground
        })
        
        self.total_interactions += 1
        point_type = "foreground point" if is_foreground else "background point"
        print(f"Adding to grain #{current_grain['id']} {point_type}: ({x:.1f}, {y:.1f})")
    
    def _run_sam_segmentation_for_grain(self, grain_id: int):
        """Execute SAM segmentation for specified grain"""
        grain = None
        for g in self.grains:
            if g['id'] == grain_id:
                grain = g
                break
        
        if grain is None or not grain['points']:
            print(f"Grain #{grain_id} has no points, cannot segment")
            return
        
        try:
            input_points = []
            input_labels = []
            
            for point in grain['points']:
                input_points.append([point['x'], point['y']])
                input_labels.append(1 if point['is_foreground'] else 0)
            
            input_points = np.array(input_points, dtype=np.float32)
            input_labels = np.array(input_labels, dtype=np.int32)
            
            print(f"Segmenting grain #{grain_id}: {len(input_points)} prompt points")
            
            start_time = time.time()
            masks, scores, _ = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True
            )
            inference_time = time.time() - start_time
            
            if len(masks) == 0:
                print("No masks generated")
                return
            
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = scores[best_idx]
            
            grain['mask'] = mask
            grain['confidence'] = float(score)
            
            if np.any(mask):
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                grain['bbox'] = (xmin, ymin, xmax, ymax)
            
            print(f"Grain #{grain_id} segmentation successful! Confidence: {score:.3f}, time: {inference_time:.3f}s")
            
            self._update_grain_display(grain_id)
            
        except Exception as e:
            print(f"Failed to segment grain #{grain_id}: {e}")
            traceback.print_exc()
    
    def _draw_grain_with_text(self, grain):
        """Draw single grain and its text label"""
        try:
            grain_id = grain['id']
            mask = grain['mask']
            
            if mask is None or not np.any(mask):
                return
            
            import cv2
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_uint8, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                sx = largest_contour[:, 0, 0]
                sy = largest_contour[:, 0, 1]
                
                patch = self.ax.fill(sx, sy, 
                                   facecolor=grain['color'], 
                                   edgecolor='black',
                                   alpha=0.4, 
                                   linewidth=1.5)
                self.grain_patches[grain_id] = patch[0]
                
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
    
    def _update_grain_display(self, grain_id: int):
        """Update grain display"""
        grain = None
        for g in self.grains:
            if g['id'] == grain_id:
                grain = g
                break
        
        if grain is None or grain['mask'] is None:
            return
        
        if grain_id in self.grain_patches:
            self.grain_patches[grain_id].remove()
            del self.grain_patches[grain_id]
        
        if grain_id in self.grain_texts:
            self.grain_texts[grain_id].remove()
            del self.grain_texts[grain_id]
        
        self._draw_grain_with_text(grain)
        
        self.fig.canvas.draw()
    
    def _refresh_grain_display(self):
        """Refresh all grain displays"""
        try:
            for patch in self.grain_patches.values():
                patch.remove()
            self.grain_patches.clear()
            
            for text in self.grain_texts.values():
                text.remove()
            self.grain_texts.clear()
        
            for grain in self.grains:
                if grain['mask'] is not None:
                    self._draw_grain_with_text(grain)
            
            self.fig.canvas.draw()
            print(f"Display refreshed, current grain count: {len(self.grains)}")
        
        except Exception as e:
            print(f"Failed to refresh display: {e}")
    
    def _on_mouse_click(self, event):
        """Handle mouse click events"""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        
        x, y = event.xdata, event.ydata
        
        # Check if in scale calibration mode
        if self.is_scale_calibration_mode and self.scale_calibrator:
            # Handle scale calibration click
            # Note: Calibration completion is handled via callback (_on_scale_calibration_complete)
            # The on_click returns True when waiting for user input (async)
            waiting_for_input = self.scale_calibrator.on_click(event)
            if waiting_for_input:
                # User has clicked two points and is now entering actual length
                # Don't exit calibration mode - callback will handle completion
                print("Waiting for user to enter actual length...")
            return
        
        clicked_grain_id = self._get_grain_at_point(x, y)
        
        if event.button == 1:  # Left button: foreground point
            color = 'lime'
            marker = 'o'
            is_foreground = True
            point_type = "foreground point"
        else:  # Right button: background point
            color = 'red'
            marker = 'x'
            is_foreground = False
            point_type = "background point"
        
        marker_obj = self.ax.plot(x, y, marker=marker, color=color, 
                                markersize=10, markeredgewidth=2, alpha=0.8)
        self.point_markers.append(marker_obj[0])
        
        if clicked_grain_id is not None:
            print(f"Clicked grain #{clicked_grain_id}, adding {point_type}")
            self._add_point_to_current_grain(x, y, is_foreground)
            self._run_sam_segmentation_for_grain(clicked_grain_id)
        else:
            print(f"Creating new grain, adding {point_type}")
            new_grain_id = self._create_new_grain(x, y, is_foreground)
            self._run_sam_segmentation_for_grain(new_grain_id)
        
        self.fig.canvas.draw()
    
    def _on_key_press(self, event):
        """Keyboard event handling"""
        try:
            # Debug: print key press
            print(f"Key pressed: {event.key}")
            
            if event.key == 'x':  # Delete last grain
                print("Deleting last grain...")
                self._delete_last_grain()
            elif event.key == 'd':  # Delete all grains
                print("Deleting all grains...")
                self._delete_all_grains()
            elif event.key == 's':  # Save results
                print("Showing save options...")
                self._show_save_options()
            elif event.key == 'c':  # Clear all point marks
                print("Clearing point markers...")
                self._clear_point_markers()
            elif event.key == 'r':  # Restart
                print("Resetting interface...")
                self._reset_interface()
            elif event.key == 'q':  # Exit
                print("Exiting interactive interface")
                self.gui_running = False
                plt.close(self.fig)
            elif event.key == 'h':  # Show help
                print("Showing help...")
                self._show_help()
            elif event.key == 'm':  # Scale calibration mode
                print("Starting scale calibration...")
                self._start_scale_calibration()
            elif event.key == 'S':  # Shift+S: Quick save complete results
                print("Quick saving complete results...")
                self._generate_complete_outputs()
        except Exception as e:
            print(f"Error in key press handler: {e}")
            import traceback
            traceback.print_exc()
    
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
            
            self._refresh_grain_display()
        else:
            print("No grains to delete")
    
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
    
    def _clear_point_markers(self):
        """Clear all point markers"""
        for marker in self.point_markers:
            marker.remove()
        self.point_markers = []
        self.fig.canvas.draw()
        print("All point markers cleared")
    
    def _reset_interface(self):
        """Reset entire interface"""
        self._delete_all_grains()
        self._clear_point_markers()
        
        self.ax.clear()
        self.ax.imshow(self.image)
        
        image_name = Path(self.image_path).name if self.image_path else "Unnamed image"
        title_text = f"MobileSAM Enhanced Interactive Segmentation - {image_name}"
        self.ax.set_title(title_text, fontsize=16)
        
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        self._show_help_text_fixed()
        self.fig.canvas.draw()
        
        print("Interface completely reset")
    
    def _start_scale_calibration(self):
        """Start scale calibration mode"""
        try:
            print(f"Scale calibration available: {SCALE_CALIBRATION_AVAILABLE}")
            
            if not SCALE_CALIBRATION_AVAILABLE:
                print("ERROR: Scale calibration module not available")
                try:
                    messagebox.showerror("Error", "Scale calibration module not available")
                except Exception as e:
                    print(f"Could not show messagebox: {e}")
                return
                
            if self.is_scale_calibration_mode:
                print("Already in scale calibration mode")
                return
            
            if self.scale_calibrator is None:
                print("ERROR: Scale calibrator is None")
                return
            
            self.is_scale_calibration_mode = True
            self.original_title = self.ax.get_title()
            
            print("\n" + "="*60)
            print("SCALE CALIBRATION MODE")
            print("="*60)
            print("1. Click the START point of a known-length line")
            print("2. Click the END point of the line")
            print("3. Enter the actual length in microns when prompted")
            print("Press 'Escape' to cancel")
            print("="*60)
            
            # Pass callback to handle completion
            self.scale_calibrator.calibrate_scale(
                self.image, self.ax, self.fig,
                callback=self._on_scale_calibration_complete
            )
            
            # Update title
            self.ax.set_title(self.original_title + " [SCALE CALIBRATION MODE - Click two points]", 
                             fontsize=14, color='red', fontweight='bold')
            self.fig.canvas.draw()
        except Exception as e:
            print(f"Error starting scale calibration: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_scale_calibration_complete(self, scale_factor: float):
        """Callback when scale calibration is completed"""
        try:
            print(f"Scale calibration callback received! Factor: {scale_factor:.4f} um/px")
            self.scale_factor = scale_factor
            self.scale_detection_success = True
            self.is_scale_calibration_mode = False
            
            # Restore title
            if hasattr(self, 'original_title'):
                self.ax.set_title(self.original_title, fontsize=16)
            else:
                self.ax.set_title("Interactive Segmentation (Press 'h' for help)", fontsize=16)
            self.fig.canvas.draw()
            
            print("Scale calibration mode exited, ready for segmentation")
        except Exception as e:
            print(f"Error in scale calibration callback: {e}")
            import traceback
            traceback.print_exc()
    
    def _show_save_options(self):
        """Show save options menu"""
        if len(self.grains) == 0:
            print("No grains to save")
            return
        
        # Create a simple dialog using tkinter
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            # Ask user for choice using simple dialog
            from tkinter import simpledialog
            
            choice = simpledialog.askinteger(
                "Save Options",
                "Select save option:\n\n"
                "1. Quick save complete results (same as Shift+S)\n"
                "2. Custom save path\n"
                "3. Cancel\n\n"
                "Enter 1, 2, or 3:",
                minvalue=1,
                maxvalue=3,
                initialvalue=1
            )
            
            root.destroy()
            
            if choice == 1:
                # Quick save
                output_dir = self._generate_complete_outputs()
                if output_dir:
                    print(f"Results saved to: {output_dir}")
                    messagebox.showinfo("Save Complete", f"Results saved to:\n{output_dir}")
            elif choice == 2:
                # Custom save path
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                folder_path = filedialog.askdirectory(title="Select save directory")
                root.destroy()
                
                if folder_path:
                    output_dir = self._generate_complete_outputs(Path(folder_path))
                    if output_dir:
                        print(f"Results saved to: {output_dir}")
                        messagebox.showinfo("Save Complete", f"Results saved to:\n{output_dir}")
            else:
                print("Save cancelled")
                
        except Exception as e:
            print(f"Save failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _show_help(self):
        """Show help information"""
        help_text = (
            "Interactive Guide:\n\n"
            "Mouse Actions:\n"
            "• Left click (green dot): Mark grain position (foreground)\n"
            "• Right click (red x): Mark non-grain position (background)\n\n"
            "Keyboard Shortcuts:\n"
            "• 's': Show save options\n"
            "• 'S' (Shift+s): Quick save complete results\n"
            "• 'x': Delete last grain\n"
            "• 'd': Delete all grains\n"
            "• 'c': Clear point markers\n"
            "• 'r': Reset interface\n"
            "• 'm': Manual scale calibration (measure known length)\n"
            "• 'q': Quit program\n"
            "• 'h': Show this help\n"
        )
        
        messagebox.showinfo("Interactive Segmentation Help", help_text)
    
    def _show_help_text_fixed(self):
        """Display help text on interface"""
        help_text = (
            "Enhanced Interactive Guide:\n"
            "• Left click: Mark grain position (foreground point)\n"
            "• Right click: Mark non-grain position (background point)\n"
            "• 's' key: Show save options\n"
            "• 'S' key (Shift+s): Quick save complete results\n"
            "• 'x' key: Delete last grain\n"
            "• 'd' key: Delete all grains\n"
            "• 'c' key: Clear point markers\n"
            "• 'r' key: Reset interface\n"
            "• 'm' key: Manual scale calibration\n"
            "• 'q' key: Exit program\n"
        )
        
        try:
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
    
    def show_interactive_interface(self):
        """Display interactive interface"""
        if self.image is None:
            print("Please load image first")
            return
        
        if self.predictor is None:
            print("SAM predictor not initialized")
            return
        
        print("Creating interactive interface...")
        
        try:
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
            
            self.fig, self.ax = plt.subplots(figsize=(14, 10))
            self.ax.imshow(self.image)
            
            image_name = Path(self.image_path).name if self.image_path else "Unnamed image"
            title_text = f"MobileSAM Enhanced Interactive Segmentation - {image_name}"
            self.ax.set_title(title_text, fontsize=16)
            
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            
            self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_click)
            self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
            
            self._show_help_text_fixed()
            
            plt.tight_layout()
            
            print("Enhanced interactive interface started")
            
            self.gui_running = True
            
            if backend == 'Agg':
                print("No GUI environment detected, will save result image")
                output_path = self.output_dir / "interactive_result.png"
                self.fig.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Results saved to: {output_path}")
                plt.close(self.fig)
                return
            
            print("Interactive window opened, you can start marking grains")
            print("Press 'Shift+S' to quickly save complete results")
            
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


def main():
    """Main function: Run enhanced interactive MobileSAM directly"""
    print("MobileSAM Enhanced Interactive Segmentation System")
    
    model_path = "models/mobile_sam.pt"
    device = "cpu"
    model_type = "vit_t"
    
    interactive_system = PureMobileSAMInteractiveEnhanced(
        model_path=model_path,
        device=device,
        model_type=model_type
    )
    
    interactive_system.run_interactive_mode()


if __name__ == "__main__":
    main()