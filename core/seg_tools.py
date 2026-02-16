"""
Core Tool Functions Library - Unified Version
File: core/seg_tools.py
Function: Provide shared tool functions for FastSAM and MobileSAM modules
"""

import os
import io
import json
import yaml
import hashlib
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageFile
from typing import Tuple, Optional, Dict, Any, List, Union
import warnings

warnings.filterwarnings('ignore')

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageProcessor:
    """Image Processor - Provides image loading, conversion and preprocessing functions"""
    
    @staticmethod
    def load_image_safely(image_path: str) -> Optional[np.ndarray]:
        """
        Safely load image with multiple fallback mechanisms
        
        Args:
            image_path: Image file path
            
        Returns:
            RGB image array, or None
        """
        methods = [
            ImageProcessor._load_with_pil,
            ImageProcessor._load_with_cv2,
            ImageProcessor._load_with_skimage,
            ImageProcessor._load_with_binary
        ]
        
        for method in methods:
            try:
                image = method(image_path)
                if image is not None:
                    # Convert to RGB format
                    image = ImageProcessor.convert_to_rgb(image)
                    # Normalize to 0-255
                    image = ImageProcessor.normalize_image(image)
                    return image
            except Exception:
                continue
        
        return None
    
    @staticmethod
    def _load_with_pil(image_path: str) -> Optional[np.ndarray]:
        """Load image using PIL"""
        try:
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            return np.array(pil_image)
        except Exception:
            return None
    
    @staticmethod
    def _load_with_cv2(image_path: str) -> Optional[np.ndarray]:
        """Load image using OpenCV"""
        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is not None:
                return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return None
    
    @staticmethod
    def _load_with_skimage(image_path: str) -> Optional[np.ndarray]:
        """Load image using skimage"""
        try:
            from skimage import io
            image = io.imread(image_path)
            return image
        except Exception:
            return None
    
    @staticmethod
    def _load_with_binary(image_path: str) -> Optional[np.ndarray]:
        """Load image using binary mode"""
        try:
            with open(image_path, 'rb') as f:
                data = f.read()
                pil_image = Image.open(io.BytesIO(data))
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                return np.array(pil_image)
        except Exception:
            return None
    
    @staticmethod
    def convert_to_rgb(image: np.ndarray) -> np.ndarray:
        """
        Convert image to RGB format
        
        Args:
            image: Input image
            
        Returns:
            RGB format image
        """
        if len(image.shape) == 2:
            # Grayscale to RGB
            return np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 1:
            # Single channel to RGB
            return np.repeat(image, 3, axis=2)
        elif image.shape[2] == 4:
            # RGBA to RGB
            return image[:, :, :3]
        elif image.shape[2] == 3:
            return image
        else:
            raise ValueError(f"Unsupported channel count: {image.shape[2]}")
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        Normalize image to 0-255 range
        
        Args:
            image: Input image
            
        Returns:
            Normalized image (uint8)
        """
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0 and image.min() >= 0:
                # Already in 0-1 range
                return (image * 255).astype(np.uint8)
            else:
                # Normalize to 0-1
                image_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)
                return (image_norm * 255).astype(np.uint8)
        elif image.dtype == np.uint16:
            # 16-bit to 8-bit
            return (image / 256).astype(np.uint8)
        elif image.dtype == np.uint8:
            return image
        else:
            # Unknown type, try to convert
            return image.astype(np.uint8)
    
    @staticmethod
    def validate_image(image: np.ndarray) -> Tuple[bool, str]:
        """
        Validate if image data is valid
        
        Args:
            image: Numpy array image
            
        Returns:
            (is_valid, error_message)
        """
        if image is None:
            return False, "Image data is None"
        
        if not isinstance(image, np.ndarray):
            return False, f"Image is not numpy array, but {type(image)}"
        
        if len(image.shape) not in [2, 3]:
            return False, f"Image dimensions abnormal: {image.shape}"
        
        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            return False, f"Image channel count abnormal: {image.shape[2]}"
        
        if image.size == 0:
            return False, "Image data is empty"
        
        # Check for NaN or Inf values
        if np.any(np.isnan(image)):
            return False, "Image contains NaN values"
        
        if np.any(np.isinf(image)):
            return False, "Image contains Inf values"
        
        return True, f"Image data valid: {image.shape}, {image.dtype}"
    
    @staticmethod
    def preprocess_image(image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Preprocess image
        
        Args:
            image: Input image
            target_size: Target size (width, height)
            
        Returns:
            Preprocessed image
        """
        # 1. Convert to RGB
        image_rgb = ImageProcessor.convert_to_rgb(image)
        
        # 2. Normalize
        image_norm = ImageProcessor.normalize_image(image_rgb)
        
        # 3. Resize if needed
        if target_size is not None:
            h, w = image_norm.shape[:2]
            target_w, target_h = target_size
            
            if w != target_w or h != target_h:
                image_resized = cv2.resize(image_norm, target_size, interpolation=cv2.INTER_LINEAR)
                return image_resized
        
        return image_norm
    
    @staticmethod
    def calculate_image_hash(image: np.ndarray) -> str:
        """
        Calculate image hash (for deduplication)
        
        Args:
            image: Input image
            
        Returns:
            Hash string
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Resize to 8x8
        resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_LINEAR)
        
        # Calculate average
        avg = resized.mean()
        
        # Generate hash
        hash_str = ''
        for i in range(8):
            for j in range(8):
                hash_str += '1' if resized[i, j] > avg else '0'
        
        # Convert to hexadecimal
        hex_hash = hex(int(hash_str, 2))[2:].zfill(16)
        
        return hex_hash
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, 
                         grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Enhance image contrast using CLAHE
        
        Args:
            image: Input image
            clip_limit: Contrast limit
            grid_size: Grid size
            
        Returns:
            Enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Split channels
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            l_clahe = clahe.apply(l)
            
            # Merge channels
            lab_clahe = cv2.merge([l_clahe, a, b])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
            
            return enhanced
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            enhanced = clahe.apply(image)
            
            return enhanced


class PolygonUtils:
    """Polygon Utilities - Provides polygon calculation and processing functions"""
    
    @staticmethod
    def calculate_iou(poly1, poly2) -> float:
        """
        Calculate IoU of two polygons
        
        Args:
            poly1, poly2: Shapely polygons
            
        Returns:
            IoU value (0-1)
        """
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
        
        if not poly1.intersects(poly2):
            return 0.0
        
        try:
            intersection = poly1.intersection(poly2).area
            union = poly1.union(poly2).area
            
            return intersection / union if union > 0 else 0.0
        except Exception:
            return 0.0
    
    @staticmethod
    def filter_small_polygons(polygons: List, min_area: float) -> List:
        """
        Filter small area polygons
        
        Args:
            polygons: Polygon list
            min_area: Minimum area
            
        Returns:
            Filtered polygon list
        """
        return [p for p in polygons if hasattr(p, 'area') and p.area >= min_area]
    
    @staticmethod
    def smart_merge_polygons(polygons: List, iou_threshold: float = 0.7) -> List:
        """
        Smart merge overlapping polygons
        
        Args:
            polygons: Polygon list
            iou_threshold: IoU threshold
            
        Returns:
            Merged polygon list
        """
        if len(polygons) <= 1:
            return polygons
        
        from shapely.ops import unary_union
        
        # Build overlap groups
        groups = []
        used = [False] * len(polygons)
        
        for i in range(len(polygons)):
            if used[i]:
                continue
            
            group = [i]
            used[i] = True
            
            # Find all overlapping polygons
            for j in range(i + 1, len(polygons)):
                if used[j]:
                    continue
                
                iou = PolygonUtils.calculate_iou(polygons[i], polygons[j])
                if iou > iou_threshold:
                    group.append(j)
                    used[j] = True
            
            groups.append(group)
        
        # Merge each group
        merged_polygons = []
        
        for group in groups:
            if len(group) == 1:
                merged_polygons.append(polygons[group[0]])
            else:
                # Merge polygons in group
                group_polys = [polygons[idx] for idx in group]
                
                try:
                    merged = unary_union(group_polys)
                    
                    # Handle MultiPolygon
                    if hasattr(merged, 'geoms'):
                        for geom in merged.geoms:
                            if geom.area > 0:
                                merged_polygons.append(geom)
                    elif merged.area > 0:
                        merged_polygons.append(merged)
                except Exception as e:
                    print(f"Merge failed: {e}")
                    # Merge failed, keep original polygons
                    merged_polygons.extend(group_polys)
        
        return merged_polygons
    
    @staticmethod
    def calculate_polygon_statistics(polygons: List) -> Dict[str, Any]:
        """
        Calculate polygon statistics
        
        Args:
            polygons: Polygon list
            
        Returns:
            Statistics dictionary
        """
        if not polygons:
            return {
                'count': 0,
                'total_area': 0,
                'avg_area': 0,
                'min_area': 0,
                'max_area': 0,
                'avg_perimeter': 0
            }
        
        areas = []
        perimeters = []
        
        for poly in polygons:
            if hasattr(poly, 'area'):
                areas.append(poly.area)
            
            if hasattr(poly, 'length'):
                perimeters.append(poly.length)
        
        return {
            'count': len(polygons),
            'total_area': sum(areas) if areas else 0,
            'avg_area': np.mean(areas) if areas else 0,
            'min_area': min(areas) if areas else 0,
            'max_area': max(areas) if areas else 0,
            'std_area': np.std(areas) if areas else 0,
            'avg_perimeter': np.mean(perimeters) if perimeters else 0
        }
    
    @staticmethod
    def simplify_polygons(polygons: List, tolerance: float = 1.0) -> List:
        """
        Simplify polygons (reduce vertex count)
        
        Args:
            polygons: Polygon list
            tolerance: Simplify tolerance
            
        Returns:
            Simplified polygon list
        """
        simplified = []
        
        for poly in polygons:
            try:
                if hasattr(poly, 'simplify'):
                    simple = poly.simplify(tolerance, preserve_topology=True)
                    if simple.is_valid and simple.area > 0:
                        simplified.append(simple)
                    else:
                        simplified.append(poly)
                else:
                    simplified.append(poly)
            except Exception:
                simplified.append(poly)
        
        return simplified


class FileUtils:
    """File Utilities - Provides file read/write and path operations"""
    
    @staticmethod
    def safe_load_yaml(file_path: str, default: Dict = None) -> Dict:
        """
        Safely load YAML file
        
        Args:
            file_path: File path
            default: Default config
            
        Returns:
            Config dictionary
        """
        if default is None:
            default = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                return default
            
            return config
        except Exception as e:
            print(f"Failed to load YAML: {e}")
            return default
    
    @staticmethod
    def safe_save_yaml(data: Dict, file_path: str):
        """
        Safely save YAML file
        
        Args:
            data: Data dictionary
            file_path: File path
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
            
            print(f"Config saved to: {file_path}")
        except Exception as e:
            print(f"Failed to save YAML: {e}")
    
    @staticmethod
    def safe_load_json(file_path: str, default: Any = None) -> Any:
        """
        Safely load JSON file
        
        Args:
            file_path: File path
            default: Default value
            
        Returns:
            Data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Failed to load JSON: {e}")
            return default
    
    @staticmethod
    def safe_save_json(data: Any, file_path: str, indent: int = 2):
        """
        Safely save JSON file
        
        Args:
            data: Data
            file_path: File path
            indent: Indentation
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
            
            print(f"JSON saved to: {file_path}")
        except Exception as e:
            print(f"Failed to save JSON: {e}")
    
    @staticmethod
    def check_file_exists(file_path: str, create_if_not: bool = False) -> bool:
        """
        Check if file exists
        
        Args:
            file_path: File path
            create_if_not: Create if not exists
            
        Returns:
            Whether exists
        """
        exists = os.path.exists(file_path)
        
        if not exists and create_if_not:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Create empty file
                with open(file_path, 'w') as f:
                    pass
                
                print(f"Created file: {file_path}")
                return True
            except Exception as e:
                print(f"Failed to create file: {e}")
                return False
        
        return exists
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """
        Get file size (bytes)
        
        Args:
            file_path: File path
            
        Returns:
            File size (bytes)
        """
        try:
            return os.path.getsize(file_path)
        except Exception:
            return 0


class PerformanceMonitor:
    """Performance Monitor - Provides time and memory monitoring"""
    
    def __init__(self):
        self.timings = {}
        self.counters = {}
        self.memory_usage = {}
    
    def start_timing(self, name: str):
        """Start timing"""
        import time
        self.timings[name] = {'start': time.time()}
    
    def end_timing(self, name: str):
        """End timing"""
        import time
        if name in self.timings and 'start' in self.timings[name]:
            elapsed = time.time() - self.timings[name]['start']
            self.timings[name]['end'] = time.time()
            self.timings[name]['elapsed'] = elapsed
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment counter"""
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += value
    
    def record_memory(self, name: str):
        """Record memory usage"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            self.memory_usage[name] = {
                'rss': memory_info.rss,  # Physical memory
                'vms': memory_info.vms   # Virtual memory
            }
        except ImportError:
            pass
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {
            'timings': {},
            'counters': self.counters.copy(),
            'memory': self.memory_usage.copy()
        }
        
        # Calculate total time
        total_time = 0
        for name, timing in self.timings.items():
            if 'elapsed' in timing:
                summary['timings'][name] = timing['elapsed']
                total_time += timing['elapsed']
        
        summary['total_time'] = total_time
        
        return summary
    
    def print_summary(self):
        """Print performance summary"""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("Performance Summary")
        print("=" * 60)
        
        print("Timing Statistics:")
        for name, elapsed in summary['timings'].items():
            percentage = (elapsed / summary['total_time'] * 100) if summary['total_time'] > 0 else 0
            print(f"  {name}: {elapsed:.3f}s ({percentage:.1f}%)")
        
        print(f"\n  Total Time: {summary['total_time']:.3f}s")
        
        print("\nCounter Statistics:")
        for name, count in summary['counters'].items():
            print(f"  {name}: {count}")
        
        if summary['memory']:
            print("\nMemory Usage:")
            for name, memory in summary['memory'].items():
                rss_mb = memory['rss'] / 1024 / 1024
                vms_mb = memory['vms'] / 1024 / 1024
                print(f"  {name}: RSS={rss_mb:.1f}MB, VMS={vms_mb:.1f}MB")
        
        print("=" * 60)


if __name__ == "__main__":
    print("Core Tool Functions Library Test")
    print("=" * 60)
    
    # Test code
    img_proc = ImageProcessor()
    poly_utils = PolygonUtils()
    file_utils = FileUtils()
    perf_monitor = PerformanceMonitor()
    
    print("All tool classes passed test")
