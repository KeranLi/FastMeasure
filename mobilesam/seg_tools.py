"""
增强工具函数库
文件名：seg_tools.py
功能：提供强大的工具函数支持
"""

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageFile
import io
import json
import yaml
import hashlib
from typing import Tuple, Optional, Dict, Any, List, Union
import warnings
warnings.filterwarnings('ignore')

# 允许加载截断的图片
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageProcessor:
    """图像处理器"""
    
    @staticmethod
    def load_image_safely(image_path: str) -> Optional[np.ndarray]:
        """
        安全加载图像（多重回退机制）
        
        Args:
            image_path: 图像路径
            
        Returns:
            RGB图像数组，或None
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
                    # 转换为RGB格式
                    image = ImageProcessor.convert_to_rgb(image)
                    # 归一化到0-255
                    image = ImageProcessor.normalize_image(image)
                    return image
            except Exception as e:
                continue
        
        return None
    
    @staticmethod
    def _load_with_pil(image_path: str) -> Optional[np.ndarray]:
        """使用PIL加载"""
        try:
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            return np.array(pil_image)
        except:
            return None
    
    @staticmethod
    def _load_with_cv2(image_path: str) -> Optional[np.ndarray]:
        """使用OpenCV加载"""
        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is not None:
                return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except:
            return None
    
    @staticmethod
    def _load_with_skimage(image_path: str) -> Optional[np.ndarray]:
        """使用skimage加载"""
        try:
            from skimage import io
            image = io.imread(image_path)
            return image
        except:
            return None
    
    @staticmethod
    def _load_with_binary(image_path: str) -> Optional[np.ndarray]:
        """使用二进制加载"""
        try:
            with open(image_path, 'rb') as f:
                data = f.read()
                pil_image = Image.open(io.BytesIO(data))
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                return np.array(pil_image)
        except:
            return None
    
    @staticmethod
    def convert_to_rgb(image: np.ndarray) -> np.ndarray:
        """
        将图像转换为RGB格式
        
        Args:
            image: 输入图像
            
        Returns:
            RGB格式图像
        """
        if len(image.shape) == 2:
            # 灰度转RGB
            return np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 1:
            # 单通道转RGB
            return np.repeat(image, 3, axis=2)
        elif image.shape[2] == 4:
            # RGBA转RGB
            return image[:, :, :3]
        elif image.shape[2] == 3:
            return image
        else:
            raise ValueError(f"不支持的通道数: {image.shape[2]}")
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        归一化图像到0-255范围
        
        Args:
            image: 输入图像
            
        Returns:
            归一化后的图像 (uint8)
        """
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0 and image.min() >= 0:
                # 已经是0-1范围
                return (image * 255).astype(np.uint8)
            else:
                # 归一化到0-1
                image_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)
                return (image_norm * 255).astype(np.uint8)
        elif image.dtype == np.uint16:
            # 16位转8位
            return (image / 256).astype(np.uint8)
        elif image.dtype == np.uint8:
            return image
        else:
            # 未知类型，尝试转换
            return image.astype(np.uint8)
    
    @staticmethod
    def validate_image(image: np.ndarray) -> Tuple[bool, str]:
        """
        验证图像数据是否有效
        
        Args:
            image: numpy数组图像
            
        Returns:
            (是否有效, 错误信息)
        """
        if image is None:
            return False, "图像数据为None"
        
        if not isinstance(image, np.ndarray):
            return False, f"图像不是numpy数组，而是{type(image)}"
        
        if len(image.shape) not in [2, 3]:
            return False, f"图像维度异常: {image.shape}"
        
        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            return False, f"图像通道数异常: {image.shape[2]}"
        
        if image.size == 0:
            return False, "图像数据为空"
        
        # 检查NaN或Inf值
        if np.any(np.isnan(image)):
            return False, "图像包含NaN值"
        
        if np.any(np.isinf(image)):
            return False, "图像包含Inf值"
        
        return True, f"图像数据有效: {image.shape}, {image.dtype}"
    
    @staticmethod
    def preprocess_image(image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        预处理图像
        
        Args:
            image: 输入图像
            target_size: 目标尺寸 (宽, 高)
            
        Returns:
            预处理后的图像
        """
        # 1. 转换为RGB
        image_rgb = ImageProcessor.convert_to_rgb(image)
        
        # 2. 归一化
        image_norm = ImageProcessor.normalize_image(image_rgb)
        
        # 3. 调整尺寸（如果需要）
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
        计算图像哈希值（用于去重）
        
        Args:
            image: 输入图像
            
        Returns:
            哈希字符串
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 调整大小为8x8
        resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_LINEAR)
        
        # 计算平均值
        avg = resized.mean()
        
        # 生成哈希
        hash_str = ''
        for i in range(8):
            for j in range(8):
                hash_str += '1' if resized[i, j] > avg else '0'
        
        # 转换为十六进制
        hex_hash = hex(int(hash_str, 2))[2:].zfill(16)
        
        return hex_hash
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        增强图像对比度（使用CLAHE）
        
        Args:
            image: 输入图像
            clip_limit: 对比度限制
            grid_size: 网格大小
            
        Returns:
            增强后的图像
        """
        if len(image.shape) == 3:
            # 转换为LAB颜色空间
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # 分离通道
            l, a, b = cv2.split(lab)
            
            # 对L通道应用CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            l_clahe = clahe.apply(l)
            
            # 合并通道
            lab_clahe = cv2.merge([l_clahe, a, b])
            
            # 转换回RGB
            enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
            
            return enhanced
        else:
            # 灰度图像
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            enhanced = clahe.apply(image)
            
            return enhanced


class PolygonUtils:
    """多边形工具类"""
    
    @staticmethod
    def calculate_iou(poly1, poly2) -> float:
        """
        计算两个多边形的IoU
        
        Args:
            poly1, poly2: Shapely多边形
            
        Returns:
            IoU值 (0-1)
        """
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
        
        if not poly1.intersects(poly2):
            return 0.0
        
        try:
            intersection = poly1.intersection(poly2).area
            union = poly1.union(poly2).area
            
            return intersection / union if union > 0 else 0.0
        except:
            return 0.0
    
    @staticmethod
    def filter_small_polygons(polygons: List, min_area: float) -> List:
        """
        过滤小面积多边形
        
        Args:
            polygons: 多边形列表
            min_area: 最小面积
            
        Returns:
            过滤后的多边形列表
        """
        return [p for p in polygons if hasattr(p, 'area') and p.area >= min_area]
    
    @staticmethod
    def smart_merge_polygons(polygons: List, iou_threshold: float = 0.7) -> List:
        """
        智能合并重叠多边形
        
        Args:
            polygons: 多边形列表
            iou_threshold: IoU阈值
            
        Returns:
            合并后的多边形列表
        """
        if len(polygons) <= 1:
            return polygons
        
        from shapely.ops import unary_union
        
        # 构建重叠组
        groups = []
        used = [False] * len(polygons)
        
        for i in range(len(polygons)):
            if used[i]:
                continue
            
            group = [i]
            used[i] = True
            
            # 查找所有重叠的多边形
            for j in range(i + 1, len(polygons)):
                if used[j]:
                    continue
                
                iou = PolygonUtils.calculate_iou(polygons[i], polygons[j])
                if iou > iou_threshold:
                    group.append(j)
                    used[j] = True
            
            groups.append(group)
        
        # 合并每个组
        merged_polygons = []
        
        for group in groups:
            if len(group) == 1:
                merged_polygons.append(polygons[group[0]])
            else:
                # 合并组内的多边形
                group_polys = [polygons[idx] for idx in group]
                
                try:
                    merged = unary_union(group_polys)
                    
                    # 处理多重多边形
                    if hasattr(merged, 'geoms'):
                        for geom in merged.geoms:
                            if geom.area > 0:
                                merged_polygons.append(geom)
                    elif merged.area > 0:
                        merged_polygons.append(merged)
                except Exception as e:
                    print(f"合并失败: {e}")
                    # 合并失败，保留原始多边形
                    merged_polygons.extend(group_polys)
        
        return merged_polygons
    
    @staticmethod
    def calculate_polygon_statistics(polygons: List) -> Dict[str, Any]:
        """
        计算多边形统计信息
        
        Args:
            polygons: 多边形列表
            
        Returns:
            统计字典
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
        简化多边形（减少顶点数）
        
        Args:
            polygons: 多边形列表
            tolerance: 简化容差
            
        Returns:
            简化后的多边形列表
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
            except:
                simplified.append(poly)
        
        return simplified


class FileUtils:
    """文件工具类"""
    
    @staticmethod
    def safe_load_yaml(file_path: str, default: Dict = None) -> Dict:
        """
        安全加载YAML文件
        
        Args:
            file_path: 文件路径
            default: 默认配置
            
        Returns:
            配置字典
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
            print(f"加载YAML失败: {e}")
            return default
    
    @staticmethod
    def safe_save_yaml(data: Dict, file_path: str):
        """
        安全保存YAML文件
        
        Args:
            data: 数据字典
            file_path: 文件路径
        """
        try:
            # 确保目录存在
            import os
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
            
            print(f"配置保存到: {file_path}")
        except Exception as e:
            print(f"保存YAML失败: {e}")
    
    @staticmethod
    def safe_load_json(file_path: str, default: Any = None) -> Any:
        """
        安全加载JSON文件
        
        Args:
            file_path: 文件路径
            default: 默认值
            
        Returns:
            数据
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"加载JSON失败: {e}")
            return default
    
    @staticmethod
    def safe_save_json(data: Any, file_path: str, indent: int = 2):
        """
        安全保存JSON文件
        
        Args:
            data: 数据
            file_path: 文件路径
            indent: 缩进
        """
        try:
            # 确保目录存在
            import os
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
            
            print(f"JSON保存到: {file_path}")
        except Exception as e:
            print(f"保存JSON失败: {e}")
    
    @staticmethod
    def check_file_exists(file_path: str, create_if_not: bool = False) -> bool:
        """
        检查文件是否存在
        
        Args:
            file_path: 文件路径
            create_if_not: 如果不存在是否创建
            
        Returns:
            是否存在
        """
        import os
        exists = os.path.exists(file_path)
        
        if not exists and create_if_not:
            try:
                # 确保目录存在
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # 创建空文件
                with open(file_path, 'w') as f:
                    pass
                
                print(f"创建文件: {file_path}")
                return True
            except Exception as e:
                print(f" 创建文件失败: {e}")
                return False
        
        return exists
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """
        获取文件大小（字节）
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件大小（字节）
        """
        import os
        try:
            return os.path.getsize(file_path)
        except:
            return 0


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.timings = {}
        self.counters = {}
        self.memory_usage = {}
        
    def start_timing(self, name: str):
        """开始计时"""
        import time
        self.timings[name] = {'start': time.time()}
    
    def end_timing(self, name: str):
        """结束计时"""
        import time
        if name in self.timings and 'start' in self.timings[name]:
            elapsed = time.time() - self.timings[name]['start']
            self.timings[name]['end'] = time.time()
            self.timings[name]['elapsed'] = elapsed
    
    def increment_counter(self, name: str, value: int = 1):
        """增加计数器"""
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += value
    
    def record_memory(self, name: str):
        """记录内存使用"""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        self.memory_usage[name] = {
            'rss': memory_info.rss,  # 实际物理内存
            'vms': memory_info.vms   # 虚拟内存
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {
            'timings': {},
            'counters': self.counters.copy(),
            'memory': self.memory_usage.copy()
        }
        
        # 计算总时间
        total_time = 0
        for name, timing in self.timings.items():
            if 'elapsed' in timing:
                summary['timings'][name] = timing['elapsed']
                total_time += timing['elapsed']
        
        summary['total_time'] = total_time
        
        return summary
    
    def print_summary(self):
        """打印性能摘要"""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("性能监控摘要")
        print("=" * 60)
        
        print("时间统计:")
        for name, elapsed in summary['timings'].items():
            percentage = (elapsed / summary['total_time'] * 100) if summary['total_time'] > 0 else 0
            print(f"  {name}: {elapsed:.3f}s ({percentage:.1f}%)")
        
        print(f"\n  总时间: {summary['total_time']:.3f}s")
        
        print("\n计数器统计:")
        for name, count in summary['counters'].items():
            print(f"  {name}: {count}")
        
        print("\n内存使用:")
        for name, memory in summary['memory'].items():
            rss_mb = memory['rss'] / 1024 / 1024
            vms_mb = memory['vms'] / 1024 / 1024
            print(f"  {name}: RSS={rss_mb:.1f}MB, VMS={vms_mb:.1f}MB")
        
        print("=" * 60)


if __name__ == "__main__":
    print("增强工具函数库测试")
    print("=" * 60)
    
    # 测试代码
    img_proc = ImageProcessor()
    poly_utils = PolygonUtils()
    file_utils = FileUtils()
    perf_monitor = PerformanceMonitor()
    
    print("所有工具类测试通过")