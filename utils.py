"""
工具函数 - 辅助函数库
文件名：utils.py
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from PIL import Image, ImageFile
import io

# 允许加载截断的图片
ImageFile.LOAD_TRUNCATED_IMAGES = True


def check_image_file_pro(image_path: str) -> Tuple[bool, str]:
    """
    专业级图片文件检查
    
    Args:
        image_path: 图片路径
        
    Returns:
        (是否有效, 错误信息)
    """
    try:
        # 基本检查
        with Image.open(image_path) as img:
            img.verify()
            return True, "图片文件有效"
    except Exception as e:
        # 尝试多重加载方法
        return _try_multiple_loading_methods(image_path)


def _try_multiple_loading_methods(image_path: str) -> Tuple[bool, str]:
    """尝试多种图片加载方法"""
    loading_methods = [
        _try_skimage_load,
        _try_pil_load,
        _try_opencv_load,
        _try_binary_load
    ]
    
    for method in loading_methods:
        success, message = method(image_path)
        if success:
            return True, f"通过{message}加载成功"
    
    return False, "所有加载方法都失败"


def _try_skimage_load(image_path: str) -> Tuple[bool, str]:
    """尝试skimage加载"""
    try:
        from skimage import io
        image = io.imread(image_path)
        if image is not None:
            return True, "skimage"
    except:
        pass
    return False, ""


def _try_pil_load(image_path: str) -> Tuple[bool, str]:
    """尝试PIL加载"""
    try:
        img = Image.open(image_path)
        img.load()  # 强制加载
        return True, "PIL"
    except:
        pass
    return False, ""


def _try_opencv_load(image_path: str) -> Tuple[bool, str]:
    """尝试OpenCV加载"""
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is not None:
            return True, "OpenCV"
    except:
        pass
    return False, ""


def _try_binary_load(image_path: str) -> Tuple[bool, str]:
    """尝试二进制加载"""
    try:
        with open(image_path, 'rb') as f:
            data = f.read()
            img = Image.open(io.BytesIO(data))
            img.load()
            return True, "二进制"
    except:
        pass
    return False, ""


def validate_image_data(image: np.ndarray) -> Tuple[bool, str]:
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


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    归一化图像到0-255范围
    
    Args:
        image: 输入图像
        
    Returns:
        归一化后的图像
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0 and image.min() >= 0:
            # 已经是0-1范围
            return (image * 255).astype(np.uint8)
        else:
            # 归一化到0-1
            image_norm = (image - image.min()) / (image.max() - image.min())
            return (image_norm * 255).astype(np.uint8)
    elif image.dtype == np.uint16:
        # 16位转8位
        return (image / 256).astype(np.uint8)
    elif image.dtype == np.uint8:
        return image
    else:
        # 未知类型，尝试转换
        return image.astype(np.uint8)


def calculate_iou(poly1, poly2) -> float:
    """
    计算两个多边形的IoU
    
    Args:
        poly1, poly2: Shapely多边形
        
    Returns:
        IoU值
    """
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    
    if not poly1.intersects(poly2):
        return 0.0
    
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    
    return intersection / union if union > 0 else 0.0


def filter_small_polygons(polygons: list, min_area: float) -> list:
    """
    过滤小面积多边形
    
    Args:
        polygons: 多边形列表
        min_area: 最小面积
        
    Returns:
        过滤后的多边形列表
    """
    return [p for p in polygons if p.is_valid and p.area >= min_area]


def merge_overlapping_polygons_simple(polygons: list, iou_threshold: float = 0.7) -> list:
    """
    简单合并重叠多边形
    
    Args:
        polygons: 多边形列表
        iou_threshold: IoU阈值
        
    Returns:
        合并后的多边形列表
    """
    if len(polygons) <= 1:
        return polygons
    
    filtered = []
    for i, poly1 in enumerate(polygons):
        if not poly1.is_valid:
            continue
        
        # 检查是否与已选择的颗粒高度重叠
        is_duplicate = False
        for poly2 in filtered:
            iou = calculate_iou(poly1, poly2)
            if iou > iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered.append(poly1)
    
    return filtered


if __name__ == "__main__":
    print("✅ 工具函数库加载成功")