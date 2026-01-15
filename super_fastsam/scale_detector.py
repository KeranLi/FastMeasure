"""
比例尺检测模块 -
文件名：scale_detector.py
功能：检测岩石显微图像中的红色比例尺条，计算每个像素对应的微米数
"""

import cv2
import numpy as np
from pathlib import Path
import urllib.parse

class ScaleDetector:
    """比例尺检测器"""
    
    def __init__(self, config):
        """
        初始化比例尺检测器
        
        Args:
            config: 配置字典，包含scale_detection配置节
        """
        self.config = config.get('scale_detection', {})
        
        # 已知比例尺的实际长度（微米）
        self.known_length_um = self.config.get('known_length_um', 1000.0)
        
        # 检测参数
        self.detection_params = self.config.get('detection_params', {})
        
        # 红色阈值（默认值，会被配置文件覆盖）
        self.red_lower1 = np.array(self.detection_params.get('red_lower1', [0, 120, 120]))
        self.red_upper1 = np.array(self.detection_params.get('red_upper1', [10, 255, 255]))
        self.red_lower2 = np.array(self.detection_params.get('red_lower2', [160, 120, 120]))
        self.red_upper2 = np.array(self.detection_params.get('red_upper2', [180, 255, 255]))
        
        # 裁剪参数
        self.crop_height = self.detection_params.get('crop_height', 220)
        self.crop_width = self.detection_params.get('crop_width', 600)
        self.search_margin = self.detection_params.get('search_margin', 80)
        
        # 筛选参数
        self.min_aspect_ratio = self.detection_params.get('min_aspect_ratio', 8)
        self.min_horizontal_score = self.detection_params.get('min_horizontal_score', 0.6)
    
    def cv_imread(self, file_path):
        """增强的图片读取函数，支持中文路径"""
        try:
            # 处理中文路径
            file_path = urllib.parse.quote(file_path, safe=':/\\')
            img_arr = np.fromfile(file_path, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f" 读取图片失败: {e}")
            return None
    
    def extract_horizontal_line(self, mask, contour, roi_y_start, roi_x_start):
        """从轮廓中精确提取水平线段"""
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # 水平投影分析
        horizontal_projection = np.sum(contour_mask, axis=1)
        if len(horizontal_projection) == 0:
            return None, None
        
        max_row = np.argmax(horizontal_projection)
        
        # 提取线段所在行附近区域
        row_start = max(0, max_row - 2)
        row_end = min(contour_mask.shape[0], max_row + 3)
        line_region = contour_mask[row_start:row_end, :]
        
        if np.any(line_region):
            column_projection = np.sum(line_region, axis=0)
            non_zero_cols = np.where(column_projection > 0)[0]
            
            if len(non_zero_cols) > 0:
                leftmost = non_zero_cols[0]
                rightmost = non_zero_cols[-1]
                precise_length = rightmost - leftmost
                
                line_info = {
                    'left': leftmost,
                    'right': rightmost,
                    'row': max_row,
                    'length': precise_length
                }
                return precise_length, line_info
        
        return None, None
    
    def detect(self, image_path, known_length_um=None):
        """
        检测图片中的比例尺并计算比例系数
        
        Args:
            image_path: 图片路径
            known_length_um: 可选，已知的实际长度（微米），如不指定则使用配置中的默认值
            
        Returns:
            scale_factor: 比例因子（μm/px），即每个像素对应的微米数
            success: 是否成功检测
        """
        if known_length_um is None:
            known_length_um = self.known_length_um
        
        # 读取图片
        img = self.cv_imread(image_path)
        if img is None:
            return None, False
        
        height, width = img.shape[:2]
        
        # 智能定位裁剪区域（右下角）
        search_roi = img[height-(self.crop_height+self.search_margin):height, 
                        width-(self.crop_width+self.search_margin):width]
        
        if search_roi.size == 0:
            return None, False
        
        # HSV颜色空间转换
        search_hsv = cv2.cvtColor(search_roi, cv2.COLOR_BGR2HSV)
        
        # 创建红色掩码
        mask1 = cv2.inRange(search_hsv, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(search_hsv, self.red_lower2, self.red_upper2)
        red_mask_search = cv2.bitwise_or(mask1, mask2)
        
        # 垂直投影分析，找到红色区域最集中的行
        vertical_projection = np.sum(red_mask_search, axis=1)
        if len(vertical_projection) > 0:
            max_row_in_search = np.argmax(vertical_projection)
        else:
            max_row_in_search = self.crop_height // 2
        
        # 计算最终裁剪区域
        target_center_y = height - (self.crop_height + self.search_margin) + max_row_in_search
        y_start = max(target_center_y - self.crop_height // 3, height - self.crop_height * 3)
        y_start = max(y_start, 0)
        y_end = min(y_start + self.crop_height, height)
        
        x_start = max(width - self.crop_width, 0)
        x_end = width
        
        # 最终裁剪
        roi = img[y_start:y_end, x_start:x_end]
        if roi.size == 0:
            return None, False
        
        # 在裁剪区域内检测
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # 形态学处理
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.dilate(cleaned, kernel, iterations=1)
        
        # 查找轮廓
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, False
        
        # 筛选最佳轮廓
        scale_contour = None
        best_score = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect
            actual_width = max(w, h)
            actual_height = min(w, h)
            
            if actual_height == 0:
                continue
            
            aspect_ratio = actual_width / actual_height
            abs_angle = abs(angle)
            if abs_angle > 45:
                abs_angle = 90 - abs_angle
            
            horizontal_score = 1.0 - min(abs_angle / 45.0, 1.0)
            
            # 筛选条件：长宽比>最小要求，水平度>最小要求
            if (aspect_ratio > self.min_aspect_ratio and 
                horizontal_score > self.min_horizontal_score):
                
                combined_score = aspect_ratio * 0.5 + horizontal_score * 0.5
                
                if combined_score > best_score:
                    best_score = combined_score
                    scale_contour = {
                        'width': actual_width,
                        'aspect_ratio': aspect_ratio,
                        'rect': rect,
                        'contour': cnt
                    }
        
        # 精确测量线段长度
        precise_length = None
        line_info = None
        
        if scale_contour:
            precise_length, line_info = self.extract_horizontal_line(
                cleaned, scale_contour['contour'], y_start, x_start
            )
        
        if precise_length:
            scale_length_px = precise_length
        elif scale_contour:
            scale_length_px = scale_contour['width']
        else:
            # 使用最大轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            precise_length, _ = self.extract_horizontal_line(
                cleaned, largest_contour, y_start, x_start
            )
            if precise_length:
                scale_length_px = precise_length
            else:
                rect = cv2.minAreaRect(largest_contour)
                scale_length_px = max(rect[1])
        
        # 计算比例系数（核心公式）
        scale_factor = known_length_um / scale_length_px
        
        return scale_factor, True