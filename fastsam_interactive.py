"""
FastSAM增强版交互式界面模块 - 修复版
文件名：fastsam_interactive.py
功能：基于FastSAM特性的交互式分割，修复点选、卡顿、偏移问题
"""

import os
import sys
import numpy as np
import matplotlib

def setup_backend():
    """智能设置后端，优先GUI后端"""
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

# 导入FastSAM
try:
    from ultralytics import FastSAM
    FASTSAM_AVAILABLE = True
    print("使用Ultralytics FastSAM")
except ImportError as e:
    FASTSAM_AVAILABLE = False
    print(f"FastSAM库未安装: {e}")

# 强制导入项目一函数
print("=" * 60)
print("配置：强制使用项目一函数")
print("=" * 60)

import sys
from pathlib import Path

# 获取项目一模块路径
current_file = Path(__file__).resolve()
project_root = current_file.parent  # 修改：只需要一个parent，因为文件在项目根目录
project1_dir = project_root / "segmenteverygrain"

# 添加到Python路径
if str(project1_dir) not in sys.path:
    sys.path.insert(0, str(project1_dir))

try:
    from segmenteverygrain import (
        create_labeled_image,
        plot_image_w_colorful_grains,
        plot_grain_axes_and_centroids,
        find_connected_components,
        merge_overlapping_polygons
    )
    PROJECT1_AVAILABLE = True
    print("成功导入项目一计算函数")
    print(f"   模块位置: {project1_dir}")
except ImportError as e:
    print(f"导入项目一计算函数失败: {e}")
    print("程序需要项目一函数才能运行")
    sys.exit(1)

# 导入几何计算模块
try:
    # 由于fastsam_interactive.py在项目根目录，geometry模块路径需要调整
    geometry_dir = project_root / "geometry"
    if str(geometry_dir) not in sys.path:
        sys.path.insert(0, str(geometry_dir))
    
    from geometry.grain_metric import GrainShapeMetrics
    from geometry.config_loader import load_geometry_config
    from geometry.export_csv import select_columns_for_grain_statistics_csv
    
    GEOMETRY_AVAILABLE = True
    print("geometry模块加载成功")
except ImportError as e:
    GEOMETRY_AVAILABLE = False
    print(f"geometry模块不可用: {e}")

print("=" * 60)


class PureFastSAMInteractiveEnhanced:
    """增强版交互式FastSAM - 修复版"""
    
    def __init__(self, model_path: str = "models/FastSAM-s.pt", 
                 device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        
        self.image = None
        self.image_path = None
        self.fastsam_model = None
        self.model_loaded = False
        
        # 全图推理结果缓存
        self.global_results = None
        self.all_masks_cache = []
        self.all_masks_scores = []
        
        # 交互状态
        self.grains = []
        self.current_grain_id = 0
        self.drawing_box = False
        self.box_start = None
        self.box_end = None
        self.current_box = None
        
        # 优化：减少重绘
        self.last_draw_time = 0
        self.draw_interval = 0.05  # 50ms，避免频繁重绘
        
        # 结果存储
        self.polygons = []
        self.labels = None
        self.mask_all = None
        self.grain_data = None
        
        # 显示相关
        self.fig = None
        self.ax = None
        self.grain_patches = {}
        self.box_artist = None
        self.grain_texts = {}
        
        # 输出目录
        self.output_dir = Path("interactive_fastsam_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # 几何配置
        self.geometry_config = None
        if GEOMETRY_AVAILABLE:
            try:
                # 修改点：调整配置文件的路径
                config_path = Path(__file__).parent / "geometry_config.yaml"
                if config_path.exists():
                    self.geometry_config = load_geometry_config(str(config_path))
                    print("geometry配置加载成功")
                else:
                    # 尝试在fastsam子目录查找
                    alt_config_path = Path(__file__).parent / "fastsam" / "geometry_config.yaml"
                    if alt_config_path.exists():
                        self.geometry_config = load_geometry_config(str(alt_config_path))
                        print("从fastsam子目录加载geometry配置成功")
            except Exception as e:
                print(f"加载geometry配置失败: {e}")
        
        # 性能统计
        self.start_time = None
        self.total_grains = 0
        self.total_interactions = 0
        
        self.gui_running = False
        
        print("=" * 70)
        print("FastSAM交互式系统（修复版）")
        print("=" * 70)
        
        self._load_fastsam_model()
    
    def _load_fastsam_model(self) -> bool:
        """加载FastSAM模型"""
        if not FASTSAM_AVAILABLE:
            print("FastSAM库不可用")
            return False
        
        try:
            print(f"加载FastSAM模型: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                print(f"模型文件不存在: {self.model_path}")
                return False
            
            # 加载FastSAM模型
            self.fastsam_model = FastSAM(self.model_path)
            self.fastsam_model.to(self.device)
            self.model_loaded = True
            
            print(f"FastSAM模型加载成功 (设备: {self.device})")
            return True
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            traceback.print_exc()
            return False
    
    def _run_global_inference(self):
        """运行全图推理并缓存结果"""
        if self.image is None or not self.model_loaded:
            return False
        
        try:
            print("运行全图推理...")
            start_time = time.time()
            
            # 运行FastSAM全图推理
            results = self.fastsam_model(
                self.image,
                device=self.device,
                imgsz=1024,
                conf=0.25,
                iou=0.3,
                verbose=False
            )
            
            if len(results) == 0 or results[0].masks is None:
                print("全图推理未生成掩码")
                return False
            
            # 缓存结果
            self.global_results = results[0]
            masks_data = results[0].masks.data.cpu().numpy()
            
            # 处理并缓存所有掩码
            self.all_masks_cache = []
            self.all_masks_scores = []
            
            h, w = self.image.shape[:2]
            
            for idx, mask in enumerate(masks_data):
                binary_mask = (mask > 0).astype(np.uint8)
                
                # 过滤小掩码
                if np.sum(binary_mask) < 10:
                    continue
                
                # 确保掩码尺寸正确
                if binary_mask.shape[0] != h or binary_mask.shape[1] != w:
                    binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # 形态学增强
                enhanced_mask = self._enhance_mask_morphology(binary_mask)
                
                # 计算质量分数
                score = self._calculate_mask_quality(enhanced_mask)
                
                self.all_masks_cache.append(enhanced_mask)
                self.all_masks_scores.append(score)
            
            inference_time = time.time() - start_time
            print(f"全图推理完成: {len(self.all_masks_cache)}个候选掩码，耗时: {inference_time:.2f}s")
            return True
            
        except Exception as e:
            print(f"全图推理失败: {e}")
            return False
    
    def _enhance_mask_morphology(self, mask: np.ndarray) -> np.ndarray:
        """使用形态学操作增强掩码"""
        # 闭运算填充小孔洞
        kernel_close = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        
        # 开运算去除小噪点
        kernel_open = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # 填充孔洞
        from scipy import ndimage
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
        
        return mask
    
    def _calculate_mask_quality(self, mask: np.ndarray) -> float:
        """计算掩码质量分数（0-1）"""
        if mask.sum() == 0:
            return 0.0
        
        # 计算实心度
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
        """设置当前图像并运行全图推理"""
        self.image = image
        print(f"图像已设置: {image.shape}")
        
        # 运行全图推理
        self._run_global_inference()
    
    def _safe_file_dialog(self):
        """安全的文件选择对话框"""
        self.selected_file = None
        
        def run_file_dialog():
            try:
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                
                file_path = filedialog.askopenfilename(
                    title="选择岩石显微图像",
                    filetypes=[
                        ("图像文件", "*.tif *.tiff *.jpg *.jpeg *.png *.bmp"),
                        ("所有文件", "*.*")
                    ]
                )
                
                if file_path:
                    self.selected_file = file_path
                
                root.destroy()
            except Exception as e:
                print(f"文件对话框错误: {e}")
        
        dialog_thread = threading.Thread(target=run_file_dialog)
        dialog_thread.daemon = True
        dialog_thread.start()
        dialog_thread.join(timeout=30)
        
        return self.selected_file
    
    def load_image_with_gui(self) -> bool:
        """通过GUI文件选择对话框加载图片"""
        if not self.model_loaded or self.fastsam_model is None:
            print("模型未加载，无法处理图片")
            return False
        
        try:
            print("请选择岩石显微图像...")
            
            file_path = self._safe_file_dialog()
            
            if not file_path:
                print("未选择文件，退出交互模式")
                return False
            
            if not os.path.exists(file_path):
                print(f"图片文件不存在: {file_path}")
                return False
            
            print(f"加载图片: {file_path}")
            pil_image = Image.open(file_path).convert('RGB')
            self.image = np.array(pil_image)
            self.image_path = file_path
            
            self.set_image(self.image)
            
            # 重置状态
            self.grains = []
            self.current_grain_id = 0
            self.grain_patches = {}
            self.grain_texts = {}
            self.polygons = []
            self.labels = None
            self.mask_all = None
            self.grain_data = None
            
            self.start_time = time.time()
            
            print(f"图片加载成功: {self.image.shape}")
            return True
            
        except Exception as e:
            print(f"图片加载失败: {e}")
            traceback.print_exc()
            return False
    
    def load_image_from_path(self, image_path: str) -> bool:
        """直接从路径加载图片"""
        if not self.model_loaded or self.fastsam_model is None:
            print("模型未加载，无法处理图片")
            return False
        
        try:
            if not os.path.exists(image_path):
                print(f"图片文件不存在: {image_path}")
                return False
            
            print(f"加载图片: {image_path}")
            pil_image = Image.open(image_path).convert('RGB')
            self.image = np.array(pil_image)
            self.image_path = image_path
            
            self.set_image(self.image)
            
            # 重置状态
            self.grains = []
            self.current_grain_id = 0
            self.grain_patches = {}
            self.grain_texts = {}
            self.polygons = []
            self.labels = None
            self.mask_all = None
            self.grain_data = None
            
            self.start_time = time.time()
            
            print(f"图片加载成功: {self.image.shape}")
            return True
            
        except Exception as e:
            print(f"图片加载失败: {e}")
            traceback.print_exc()
            return False
    
    def _find_mask_at_point(self, x: float, y: float) -> Optional[Dict]:
        """查找点击位置所在的掩码 - 修复版"""
        ix, iy = int(x), int(y)
        h, w = self.image.shape[:2]
        
        if not (0 <= ix < w and 0 <= iy < h):
            return None
        
        # 从缓存的掩码中查找
        for idx, mask in enumerate(self.all_masks_cache):
            # 确保掩码尺寸正确
            if mask.shape[0] != h or mask.shape[1] != w:
                continue
                
            if mask[iy, ix] > 0:
                return {
                    'mask': mask,
                    'score': self.all_masks_scores[idx] if idx < len(self.all_masks_scores) else 0.5,
                    'index': idx
                }
        
        # 如果没有找到，尝试在点周围进行小范围搜索
        search_radius = 5
        for idx, mask in enumerate(self.all_masks_cache):
            # 检查点周围区域
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
        """查找框内的所有掩码"""
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        masks_in_box = []
        
        for idx, mask in enumerate(self.all_masks_cache):
            # 计算掩码在框内的面积
            mask_in_box = mask[y1:y2, x1:x2]
            intersection = np.sum(mask_in_box > 0)
            
            if intersection > 0:
                # 计算覆盖率
                box_area = (x2 - x1) * (y2 - y1)
                coverage = intersection / box_area if box_area > 0 else 0
                
                # 计算掩码中心
                mask_indices = np.where(mask > 0)
                if len(mask_indices[0]) > 0:
                    mask_center_y = np.mean(mask_indices[0])
                    mask_center_x = np.mean(mask_indices[1])
                    
                    # 计算中心距离
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
        
        # 按覆盖率和中心距离综合排序
        masks_in_box.sort(key=lambda x: (x['coverage'], -x['center_distance']), reverse=True)
        return masks_in_box
    
    def _run_local_inference(self, box: Tuple) -> List[Dict]:
        """对局部区域运行推理 - 修复版（解决偏移问题）"""
        x1, y1, x2, y2 = map(int, box)
        h, w = self.image.shape[:2]
        
        # 确保边界框在图像范围内
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return []
        
        # 扩展边界框，确保包含完整颗粒
        expand_pixels = 20
        x1_exp = max(0, x1 - expand_pixels)
        y1_exp = max(0, y1 - expand_pixels)
        x2_exp = min(w, x2 + expand_pixels)
        y2_exp = min(h, y2 + expand_pixels)
        
        # 裁剪区域
        crop = self.image[y1_exp:y2_exp, x1_exp:x2_exp]
        if crop.size == 0:
            return []
        
        try:
            # 根据裁剪区域大小选择合适的推理尺寸
            crop_h, crop_w = crop.shape[:2]
            if crop_h * crop_w < 10000:  # 小区域
                imgsz = 256
            elif crop_h * crop_w < 40000:  # 中等区域
                imgsz = 512
            else:  # 大区域
                imgsz = 1024
            
            # 运行局部推理
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
                
                # 调整掩码大小与裁剪区域一致
                if binary_mask.shape[0] != crop_h or binary_mask.shape[1] != crop_w:
                    binary_mask = cv2.resize(binary_mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                
                # 计算掩码在原图中的位置
                mask_h, mask_w = binary_mask.shape
                
                # 找到掩码的边界框
                rows = np.any(binary_mask, axis=1)
                cols = np.any(binary_mask, axis=0)
                
                if np.any(rows) and np.any(cols):
                    y_min_local, y_max_local = np.where(rows)[0][[0, -1]]
                    x_min_local, x_max_local = np.where(cols)[0][[0, -1]]
                    
                    # 转换到原图坐标
                    x_min_global = x1_exp + x_min_local
                    y_min_global = y1_exp + y_min_local
                    x_max_global = x1_exp + x_max_local
                    y_max_global = y1_exp + y_max_local
                    
                    # 确保掩码在原始框内有足够覆盖
                    overlap_x = max(0, min(x_max_global, x2) - max(x_min_global, x1))
                    overlap_y = max(0, min(y_max_global, y2) - max(y_min_global, y1))
                    overlap_area = overlap_x * overlap_y
                    original_box_area = (x2 - x1) * (y2 - y1)
                    
                    if original_box_area > 0 and overlap_area / original_box_area < 0.3:
                        continue  # 跳过与原框重叠太少的掩码
                
                # 创建完整图像掩码
                full_mask = np.zeros((h, w), dtype=np.uint8)
                full_mask[y1_exp:y1_exp+mask_h, x1_exp:x1_exp+mask_w] = binary_mask
                
                # 增强掩码
                enhanced_mask = self._enhance_mask_morphology(full_mask)
                score = self._calculate_mask_quality(enhanced_mask)
                
                local_masks.append({
                    'mask': enhanced_mask,
                    'score': score,
                    'box': (x1_exp, y1_exp, x1_exp+mask_w, y1_exp+mask_h)
                })
            
            return local_masks
            
        except Exception as e:
            print(f"局部推理失败: {e}")
            return []
    
    def _create_grain_from_mask(self, mask_data: Dict) -> int:
        """从掩码创建颗粒"""
        self.current_grain_id += 1
        
        new_grain = {
            'id': self.current_grain_id,
            'mask': mask_data['mask'],
            'score': mask_data.get('score', 0.5),
            'color': np.random.rand(3,),
            'bbox': None
        }
        
        # 计算边界框
        if mask_data['mask'] is not None and np.any(mask_data['mask']):
            rows = np.any(mask_data['mask'], axis=1)
            cols = np.any(mask_data['mask'], axis=0)
            
            if np.any(rows) and np.any(cols):
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                new_grain['bbox'] = (xmin, ymin, xmax, ymax)
        
        self.grains.append(new_grain)
        self.total_grains += 1
        
        print(f"创建新颗粒 #{self.current_grain_id}, 质量: {new_grain['score']:.3f}")
        
        return self.current_grain_id
    
    def _draw_grain_with_text(self, grain):
        """绘制单个颗粒及其文本标签"""
        try:
            grain_id = grain['id']
            mask = grain['mask']
            
            if mask is None or not np.any(mask):
                return
            
            # 找到轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 简化轮廓
                epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                if len(approx) >= 3:
                    # 提取坐标
                    sx = approx[:, 0, 0]
                    sy = approx[:, 0, 1]
                    
                    # 绘制填充多边形
                    patch = self.ax.fill(sx, sy, 
                                       facecolor=grain['color'], 
                                       edgecolor='black',
                                       alpha=0.4, 
                                       linewidth=1.5)
                    self.grain_patches[grain_id] = patch[0]
                    
                    # 添加文本标签
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
            print(f"绘制颗粒 #{grain.get('id', '未知')} 失败: {e}")
    
    def _refresh_grain_display(self):
        """刷新所有颗粒的显示"""
        try:
            # 清除现有显示
            for patch in self.grain_patches.values():
                patch.remove()
            self.grain_patches.clear()
            
            for text in self.grain_texts.values():
                text.remove()
            self.grain_texts.clear()
            
            # 重新绘制所有颗粒
            for grain in self.grains:
                if grain['mask'] is not None:
                    self._draw_grain_with_text(grain)
            
            self.fig.canvas.draw()
            print(f"已刷新显示，当前颗粒数: {len(self.grains)}")
        
        except Exception as e:
            print(f"刷新显示失败: {e}")
    
    def _on_mouse_press(self, event):
        """鼠标按下事件"""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        
        current_time = time.time()
        if current_time - self.last_draw_time < self.draw_interval:
            return  # 避免过快响应
        
        if event.button == 1:  # 左键：开始绘制框或点选
            self.drawing_box = True
            self.box_start = (event.xdata, event.ydata)
            self.box_end = (event.xdata, event.ydata)
            
            # 清除之前的框
            if self.box_artist:
                self.box_artist.remove()
                self.box_artist = None
            
            self.last_draw_time = current_time
    
    def _on_mouse_move(self, event):
        """鼠标移动事件 - 优化版（减少重绘）"""
        if not self.drawing_box or event.inaxes != self.ax:
            return
        
        current_time = time.time()
        if current_time - self.last_draw_time < self.draw_interval:
            return  # 限制重绘频率
        
        if event.xdata is not None and event.ydata is not None:
            self.box_end = (event.xdata, event.ydata)
            self._draw_current_box()
            self.last_draw_time = current_time
    
    def _on_mouse_release(self, event):
        """鼠标释放事件"""
        if not self.drawing_box or event.inaxes != self.ax:
            return
        
        if event.button == 1:  # 左键释放
            self.drawing_box = False
            
            if self.box_start and self.box_end:
                # 计算框坐标
                x1, y1 = self.box_start
                x2, y2 = self.box_end
                
                # 确保坐标正确
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                box = (x1, y1, x2, y2)
                box_area = (x2 - x1) * (y2 - y1)
                
                if box_area < 100:  # 如果框太小，视为点选
                    print(f"点选: ({x1:.1f}, {y1:.1f})")
                    self._handle_point_click(x1, y1)
                else:
                    print(f"框选: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                    self._handle_box_selection(box)
            
            # 清除框显示
            if self.box_artist:
                self.box_artist.remove()
                self.box_artist = None
                self.fig.canvas.draw()
            
            self.box_start = None
            self.box_end = None
    
    def _draw_current_box(self):
        """绘制当前框 - 优化版"""
        if not self.box_start or not self.box_end:
            return
        
        # 清除之前的框
        if self.box_artist:
            self.box_artist.remove()
        
        # 绘制新框
        x1, y1 = self.box_start
        x2, y2 = self.box_end
        
        rect = plt.Rectangle((min(x1, x2), min(y1, y2)), 
                           abs(x2 - x1), abs(y2 - y1),
                           fill=False, edgecolor='cyan', 
                           linewidth=2, linestyle='--', alpha=0.7)
        
        self.box_artist = self.ax.add_patch(rect)
        
        # 只更新框的部分，而不是整个图形
        self.fig.canvas.draw_idle()
    
    def _handle_point_click(self, x: float, y: float):
        """处理点点击事件 - 修复版"""
        # 查找点击位置的掩码
        mask_data = self._find_mask_at_point(x, y)
        
        if mask_data:
            # 检查这个掩码是否已经被选择
            for grain in self.grains:
                if np.array_equal(grain['mask'], mask_data['mask']):
                    print(f"颗粒 #{grain['id']} 已被选择")
                    return
            
            # 创建新颗粒
            grain_id = self._create_grain_from_mask(mask_data)
            self._refresh_grain_display()
        else:
            # 尝试在点周围进行局部推理
            print("未找到掩码，进行局部推理...")
            box_size = 50
            box = (x - box_size, y - box_size, x + box_size, y + box_size)
            local_masks = self._run_local_inference(box)
            
            if local_masks:
                # 选择质量最高的掩码
                best_local_mask = max(local_masks, key=lambda x: x['score'])
                grain_id = self._create_grain_from_mask(best_local_mask)
                self._refresh_grain_display()
                print(f"局部推理成功，生成新掩码")
            else:
                print("未找到该位置的掩码")
    
    def _handle_box_selection(self, box: Tuple):
        """处理框选择事件 - 修复版"""
        # 查找框内的掩码
        masks_in_box = self._find_masks_in_box(box)
        
        if masks_in_box:
            # 选择最佳掩码（覆盖率最高，中心距离最近）
            best_mask = masks_in_box[0]
            
            # 检查这个掩码是否已经被选择
            for grain in self.grains:
                if np.array_equal(grain['mask'], best_mask['mask']):
                    print(f"颗粒 #{grain['id']} 已被选择")
                    return
            
            # 创建新颗粒
            grain_id = self._create_grain_from_mask(best_mask)
            self._refresh_grain_display()
            
            print(f"选择了框内最佳掩码，覆盖率: {best_mask['coverage']:.3f}")
        else:
            # 如果没有找到掩码，运行局部推理
            print("未找到框内掩码，进行局部推理...")
            local_masks = self._run_local_inference(box)
            
            if local_masks:
                # 选择质量最高的掩码
                best_local_mask = max(local_masks, key=lambda x: x['score'])
                grain_id = self._create_grain_from_mask(best_local_mask)
                self._refresh_grain_display()
                print(f"局部推理成功，生成新掩码")
            else:
                print("局部推理也未生成掩码")
    
    def _on_key_press(self, event):
        """键盘按键事件处理"""
        if event.key == 'x':  # 删除最后一个颗粒
            self._delete_last_grain()
        elif event.key == 'd':  # 删除所有颗粒
            self._delete_all_grains()
        elif event.key == 's':  # 保存结果
            self._show_save_options()
        elif event.key == 'r':  # 重新开始
            self._reset_interface()
        elif event.key == 'q':  # 退出
            print("退出交互界面")
            self.gui_running = False
            plt.close(self.fig)
        elif event.key == 'h':  # 显示帮助
            self._show_help()
        elif event.key == 'S':  # Shift+S：快速保存完整结果
            print("快速保存完整结果...")
            self._generate_complete_outputs()
    
    def _delete_last_grain(self):
        """删除最后一个颗粒"""
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
            print(f"已删除颗粒 #{grain_id}")
    
    def _delete_all_grains(self):
        """删除所有颗粒"""
        for patch in self.grain_patches.values():
            patch.remove()
        self.grain_patches.clear()
        
        for text in self.grain_texts.values():
            text.remove()
        self.grain_texts.clear()
        
        self.grains = []
        self.current_grain_id = 0
        
        self.fig.canvas.draw()
        print("已删除所有颗粒")
    
    def _show_help(self):
        """显示帮助信息"""
        help_text = (
            "FastSAM交互式分割指南:\n\n"
            "鼠标操作:\n"
            "• 左键拖动: 绘制选择框\n"
            "• 左键单击: 点选颗粒（小范围）\n\n"
            "键盘快捷键:\n"
            "• 's': 显示保存选项\n"
            "• 'S' (Shift+s): 快速保存完整结果\n"
            "• 'x': 删除最后一个颗粒\n"
            "• 'd': 删除所有颗粒\n"
            "• 'r': 重置界面\n"
            "• 'q': 退出程序\n"
            "• 'h': 显示此帮助\n"
        )
        
        messagebox.showinfo("FastSAM交互式分割帮助", help_text)
    
    def _show_help_text_fixed(self):
        """在界面上显示帮助文本"""
        help_text = (
            "FastSAM交互指南：\n"
            "• 左键拖动：绘制选择框\n"
            "• 左键单击：点选颗粒（小范围）\n"
            "• 's'键：显示保存选项\n"
            "• 'S'键 (Shift+s)：快速保存完整结果\n"
            "• 'x'键：删除最后一个颗粒\n"
            "• 'd'键：删除所有颗粒\n"
            "• 'r'键：重置界面\n"
            "• 'q'键：退出程序\n"
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
        """重置整个界面"""
        self._delete_all_grains()
        
        # 重新运行全图推理
        if self.image is not None:
            self._run_global_inference()
        
        self.ax.clear()
        self.ax.imshow(self.image)
        
        image_name = Path(self.image_path).name if self.image_path else "未命名图片"
        title_text = f"FastSAM增强版交互式分割 - {image_name}"
        self.ax.set_title(title_text, fontsize=16, fontproperties='Microsoft YaHei')
        
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        self._show_help_text_fixed()
        self.fig.canvas.draw()
        
        print("界面已完全重置")
    
    def show_interactive_interface(self):
        """显示交互式界面"""
        if self.image is None:
            print("请先加载图片")
            return
        
        if self.fastsam_model is None:
            print("FastSAM模型未初始化")
            return
        
        print("正在创建交互式界面...")
        
        try:
            # 设置中文字体
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
                print(f"设置中文字体失败: {e}")
            
            # 创建图形
            self.fig, self.ax = plt.subplots(figsize=(14, 10))
            self.ax.imshow(self.image)
            
            image_name = Path(self.image_path).name if self.image_path else "未命名图片"
            title_text = f"FastSAM增强版交互式分割 - {image_name}"
            self.ax.set_title(title_text, fontsize=16, fontproperties='Microsoft YaHei')
            
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            
            # 连接事件
            self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
            self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
            self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
            self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
            
            self._show_help_text_fixed()
            
            plt.tight_layout()
            
            print("FastSAM交互式界面已启动")
            print("提示:")
            print("  1. 左键拖动绘制选择框")
            print("  2. 左键单击点选小颗粒")
            print("  3. 按 'Shift+S' 快速保存结果")
            
            self.gui_running = True
            
            if backend == 'Agg':
                print("检测到无GUI环境，将保存结果图片")
                output_path = self.output_dir / "interactive_result.png"
                self.fig.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"结果已保存到: {output_path}")
                plt.close(self.fig)
                return
            
            plt.show(block=True)
            
            print("交互式窗口已关闭")
            
        except Exception as e:
            print(f"显示交互界面失败: {e}")
            traceback.print_exc()
    
    def run_interactive_mode(self, image_path: str = None):
        """运行完整的交互式模式"""
        try:
            if not self.model_loaded:
                print("模型未加载，无法运行交互模式")
                return
            
            if image_path:
                print(f"加载指定图片: {image_path}")
                if not self.load_image_from_path(image_path):
                    print("图片加载失败，退出交互模式")
                    return
            else:
                print("请通过文件选择对话框选择图片...")
                if not self.load_image_with_gui():
                    print("图片选择失败，退出交互模式")
                    return
            
            print("启动交互式界面...")
            self.show_interactive_interface()
            
        except Exception as e:
            print(f"交互模式运行失败: {e}")
            traceback.print_exc()
    
    def _masks_to_polygons(self) -> List[ShapelyPolygon]:
        """将掩码转换为多边形列表"""
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
                    print(f"转换掩码为多边形失败（颗粒#{grain['id']}）: {e}")
        
        return polygons
    
    def _generate_grain_dataframe(self) -> pd.DataFrame:
        """生成颗粒DataFrame - 修复几何计算问题"""
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
                        
                        # 计算周长
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
                            # 近似计算
                            perimeter = 4 * np.sqrt(area) * 0.9
                        
                        # 计算基本几何参数
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
            
            # 添加缺少的列以匹配GrainShapeMetrics的期望
            if 'major_axis_length' not in basic_df.columns:
                # 估算主轴长度
                basic_df['major_axis_length'] = basic_df['bbox_width']
            
            if 'minor_axis_length' not in basic_df.columns:
                # 估算次轴长度
                basic_df['minor_axis_length'] = basic_df['bbox_height']
            
            if 'orientation' not in basic_df.columns:
                # 默认方向
                basic_df['orientation'] = 0.0
            
            return basic_df
                
        except Exception as e:
            print(f"生成颗粒数据失败: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    
    def _generate_complete_outputs(self, output_dir: Optional[Path] = None) -> Path:
        """生成完整输出文件"""
        if len(self.grains) == 0:
            print("没有分割颗粒，无法生成输出文件")
            return None
        
        if output_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_name = Path(self.image_path).stem if self.image_path else "interactive"
            output_dir = self.output_dir / f"{image_name}_{timestamp}"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n生成完整输出到: {output_dir}")
        
        try:
            # 1. 生成多边形
            self.polygons = self._masks_to_polygons()
            
            if len(self.polygons) == 0:
                print("无法生成有效多边形")
                return None
            
            print(f"生成了 {len(self.polygons)} 个多边形")
            
            # 2. 生成颗粒数据
            self.grain_data = self._generate_grain_dataframe()
            
            if self.grain_data.empty:
                print("无法生成颗粒数据")
                return None
            
            print(f"颗粒数据包含 {len(self.grain_data.columns)} 个参数")
            
            # 3. 保存交互式界面截图
            if self.fig is not None:
                vis_path = output_dir / "interactive_visualization.png"
                try:
                    self.fig.savefig(vis_path, dpi=300, bbox_inches='tight')
                    print(f"交互式界面截图保存至: {vis_path}")
                except Exception as e:
                    print(f"保存交互式界面截图失败: {e}")
            
            # 4. 生成YOLO风格可视化图
            if self.image is not None:
                fig, axes = plt.subplots(1, 2, figsize=(20, 10))
                
                # 左侧：轮廓图
                axes[0].imshow(self.image)
                axes[0].set_title(f'FastSAM Grain Segmentation (n={len(self.polygons)})', fontsize=16)
                axes[0].axis('off')
                
                for poly in self.polygons:
                    if poly.is_valid:
                        x, y = poly.exterior.xy
                        axes[0].plot(x, y, color='red', linewidth=1, alpha=0.8)
                
                # 右侧：彩色填充图
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
                
                print(f"YOLO风格可视化图保存至: {plot_path}")
            
            # 5. 保存CSV文件
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
                        print(f"配置筛选失败: {e}")
                        self.grain_data.to_csv(csv_path, index=False, encoding='utf-8')
                else:
                    self.grain_data.to_csv(csv_path, index=False, encoding='utf-8')
                
                print(f"颗粒统计表保存至: {csv_path}")
            
            # 6. 创建JSON摘要
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
            print(f"JSON摘要保存至: {json_path}")
            
            print(f"\n所有结果已保存到: {output_dir}")
            
            return output_dir
            
        except Exception as e:
            print(f"生成完整输出失败: {e}")
            traceback.print_exc()
            return None
    
    def _show_save_options(self):
        """显示保存选项"""
        if len(self.grains) == 0:
            print("没有颗粒可保存")
            return
        
        save_choice = input("\n保存选项:\n1. 快速保存完整结果\n2. 自定义保存路径\n3. 取消\n选择 (1-3): ").strip()
        
        if save_choice == '1':
            output_dir = self._generate_complete_outputs()
            if output_dir:
                print(f"结果已保存到: {output_dir}")
        elif save_choice == '2':
            try:
                root = tk.Tk()
                root.withdraw()
                folder_path = filedialog.askdirectory(title="选择保存目录")
                root.destroy()
                
                if folder_path:
                    output_dir = self._generate_complete_outputs(Path(folder_path))
                    if output_dir:
                        print(f"结果已保存到: {output_dir}")
            except Exception as e:
                print(f"保存失败: {e}")


def main():
    """主函数：直接运行增强版交互式FastSAM"""
    print("=" * 70)
    print("FastSAM增强版交互式分割系统（修复版）")
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