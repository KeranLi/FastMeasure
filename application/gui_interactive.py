#!/usr/bin/env python
"""
FastMeasure 嵌入式 Interactive 模式
支持 FastSAM 和 MobileSAM 两种模型
"""

import sys
import os
import time
import traceback
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import numpy as np
import warnings
import cv2

# 修复打包环境中的缺失模块 - 完整的 unittest.mock 实现
import unittest
if not hasattr(unittest, 'mock'):
    import types
    from functools import wraps
    
    mock_module = types.ModuleType('unittest.mock')
    
    # Mock 类
    class Mock:
        def __init__(self, *args, **kwargs):
            self._children = {}
            self._return_value = None
            self._side_effect = None
        
        def __call__(self, *args, **kwargs):
            if self._side_effect:
                if callable(self._side_effect):
                    return self._side_effect(*args, **kwargs)
                elif isinstance(self._side_effect, (list, tuple)):
                    return self._side_effect.pop(0)
            return self._return_value
        
        def __getattr__(self, name):
            if name not in self._children:
                self._children[name] = Mock()
            return self._children[name]
        
        def __setattr__(self, name, value):
            if name.startswith('_'):
                super().__setattr__(name, value)
            else:
                if name not in ['_children', '_return_value', '_side_effect']:
                    self._children[name] = value
                else:
                    super().__setattr__(name, value)
        
        @property
        def return_value(self):
            return self._return_value
        
        @return_value.setter
        def return_value(self, value):
            self._return_value = value
        
        @property
        def side_effect(self):
            return self._side_effect
        
        @side_effect.setter
        def side_effect(self, value):
            self._side_effect = value
    
    # MagicMock 类
    class MagicMock(Mock):
        pass
    
    # patch 装饰器/上下文管理器
    class _Patch:
        def __init__(self, target, new=Mock):
            self.target = target
            self.new = new
            self.original = None
        
        def __enter__(self):
            parts = self.target.split('.')
            module_name = parts[0]
            if module_name in sys.modules:
                module = sys.modules[module_name]
                if len(parts) > 1:
                    obj = module
                    for part in parts[1:-1]:
                        obj = getattr(obj, part, None)
                        if obj is None:
                            break
                    if obj:
                        self.original = getattr(obj, parts[-1], None)
                        setattr(obj, parts[-1], self.new)
            return self.new
        
        def __exit__(self, *args):
            if self.original is not None:
                parts = self.target.split('.')
                module_name = parts[0]
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                    if len(parts) > 1:
                        obj = module
                        for part in parts[1:-1]:
                            obj = getattr(obj, part, None)
                            if obj is None:
                                break
                        if obj:
                            setattr(obj, parts[-1], self.original)
        
        def __call__(self, func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)
            return wrapper
    
    def patch(target, new=Mock):
        return _Patch(target, new)
    
    mock_module.Mock = Mock
    mock_module.MagicMock = MagicMock
    mock_module.patch = patch
    
    unittest.mock = mock_module
    sys.modules['unittest.mock'] = mock_module
    sys.modules['mock'] = mock_module

warnings.filterwarnings('ignore')

# 资源路径
def get_resource_path():
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).parent

RESOURCE_PATH = get_resource_path()
sys.path.insert(0, str(RESOURCE_PATH))


class InteractiveSegmenter:
    """交互式分割器 - 支持 FastSAM 和 MobileSAM"""
    
    def __init__(self, parent_window, model_type="fastsam"):
        self.parent = parent_window
        self.model_type = model_type
        self.device = "cpu"
        
        # 图像数据
        self.image = None
        self.image_path = None
        self.image_tk = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # 模型
        self.fastsam_model = None  # FastSAM 模型
        self.predictor = None       # MobileSAM predictor
        self.model_loaded = False
        
        # FastSAM 缓存
        self.all_masks_cache = []      # 全局推理缓存的掩码
        self.all_masks_scores = []     # 掩码质量分数
        self.global_results = None     # 全局推理结果
        
        # 颗粒数据
        self.grains = []
        self.current_grain_id = 0
        
        # 输出目录
        self.output_dir = RESOURCE_PATH / "interactive_results"
        self.output_dir.mkdir(exist_ok=True)
        
        # 比例尺标定
        self.scale_factor = None
        self.scale_calibration_points = []
        self.is_scale_calibration_mode = False
        self.scale_line_id = None
        
        # 性能统计
        self.start_time = None
        
        self._create_ui()
        self._load_model()
    
    def _create_ui(self):
        """创建用户界面"""
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        model_name = "FastSAM" if self.model_type == "fastsam" else "MobileSAM"
        ttk.Label(title_frame, text=f"{model_name} 交互式分割", 
                 font=("Helvetica", 16, "bold")).pack(side=tk.LEFT)
        
        # 工具栏
        toolbar = ttk.LabelFrame(main_frame, text="工具", padding="5")
        toolbar.pack(fill=tk.X, pady=5)
        
        # 操作按钮
        btn_frame = ttk.Frame(toolbar)
        btn_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="🗑️ 删除最后", command=self._delete_last_grain, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="💾 保存结果", command=self._save_results, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="❓ 帮助", command=self._show_help, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="✅ 完成", command=self._finish, width=8).pack(side=tk.LEFT, padx=2)
        
        # 图像显示区域
        canvas_frame = ttk.LabelFrame(main_frame, text="图像 - 点击颗粒进行分割", padding="5")
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.canvas = tk.Canvas(canvas_frame, bg='gray', cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 绑定事件
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.parent.bind("<Key>", self._on_key_press)
        
        # 状态栏
        self.status_var = tk.StringVar(value="请加载图片开始分割")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(5, 0))
        
        # 信息面板
        info_frame = ttk.LabelFrame(main_frame, text="信息", padding="5")
        info_frame.pack(fill=tk.X, pady=5)
        
        self.info_text = tk.Text(info_frame, height=6, wrap=tk.WORD, 
                                font=("Consolas", 9))
        self.info_text.pack(fill=tk.X)
        
        # 存储画布对象
        self.canvas_objects = []
        
        self._log(f"🚀 {model_name} 交互式系统")
        self._log("点击图像上的颗粒进行分割")
    
    def _load_model(self):
        """加载模型"""
        try:
            self._log("=" * 50)
            self._log(f"加载 {self.model_type.upper()} 模型...")
            
            if self.model_type == "fastsam":
                # FastSAM 使用 ultralytics
                from ultralytics import FastSAM
                import torch
                
                model_path = RESOURCE_PATH / "models" / "FastSAM-s.pt"
                self._log(f"模型路径: {model_path}")
                
                if not model_path.exists():
                    self._log(f"❌ 模型文件不存在: {model_path}")
                    return False
                
                # 检测 MPS
                if torch.backends.mps.is_available():
                    self.device = "mps"
                    self._log("✅ 使用 MPS 加速")
                
                self._log("正在加载 FastSAM 模型...")
                self.fastsam_model = FastSAM(str(model_path))
                self.fastsam_model.to(self.device)
                self.model_loaded = True
                self._log(f"✅ FastSAM 模型加载成功")
                
            else:  # mobilesam
                from mobile_sam import sam_model_registry, SamPredictor
                import torch
                
                model_path = RESOURCE_PATH / "models" / "mobile_sam.pt"
                self._log(f"模型路径: {model_path}")
                
                if not model_path.exists():
                    self._log(f"❌ 模型文件不存在: {model_path}")
                    return False
                
                if torch.backends.mps.is_available():
                    self.device = "mps"
                    self._log("✅ 使用 MPS 加速")
                
                self._log("正在加载 MobileSAM 模型...")
                sam = sam_model_registry["vit_t"](checkpoint=str(model_path))
                sam.to(device=self.device)
                self.predictor = SamPredictor(sam)
                self.model_loaded = True
                self._log(f"✅ MobileSAM 模型加载成功")
            
            return True
            
        except Exception as e:
            self._log(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_image(self, image_path):
        """加载图片"""
        try:
            self.image_path = image_path
            
            # 使用 PIL 加载图片
            pil_image = Image.open(image_path).convert('RGB')
            self.image = np.array(pil_image)
            
            # 显示图片
            self._display_image()
            
            # FastSAM: 运行全局推理
            if self.model_type == "fastsam" and self.fastsam_model:
                self._run_fastsam_global_inference()
            
            # MobileSAM: 设置图片到 predictor
            if self.model_type == "mobilesam" and self.predictor:
                self.predictor.set_image(self.image)
            
            self.start_time = time.time()
            self._log(f"✅ 图片加载成功: {self.image.shape}")
            self.status_var.set("点击图像上的颗粒进行分割")
            
            return True
            
        except Exception as e:
            self._log(f"❌ 图片加载失败: {e}")
            messagebox.showerror("错误", f"无法加载图片: {e}")
            return False
    
    def _run_fastsam_global_inference(self):
        """FastSAM 全局推理 - 预计算所有候选掩码"""
        try:
            self._log("正在运行 FastSAM 全局推理...")
            start_time = time.time()
            
            results = self.fastsam_model(
                self.image,
                device=self.device,
                imgsz=1024,
                conf=0.25,
                iou=0.3,
                verbose=False
            )
            
            if len(results) == 0 or results[0].masks is None:
                self._log("⚠️ 全局推理未生成掩码")
                return False
            
            # 缓存结果
            self.global_results = results[0]
            masks_data = results[0].masks.data.cpu().numpy()
            
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
                kernel = np.ones((3, 3), np.uint8)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
                
                # 计算质量分数
                score = np.sum(binary_mask) / (h * w)  # 简单分数
                
                self.all_masks_cache.append(binary_mask)
                self.all_masks_scores.append(score)
            
            inference_time = time.time() - start_time
            self._log(f"✅ 全局推理完成: {len(self.all_masks_cache)} 个候选掩码, 耗时: {inference_time:.2f}s")
            return True
            
        except Exception as e:
            self._log(f"❌ 全局推理失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _display_image(self):
        """在画布上显示图片"""
        if self.image is None:
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width < 100:
            canvas_width = 800
            canvas_height = 600
        
        h, w = self.image.shape[:2]
        scale = min((canvas_width - 40) / w, (canvas_height - 40) / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        self.scale_x = scale
        self.scale_y = scale
        self.offset_x = (canvas_width - new_w) // 2
        self.offset_y = (canvas_height - new_h) // 2
        
        pil_img = Image.fromarray(self.image)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.image_tk = ImageTk.PhotoImage(pil_img)
        
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, 
                                anchor=tk.NW, image=self.image_tk)
        
        self.canvas_objects = []
    
    def _on_left_click(self, event):
        """左键点击 - 分割颗粒"""
        if self.is_scale_calibration_mode:
            self._handle_scale_calibration_click(event.x, event.y)
            return
        
        self._handle_segmentation_click(event.x, event.y)
    
    def _handle_segmentation_click(self, canvas_x, canvas_y):
        """处理分割点击"""
        if self.image is None:
            return
        
        # 转换为原始图像坐标
        orig_x = int((canvas_x - self.offset_x) / self.scale_x)
        orig_y = int((canvas_y - self.offset_y) / self.scale_y)
        
        h, w = self.image.shape[:2]
        if not (0 <= orig_x < w and 0 <= orig_y < h):
            return
        
        self._log(f"点击位置: ({orig_x}, {orig_y})")
        
        if self.model_type == "fastsam":
            self._run_fastsam_segmentation(orig_x, orig_y)
        else:
            self._run_mobilesam_segmentation(orig_x, orig_y)
    
    def _run_fastsam_segmentation(self, x, y):
        """FastSAM 分割 - 从预计算掩码中选择"""
        if not self.all_masks_cache:
            self._log("⚠️ 没有可用的候选掩码")
            return
        
        # 查找点击位置的掩码
        clicked_mask = None
        clicked_score = 0
        
        for mask, score in zip(self.all_masks_cache, self.all_masks_scores):
            if mask[y, x] > 0:
                # 检查是否已选择
                already_selected = False
                for grain in self.grains:
                    if grain['mask'] is not None and np.array_equal(grain['mask'], mask):
                        already_selected = True
                        break
                
                if not already_selected and score > clicked_score:
                    clicked_mask = mask
                    clicked_score = score
        
        if clicked_mask is not None:
            self._create_grain_from_mask(clicked_mask, clicked_score)
        else:
            self._log("⚠️ 该位置没有检测到颗粒")
    
    def _run_mobilesam_segmentation(self, x, y):
        """MobileSAM 分割 - 使用点提示"""
        if self.predictor is None:
            self._log("⚠️ 模型未加载")
            return
        
        # 检查是否点击在现有颗粒上
        clicked_grain = None
        for grain in reversed(self.grains):
            if grain['mask'] is not None and grain['mask'][y, x] > 0:
                clicked_grain = grain
                break
        
        if clicked_grain:
            # 添加到现有颗粒
            grain_id = clicked_grain['id']
            clicked_grain['points'].append({'x': x, 'y': y, 'is_foreground': True})
        else:
            # 创建新颗粒
            grain_id = self._create_new_grain(x, y)
        
        # 运行分割
        self._run_mobilesam_predict(grain_id)
    
    def _create_new_grain(self, x, y):
        """创建新颗粒"""
        self.current_grain_id += 1
        grain = {
            'id': self.current_grain_id,
            'points': [{'x': x, 'y': y, 'is_foreground': True}],
            'mask': None,
            'confidence': 0.5,
            'bbox': None
        }
        self.grains.append(grain)
        return self.current_grain_id
    
    def _run_mobilesam_predict(self, grain_id):
        """MobileSAM 预测"""
        try:
            grain = None
            for g in self.grains:
                if g['id'] == grain_id:
                    grain = g
                    break
            
            if grain is None or not grain['points']:
                return
            
            input_points = []
            input_labels = []
            
            for point in grain['points']:
                input_points.append([point['x'], point['y']])
                input_labels.append(1 if point['is_foreground'] else 0)
            
            input_points = np.array(input_points, dtype=np.float32)
            input_labels = np.array(input_labels, dtype=np.int32)
            
            masks, scores, _ = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True
            )
            
            if len(masks) == 0:
                return
            
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            
            grain['mask'] = mask
            grain['confidence'] = float(scores[best_idx])
            
            if np.any(mask):
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                grain['bbox'] = (xmin, ymin, xmax, ymax)
            
            self._log(f"✅ 颗粒 #{grain_id} 分割完成，置信度: {scores[best_idx]:.3f}")
            self._update_grain_display(grain_id)
            
        except Exception as e:
            self._log(f"❌ 分割错误: {e}")
    
    def _create_grain_from_mask(self, mask, score):
        """从掩码创建颗粒"""
        self.current_grain_id += 1
        grain = {
            'id': self.current_grain_id,
            'points': [],
            'mask': mask.copy(),
            'confidence': float(score),
            'bbox': None
        }
        
        # 计算边界框
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if np.any(rows) and np.any(cols):
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            grain['bbox'] = (xmin, ymin, xmax, ymax)
        
        self.grains.append(grain)
        self._log(f"✅ 创建颗粒 #{self.current_grain_id}")
        self._update_grain_display(self.current_grain_id)
        self.status_var.set(f"颗粒数: {len(self.grains)}")
    
    def _update_grain_display(self, grain_id):
        """更新颗粒显示"""
        grain = None
        for g in self.grains:
            if g['id'] == grain_id:
                grain = g
                break
        
        if grain is None or grain['mask'] is None:
            return
        
        # 清除该颗粒的旧显示
        self.canvas_objects = [
            obj for obj in self.canvas_objects 
            if not (obj['grain_id'] == grain_id and obj['type'] in ['mask', 'text'])
        ]
        
        # 绘制轮廓
        mask = grain['mask'].astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            scaled_contour = []
            for point in contour:
                x = int(point[0][0] * self.scale_x + self.offset_x)
                y = int(point[0][1] * self.scale_y + self.offset_y)
                scaled_contour.extend([x, y])
            
            if len(scaled_contour) >= 6:
                contour_id = self.canvas.create_polygon(
                    scaled_contour, outline='lime', fill='', width=2
                )
                self.canvas_objects.append({
                    'type': 'mask', 'id': contour_id, 'grain_id': grain_id
                })
        
        # 添加编号
        y_indices, x_indices = np.where(grain['mask'])
        if len(y_indices) > 0:
            cx = int(np.mean(x_indices) * self.scale_x + self.offset_x)
            cy = int(np.mean(y_indices) * self.scale_y + self.offset_y)
            
            text_id = self.canvas.create_text(
                cx, cy, text=str(grain_id), 
                fill='yellow', font=('Arial', 12, 'bold')
            )
            self.canvas_objects.append({
                'type': 'text', 'id': text_id, 'grain_id': grain_id
            })
    
    def _delete_last_grain(self):
        """删除最后一个颗粒"""
        if not self.grains:
            return
        
        grain = self.grains.pop()
        grain_id = grain['id']
        
        for obj in self.canvas_objects:
            if obj['grain_id'] == grain_id:
                self.canvas.delete(obj['id'])
        
        self.canvas_objects = [
            obj for obj in self.canvas_objects 
            if obj['grain_id'] != grain_id
        ]
        
        if self.grains:
            self.current_grain_id = max(g['id'] for g in self.grains)
        else:
            self.current_grain_id = 0
        
        self._log(f"🗑️ 删除颗粒 #{grain_id}")
        self.status_var.set(f"颗粒数: {len(self.grains)}")
    
    def _save_results(self):
        """保存结果"""
        if not self.grains:
            messagebox.showwarning("提示", "没有分割结果")
            return
        
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = Path(self.image_path).stem if self.image_path else "interactive"
            output_subdir = self.output_dir / f"{image_name}_{timestamp}"
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # 保存掩码
            h, w = self.image.shape[:2]
            mask_all = np.zeros((h, w), dtype=np.uint8)
            for grain in self.grains:
                if grain['mask'] is not None:
                    mask_all = np.maximum(mask_all, grain['mask'].astype(np.uint8))
            
            mask_path = output_subdir / "segmentation_mask.png"
            Image.fromarray(mask_all * 255).save(mask_path)
            
            # 保存统计
            grain_stats = []
            for grain in self.grains:
                if grain['mask'] is not None:
                    area_px = np.sum(grain['mask'])
                    area_um2 = area_px * (self.scale_factor ** 2) if self.scale_factor else None
                    
                    y_indices, x_indices = np.where(grain['mask'])
                    if len(y_indices) > 0:
                        grain_stats.append({
                            'grain_id': grain['id'],
                            'area_pixels': int(area_px),
                            'area_um2': float(area_um2) if area_um2 else None,
                            'confidence': grain.get('confidence', 0.5)
                        })
            
            if grain_stats:
                import csv
                csv_path = output_subdir / "grain_statistics.csv"
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=grain_stats[0].keys())
                    writer.writeheader()
                    writer.writerows(grain_stats)
            
            # 保存摘要
            summary = {
                'image_path': str(self.image_path),
                'grains_count': len(self.grains),
                'scale_factor': float(self.scale_factor) if self.scale_factor else None,
                'model_type': self.model_type,
                'timestamp': timestamp
            }
            
            json_path = output_subdir / "summary.json"
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self._log(f"✅ 结果保存到: {output_subdir}")
            messagebox.showinfo("保存成功", f"结果已保存到:\n{output_subdir}")
            
        except Exception as e:
            self._log(f"❌ 保存失败: {e}")
    
    def _show_help(self):
        """显示帮助"""
        help_text = f"""快捷键:
  x - 删除最后一个颗粒
  s - 保存结果
  m - 比例尺标定
  q - 退出

当前模型: {self.model_type.upper()}

使用方法:
  直接点击图像上的颗粒进行分割
"""
        messagebox.showinfo("帮助", help_text)
    
    def _finish(self):
        """完成"""
        if self.grains and messagebox.askyesno("确认", "保存结果后退出?"):
            self._save_results()
        self.parent.destroy()
    
    def _on_right_click(self, event):
        """右键点击"""
        pass  # 可扩展为背景点功能
    
    def _on_mouse_move(self, event):
        """鼠标移动"""
        if self.image is None:
            return
        
        orig_x = int((event.x - self.offset_x) / self.scale_x)
        orig_y = int((event.y - self.offset_y) / self.scale_y)
        
        h, w = self.image.shape[:2]
        if 0 <= orig_x < w and 0 <= orig_y < h:
            scale_info = f"比例尺: {self.scale_factor:.2f}μm/px | " if self.scale_factor else ""
            self.status_var.set(f"{scale_info}位置: ({orig_x}, {orig_y}) | 颗粒: {len(self.grains)}")
    
    def _on_key_press(self, event):
        """键盘按键"""
        key = event.char.lower() if event.char else event.keysym.lower()
        
        if key == 'x':
            self._delete_last_grain()
        elif key == 's':
            self._save_results()
        elif key == 'm':
            self._start_scale_calibration()
        elif key == 'q':
            self._finish()
        elif key == 'h':
            self._show_help()
    
    def _start_scale_calibration(self):
        """开始比例尺标定"""
        if self.image is None:
            return
        
        self.is_scale_calibration_mode = True
        self.scale_calibration_points = []
        messagebox.showinfo("比例尺标定", "点击两个点测量已知长度")
        self.status_var.set("【比例尺标定】点击起点")
    
    def _handle_scale_calibration_click(self, canvas_x, canvas_y):
        """处理比例尺标定点击"""
        orig_x = int((canvas_x - self.offset_x) / self.scale_x)
        orig_y = int((canvas_y - self.offset_y) / self.scale_y)
        
        h, w = self.image.shape[:2]
        if not (0 <= orig_x < w and 0 <= orig_y < h):
            return
        
        if len(self.scale_calibration_points) == 0:
            self.scale_calibration_points.append((orig_x, orig_y, canvas_x, canvas_y))
            self._log(f"起点: ({orig_x}, {orig_y})")
            self.status_var.set("【比例尺标定】点击终点")
        else:
            x1, y1, cx1, cy1 = self.scale_calibration_points[0]
            x2, y2 = orig_x, orig_y
            
            pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            self._log(f"像素距离: {pixel_distance:.2f} px")
            
            # 询问实际长度
            dialog = tk.Toplevel(self.parent)
            dialog.title("输入实际长度")
            dialog.geometry("300x120")
            dialog.transient(self.parent)
            dialog.grab_set()
            
            ttk.Label(dialog, text=f"像素距离: {pixel_distance:.2f} px\n输入实际长度 (μm):").pack(pady=5)
            
            entry = ttk.Entry(dialog, width=15)
            entry.pack()
            entry.insert(0, "1000")
            
            result = [None]
            
            def on_ok():
                try:
                    result[0] = float(entry.get())
                    dialog.destroy()
                except:
                    pass
            
            ttk.Button(dialog, text="确定", command=on_ok).pack(pady=10)
            self.parent.wait_window(dialog)
            
            if result[0]:
                self.scale_factor = result[0] / pixel_distance
                self._log(f"✅ 比例因子: {self.scale_factor:.4f} μm/px")
            
            self.is_scale_calibration_mode = False
    
    def _log(self, message):
        """添加日志"""
        self.info_text.insert(tk.END, message + "\n")
        self.info_text.see(tk.END)


def run_interactive_gui(parent, model_type, image_path):
    """运行交互式分割"""
    window = tk.Toplevel(parent)
    window.title(f"交互式分割 - {model_type.upper()}")
    window.geometry("1000x800")
    window.transient(parent)
    window.grab_set()
    
    segmenter = InteractiveSegmenter(window, model_type)
    
    if image_path and Path(image_path).exists():
        segmenter.load_image(image_path)
    else:
        messagebox.showerror("错误", f"无法加载图片: {image_path}")
        window.destroy()
        return None
    
    return segmenter


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename()
    if path:
        run_interactive_gui(root, "fastsam", path)
        root.mainloop()
