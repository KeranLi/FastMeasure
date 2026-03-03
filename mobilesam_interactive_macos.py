#!/usr/bin/env python3
"""
MobileSAM修复版交互式界面模块（跨平台兼容版）
文件名：mobilesam_interactive_macos.py
功能：基于点提示的纯交互式MobileSAM分割，支持Windows/macOS/Linux
修复：解决macOS子线程GUI崩溃问题，使用AppleScript/PyQt6替代Tkinter文件对话框
"""

import os
import sys
import platform
import numpy as np
import matplotlib

def setup_backend():
    """智能设置后端，优先GUI后端"""
    system = platform.system()
    
    # macOS特殊处理
    if system == "Darwin":
        # 尝试MacOSX后端（原生体验最好）
        try:
            matplotlib.use('MacOSX')
            return 'MacOSX'
        except:
            pass
    
    # 通用方案
    try:
        import tkinter
        matplotlib.use('TkAgg')
        return 'TkAgg'
    except ImportError:
        pass
    except Exception:
        pass
    
    if os.getenv('DISPLAY') and not os.getenv('SSH_CONNECTION'):
        for backend in ['Qt5Agg', 'Qt6Agg', 'TkAgg', 'WXAgg']:
            try:
                matplotlib.use(backend)
                return backend
            except:
                continue
    
    matplotlib.use('Agg')
    return 'Agg'

backend = setup_backend()

import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import pandas as pd
from shapely.geometry import Polygon as ShapelyPolygon
import json
from skimage import measure
import traceback
import time
import threading
from typing import List, Dict, Optional, Tuple, Any
import warnings
import cv2
import subprocess

warnings.filterwarnings('ignore')

# ==================== 跨平台文件选择模块 ====================

def select_image_cross_platform(title="选择图片", parent=None):
    """
    跨平台文件选择对话框
    macOS: 使用AppleScript避免子线程GUI崩溃
    Windows/Linux: 使用标准Tkinter
    """
    system = platform.system()
    
    if system == "Darwin":  # macOS
        return _select_image_mac(title)
    else:
        return _select_image_tk(title, parent)

def _select_image_mac(title):
    """macOS专用文件选择 - AppleScript方案"""
    
    # 方案1: AppleScript（推荐，避免所有GUI线程问题）
    try:
        script = f'''
        tell application "System Events"
            activate
            set imageFile to choose file with prompt "{title}" of type {{"jpg", "jpeg", "png", "tif", "tiff", "bmp", "gif"}}
            return POSIX path of imageFile
        end tell
        '''
        result = subprocess.run(
            ['osascript', '-e', script],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and result.stdout.strip():
            path = result.stdout.strip()
            print(f"✅ AppleScript选择成功: {path}")
            return path
            
    except Exception as e:
        print(f"⚠️ AppleScript失败: {e}")
    
    # 方案2: PyQt6（需要安装: pip install PyQt6）
    try:
        from PyQt6.QtWidgets import QApplication, QFileDialog
        
        # 获取或创建应用实例
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        dialog = QFileDialog()
        dialog.setWindowTitle(title)
        dialog.setNameFilters([
            "Image files (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.gif)",
            "All files (*.*)"
        ])
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        
        if dialog.exec() == QFileDialog.DialogCode.Accepted:
            files = dialog.selectedFiles()
            if files:
                print(f"✅ PyQt6选择成功: {files[0]}")
                return files[0]
                
    except ImportError:
        print("⚠️ PyQt6未安装，跳过")
    except Exception as e:
        print(f"⚠️ PyQt6失败: {e}")
    
    # 方案3: 命令行输入（终极回退）
    return _fallback_cli_input(title)

def _select_image_tk(title, parent=None):
    """Windows/Linux标准Tkinter方案"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = parent
        if root is None:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
        
        # 更新窗口防止卡顿
        root.update()
        
        path = filedialog.askopenfilename(
            title=title,
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.gif"),
                ("All files", "*.*")
            ],
            parent=root
        )
        
        if parent is None:
            root.destroy()
        
        if path:
            print(f"✅ Tkinter选择成功: {path}")
            return path
        else:
            print("⚠️ 用户取消选择")
            return None
            
    except Exception as e:
        print(f"⚠️ Tkinter失败: {e}")
        return _fallback_cli_input(title)

def _fallback_cli_input(title):
    """终极回退 - 命令行输入"""
    print(f"\n{'='*60}")
    print(f"📁 {title}")
    print(f"{'='*60}")
    print("💡 提示: 可将图片文件拖拽到终端窗口")
    print("         或手动输入完整路径")
    print(f"{'='*60}")
    
    try:
        path = input("请输入图片路径: ").strip().strip('\'"')
        if os.path.exists(path):
            print(f"✅ 路径有效: {path}")
            return path
        else:
            print(f"❌ 路径不存在: {path}")
            return None
    except KeyboardInterrupt:
        print("\n⚠️ 用户取消")
        return None


# ==================== 原有导入保持不变 ====================

try:
    from mobile_sam import sam_model_registry, SamPredictor
    MOBILESAM_AVAILABLE = True
except ImportError:
    MOBILESAM_AVAILABLE = False
    print("⚠️ MobileSAM库未安装")

try:
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from geometry.grain_metric import GrainShapeMetrics
    from geometry.config_loader import load_geometry_config
    from geometry.export_csv import select_columns_for_grain_statistics_csv
    
    GEOMETRY_AVAILABLE = True
    print("✅ geometry模块加载成功")
except ImportError as e:
    GEOMETRY_AVAILABLE = False
    print(f"⚠️ geometry模块不可用: {e}")

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
    print(f"⚠️ Core segmentation functions unavailable: {e}")


# ==================== 主类（修改文件选择部分） ====================

class PureMobileSAMInteractiveEnhanced:
    """增强版纯交互式MobileSAM（跨平台兼容）"""
    
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
        
        self.output_dir = Path("interactive_results")
        self.output_dir.mkdir(exist_ok=True)
        
        self.geometry_config = None
        if GEOMETRY_AVAILABLE:
            try:
                config_path = Path(__file__).parent / "configs" / "geometry.yaml"
                if config_path.exists():
                    self.geometry_config = load_geometry_config(str(config_path))
                    print("✅ geometry配置加载成功")
            except Exception as e:
                print(f"⚠️ 加载geometry配置失败: {e}")
        
        self.scale_factor = None
        self.scale_detection_success = False
        
        self.start_time = None
        self.total_grains = 0
        self.total_interactions = 0
        
        self.gui_running = False
        
        print("🚀 MobileSAM交互式系统（跨平台兼容版）")
        print(f"   操作系统: {platform.system()}")
        print(f"   后端: {backend}")
        
        self._load_sam_model()
    
    def _load_sam_model(self) -> bool:
        """加载MobileSAM模型"""
        if not MOBILESAM_AVAILABLE:
            print("❌ MobileSAM库不可用")
            return False
        
        try:
            print(f"📦 加载MobileSAM模型: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                print(f"❌ 模型文件不存在: {self.model_path}")
                return False
            
            sam = sam_model_registry[self.model_type](checkpoint=self.model_path)
            sam.to(device=self.device)
            
            self.predictor = SamPredictor(sam)
            self.model_loaded = True
            
            print(f"✅ MobileSAM模型加载成功 (设备: {self.device}, 类型: {self.model_type})")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            traceback.print_exc()
            return False
    
    # ==================== 关键修复：移除子线程文件选择 ====================
    
    def load_image_with_gui(self) -> bool:
        """通过GUI文件选择对话框加载图片（跨平台修复版）"""
        if not self.model_loaded or self.predictor is None:
            print("❌ 模型未加载，无法处理图片")
            return False
        
        try:
            print("📂 请选择岩石显微图像...")
            
            # 直接使用跨平台选择函数（避免子线程）
            file_path = select_image_cross_platform("请选择岩石显微图像")
            
            if not file_path:
                print("⚠️ 未选择文件，退出交互模式")
                return False
            
            if not os.path.exists(file_path):
                print(f"❌ 图片文件不存在: {file_path}")
                return False
            
            return self._load_image_internal(file_path)
            
        except Exception as e:
            print(f"❌ 图片加载失败: {e}")
            traceback.print_exc()
            return False
    
    def load_image_from_path(self, image_path: str) -> bool:
        """直接从路径加载图片"""
        if not self.model_loaded or self.predictor is None:
            print("❌ 模型未加载，无法处理图片")
            return False
        
        if not os.path.exists(image_path):
            print(f"❌ 图片文件不存在: {image_path}")
            return False
        
        return self._load_image_internal(image_path)
    
    def _load_image_internal(self, file_path: str) -> bool:
        """内部图片加载逻辑"""
        try:
            print(f"📷 加载图片: {file_path}")
            pil_image = Image.open(file_path).convert('RGB')
            self.image = np.array(pil_image)
            self.image_path = file_path
            
            print("⚙️  设置图像到SAM预测器...")
            self.predictor.set_image(self.image)
            
            # 重置状态
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
            
            print(f"✅ 图片加载成功: {self.image.shape}")
            return True
            
        except Exception as e:
            print(f"❌ 图片加载失败: {e}")
            traceback.print_exc()
            return False
    
    # ==================== 以下方法保持不变（省略以节省空间） ====================
    
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
                                continue
                    
                    # 回退到skimage
                    contours = measure.find_contours(mask, 0.5)
                    if len(contours) > 0:
                        main_contour = max(contours, key=lambda x: len(x))
                        if len(main_contour) >= 3:
                            polygon_points = [(point[1], point[0]) for point in main_contour]
                            polygon = ShapelyPolygon(polygon_points)
                            if polygon.is_valid and polygon.area > 0:
                                polygons.append(polygon)
                    
                except Exception as e:
                    print(f"⚠️ 转换掩码为多边形失败（颗粒#{grain['id']}）: {e}")
        
        return polygons
    
    def _generate_unified_grain_dataframe(self) -> pd.DataFrame:
        """生成与YOLO流程完全一致的颗粒DataFrame"""
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
                        
                        # 计算 major_axis_length 和 minor_axis_length 用于几何参数计算
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
                            # 如果 skimage 不可用则使用近似值
                            major_axis_length = max(width, height)
                            minor_axis_length = min(width, height)
                        
                        basic_data.append({
                            'grain_id': grain['id'],
                            'area': float(area),
                            'centroid_x': float(centroid_x),
                            'centroid_y': float(centroid_y),
                            'width': float(width),
                            'height': float(height),
                            'perimeter': float(perimeter),
                            'confidence': float(confidence),
                            'mask_area_pixels': int(area),
                            'major_axis_length': float(major_axis_length),
                            'minor_axis_length': float(minor_axis_length)
                        })
            
            if not basic_data:
                return pd.DataFrame()
            
            basic_df = pd.DataFrame(basic_data)
            
            if GEOMETRY_AVAILABLE and len(self.polygons) > 0:
                try:
                    shape_calculator = GrainShapeMetrics(basic_df)
                    geometry_df = shape_calculator.compute_all_metrics()
                    
                    existing_columns = [col for col in geometry_df.columns 
                                      if col in ['grain_id', 'area', 'perimeter', 'circularity',
                                                'aspect_ratio', 'compactness', 'centroid_x',
                                                'centroid_y', 'width', 'height', 'confidence']]
                    geometry_df = geometry_df[existing_columns]
                    
                    print(f"✅ 高级几何参数计算完成，共{len(geometry_df.columns)}个参数")
                    return geometry_df
                    
                except Exception as e:
                    print(f"⚠️ GrainShapeMetrics计算失败: {e}")
                    return basic_df
            else:
                print("ℹ️  geometry模块不可用，使用基础几何参数")
                return basic_df
                
        except Exception as e:
            print(f"❌ 生成颗粒数据失败: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    
    def _generate_complete_outputs(self, output_dir: Optional[Path] = None) -> Path:
        """生成完整输出文件（与YOLO流程完全一致）"""
        if len(self.grains) == 0:
            print("⚠️ 没有分割颗粒，无法生成输出文件")
            return None
        
        if output_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_name = Path(self.image_path).stem if self.image_path else "interactive"
            output_dir = self.output_dir / f"{image_name}_{timestamp}"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 生成完整输出到: {output_dir}")
        
        try:
            self.polygons = self._masks_to_polygons()
            
            if len(self.polygons) == 0:
                print("❌ 无法生成有效多边形")
                return None
            
            print(f"✅ 生成了 {len(self.polygons)} 个多边形")
            
            self.grain_data = self._generate_unified_grain_dataframe()
            
            if self.grain_data.empty:
                print("❌ 无法生成颗粒数据")
                return None
            
            print(f"📊 颗粒数据包含 {len(self.grain_data.columns)} 个参数")
            
            # 保存可视化
            if self.fig is not None:
                vis_path = output_dir / "segmentation_result.png"
                self.fig.savefig(vis_path, dpi=300, bbox_inches='tight')
                print(f"💾 交互式界面截图保存至: {vis_path}")
                
                self._generate_yolo_style_visualization(output_dir)
            
            self._generate_simple_masks(output_dir)
            
            # 保存CSV
            if not self.grain_data.empty:
                csv_path = output_dir / "grain_statistics.csv"
                
                if GEOMETRY_AVAILABLE and self.geometry_config:
                    try:
                        grain_data_to_save = select_columns_for_grain_statistics_csv(
                            self.grain_data,
                            self.geometry_config,
                            strict=False
                        )
                        if grain_data_to_save is None or grain_data_to_save.empty:
                            grain_data_to_save = self.grain_data
                    except Exception as e:
                        print(f"⚠️ 配置筛选失败: {e}")
                        grain_data_to_save = self.grain_data
                else:
                    grain_data_to_save = self.grain_data
                
                grain_data_to_save.to_csv(csv_path, index=False, encoding='utf-8')
                print(f"💾 颗粒统计表保存至: {csv_path}")
                
                self._print_statistics_summary(grain_data_to_save)
            
            # 保存JSON
            summary = self._create_yolo_style_summary()
            json_path = output_dir / "summary.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"💾 JSON摘要保存至: {json_path}")
            
            self._save_performance_info(output_dir)
            
            print(f"✅ 所有结果已保存到: {output_dir}")
            return output_dir
            
        except Exception as e:
            print(f"❌ 生成完整输出失败: {e}")
            traceback.print_exc()
            return None
    
    def _generate_yolo_style_visualization(self, output_dir: Path):
        """生成YOLO风格的可视化图"""
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
            
            print(f"💾 YOLO风格可视化图保存至: {plot_path}")
            
        except Exception as e:
            print(f"⚠️ 生成YOLO风格可视化图失败: {e}")
    
    def _generate_simple_masks(self, output_dir: Path):
        """生成分割掩码图"""
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
        print(f"💾 分割掩码图保存至: {mask_path}")
        
        self.mask_all = mask_all
    
    def _print_statistics_summary(self, grain_data: pd.DataFrame = None):
        """打印统计摘要"""
        if grain_data is None:
            grain_data = self.grain_data
            
        if grain_data is None or grain_data.empty:
            return
        
        print(f"\n📈 颗粒统计摘要:")
        print(f"  总颗粒数: {len(grain_data)}")
        
        if 'area' in grain_data.columns:
            area_sum = grain_data['area'].sum()
            area_mean = grain_data['area'].mean()
            print(f"  总像素面积: {area_sum:.0f}")
            print(f"  平均像素面积: {area_mean:.1f}")
    
    def _create_yolo_style_summary(self) -> Dict[str, Any]:
        """创建与YOLO流程一致的JSON摘要"""
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
            'system_version': 'MobileSAM Interactive macOS v2.0',
            'processing_mode': 'interactive',
            'platform': platform.system()
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
        
        return summary
    
    def _save_performance_info(self, output_dir: Path):
        """保存性能信息"""
        processing_time = time.time() - self.start_time if self.start_time else 0
        
        performance = {
            "processing_time_seconds": float(processing_time),
            "total_grains": len(self.grains),
            "total_interactions": self.total_interactions,
            "average_time_per_grain": float(processing_time / max(len(self.grains), 1)),
            "average_points_per_grain": float(self.total_interactions / max(len(self.grains), 1)),
            "average_confidence": float(self.grain_data['confidence'].mean()) 
                if not self.grain_data.empty and 'confidence' in self.grain_data.columns else 0,
            "version": "interactive_macos_v2.0",
            "platform": platform.system()
        }
        
        perf_path = output_dir / "performance.json"
        with open(perf_path, 'w', encoding='utf-8') as f:
            json.dump(performance, f, indent=2, ensure_ascii=False)
        print(f"💾 性能信息保存至: {perf_path}")
    
    # ==================== 交互式界面方法（保持不变） ====================
    
    def _get_grain_at_point(self, x: float, y: float) -> Optional[int]:
        """获取点击位置所在的颗粒ID"""
        for grain in self.grains:
            if 'mask' in grain and grain['mask'] is not None:
                h, w = grain['mask'].shape
                ix, iy = int(x), int(y)
                
                if 0 <= ix < w and 0 <= iy < h:
                    if grain['mask'][iy, ix]:
                        return grain['id']
        return None
    
    def _create_new_grain(self, x: float, y: float, is_foreground: bool = True) -> int:
        """创建新颗粒"""
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
        print(f"➕ 创建新颗粒 #{self.current_grain_id}")
        
        return self.current_grain_id
    
    def _add_point_to_current_grain(self, x: float, y: float, is_foreground: bool = True):
        """添加点到当前颗粒"""
        if not self.grains:
            print("⚠️ 没有当前颗粒，先创建新颗粒")
            return
        
        current_grain = self.grains[-1]
        current_grain['points'].append({
            'x': x, 
            'y': y, 
            'is_foreground': is_foreground
        })
        
        self.total_interactions += 1
        point_type = "前景点" if is_foreground else "背景点"
        print(f"➕ 为颗粒 #{current_grain['id']} 添加{point_type}: ({x:.1f}, {y:.1f})")
    
    def _run_sam_segmentation_for_grain(self, grain_id: int):
        """为指定颗粒执行SAM分割"""
        grain = None
        for g in self.grains:
            if g['id'] == grain_id:
                grain = g
                break
        
        if grain is None or not grain['points']:
            print(f"⚠️ 颗粒 #{grain_id} 没有点，无法分割")
            return
        
        try:
            input_points = []
            input_labels = []
            
            for point in grain['points']:
                input_points.append([point['x'], point['y']])
                input_labels.append(1 if point['is_foreground'] else 0)
            
            input_points = np.array(input_points, dtype=np.float32)
            input_labels = np.array(input_labels, dtype=np.int32)
            
            print(f"🔍 分割颗粒 #{grain_id}: {len(input_points)}个提示点")
            
            start_time = time.time()
            masks, scores, _ = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True
            )
            inference_time = time.time() - start_time
            
            if len(masks) == 0:
                print("⚠️ 未生成任何掩码")
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
            
            print(f"✅ 颗粒 #{grain_id} 分割成功! 置信度: {score:.3f}, 耗时: {inference_time:.3f}s")
            
            self._update_grain_display(grain_id)
            
        except Exception as e:
            print(f"❌ 分割颗粒 #{grain_id} 失败: {e}")
            traceback.print_exc()
    
    def _draw_grain_with_text(self, grain):
        """绘制单个颗粒及其文本标签"""
        try:
            grain_id = grain['id']
            mask = grain['mask']
            
            if mask is None or not np.any(mask):
                return
            
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
            print(f"⚠️ 绘制颗粒 #{grain.get('id', '未知')} 失败: {e}")
    
    def _update_grain_display(self, grain_id: int):
        """更新颗粒显示"""
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
        """刷新所有颗粒的显示"""
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
            print(f"🔄 已刷新显示，当前颗粒数: {len(self.grains)}")
        
        except Exception as e:
            print(f"⚠️ 刷新显示失败: {e}")
    
    def _on_mouse_click(self, event):
        """鼠标点击事件处理"""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        
        x, y = event.xdata, event.ydata
        
        clicked_grain_id = self._get_grain_at_point(x, y)
        
        if event.button == 1:  # 左键：前景点
            color = 'lime'
            marker = 'o'
            is_foreground = True
            point_type = "前景点"
        else:  # 右键：背景点
            color = 'red'
            marker = 'x'
            is_foreground = False
            point_type = "背景点"
        
        marker_obj = self.ax.plot(x, y, marker=marker, color=color, 
                                markersize=10, markeredgewidth=2, alpha=0.8)
        self.point_markers.append(marker_obj[0])
        
        if clicked_grain_id is not None:
            print(f"👆 点击颗粒 #{clicked_grain_id}，添加{point_type}")
            self._add_point_to_current_grain(x, y, is_foreground)
            self._run_sam_segmentation_for_grain(clicked_grain_id)
        else:
            print(f"👆 创建新颗粒，添加{point_type}")
            new_grain_id = self._create_new_grain(x, y, is_foreground)
            self._run_sam_segmentation_for_grain(new_grain_id)
        
        self.fig.canvas.draw()
    
    def _on_key_press(self, event):
        """键盘按键事件处理"""
        if event.key == 'x':  # 删除最后一个颗粒
            self._delete_last_grain()
        elif event.key == 'd':  # 删除所有颗粒
            self._delete_all_grains()
        elif event.key == 's':  # 保存结果
            self._generate_complete_outputs()
        elif event.key == 'c':  # 清除所有点标记
            self._clear_point_markers()
        elif event.key == 'r':  # 重新开始
            self._reset_interface()
        elif event.key == 'q':  # 退出
            print("👋 退出交互界面")
            self.gui_running = False
            plt.close(self.fig)
        elif event.key == 'h':  # 显示帮助
            self._show_help()
    
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
            print(f"🗑️  已删除颗粒 #{grain_id}")
            
            self._refresh_grain_display()
        else:
            print("⚠️ 没有颗粒可删除")
    
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
        print("🗑️  已删除所有颗粒")
    
    def _clear_point_markers(self):
        """清除所有点标记"""
        for marker in self.point_markers:
            marker.remove()
        self.point_markers = []
        self.fig.canvas.draw()
        print("🧹 已清除所有点标记")
    
    def _reset_interface(self):
        """重置整个界面"""
        self._delete_all_grains()
        self._clear_point_markers()
        
        self.ax.clear()
        self.ax.imshow(self.image)
        
        image_name = Path(self.image_path).name if self.image_path else "未命名图片"
        title_text = f"MobileSAM增强版交互式分割 - {image_name}"
        self.ax.set_title(title_text, fontsize=16)
        
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        self._show_help_text_fixed()
        self.fig.canvas.draw()
        
        print("🔄 界面已完全重置")
    
    def _show_help(self):
        """显示帮助信息"""
        help_text = (
            " 交互指南:\n\n"
            "鼠标操作:\n"
            "• 左键点击（绿色圆点）: 标记颗粒位置\n"
            "• 右键点击（红色叉号）: 标记非颗粒位置\n\n"
            "键盘快捷键:\n"
            "• 's': 保存完整结果\n"
            "• 'x': 删除最后一个颗粒\n"
            "• 'd': 删除所有颗粒\n"
            "• 'c': 清除点标记\n"
            "• 'r': 重置界面\n"
            "• 'q': 退出程序\n"
            "• 'h': 显示此帮助\n"
        )
        
        # macOS使用print，Windows/Linux使用messagebox
        if platform.system() == "Darwin":
            print(f"\n{'='*50}")
            print(help_text)
            print(f"{'='*50}")
        else:
            try:
                from tkinter import messagebox
                messagebox.showinfo("交互式分割帮助", help_text)
            except:
                print(help_text)
    
    def _show_help_text_fixed(self):
        """在界面上显示帮助文本"""
        help_text = (
            "增强版交互指南：\n"
            "• 左键点击：标记颗粒位置（前景点）\n"
            "• 右键点击：标记非颗粒位置（背景点）\n"
            "• 's'键：保存完整结果\n"
            "• 'x'键：删除最后一个颗粒\n"
            "• 'd'键：删除所有颗粒\n"
            "• 'c'键：清除点标记\n"
            "• 'r'键：重置界面\n"
            "• 'q'键：退出程序\n"
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
            pass
    
    def show_interactive_interface(self):
        """显示交互式界面"""
        if self.image is None:
            print("❌ 请先加载图片")
            return
        
        if self.predictor is None:
            print("❌ SAM预测器未初始化")
            return
        
        print("🖥️  正在创建交互式界面...")
        
        try:
            # 设置中文字体
            from matplotlib import rcParams
            
            system = platform.system()
            if system == 'Windows':
                rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
            elif system == 'Darwin':
                rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB']
            else:
                rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Zen Hei']
            
            rcParams['axes.unicode_minus'] = False
            
            self.fig, self.ax = plt.subplots(figsize=(14, 10))
            self.ax.imshow(self.image)
            
            image_name = Path(self.image_path).name if self.image_path else "未命名图片"
            title_text = f"MobileSAM增强版交互式分割 - {image_name}"
            self.ax.set_title(title_text, fontsize=16)
            
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            
            self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_click)
            self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
            
            self._show_help_text_fixed()
            
            plt.tight_layout()
            
            print("✅ 增强版交互式界面已启动")
            
            self.gui_running = True
            
            if backend == 'Agg':
                print("⚠️ 检测到无GUI环境，将保存结果图片")
                output_path = self.output_dir / "interactive_result.png"
                self.fig.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"💾 结果已保存到: {output_path}")
                plt.close(self.fig)
                return
            
            print("🎯 交互窗口已打开，您可以开始标记颗粒")
            print("💡 按 's' 保存完整结果，'h' 显示帮助")
            
            plt.show(block=True)
            
            print("👋 交互式窗口已关闭")
            
        except Exception as e:
            print(f"❌ 显示交互界面失败: {e}")
            traceback.print_exc()
    
    def run_interactive_mode(self, image_path: str = None):
        """运行完整的交互式模式"""
        try:
            if not self.model_loaded:
                print("❌ 模型未加载，无法运行交互模式")
                return
            
            if image_path:
                print(f"📷 加载指定图片: {image_path}")
                if not self.load_image_from_path(image_path):
                    print("❌ 图片加载失败，退出交互模式")
                    return
            else:
                print("📂 请选择图片...")
                if not self.load_image_with_gui():
                    print("❌ 图片选择失败，退出交互模式")
                    return
            
            print("🚀 启动交互式界面...")
            self.show_interactive_interface()
            
        except Exception as e:
            print(f"❌ 交互模式运行失败: {e}")
            traceback.print_exc()


def main():
    """主函数：直接运行增强版交互式MobileSAM"""
    print("="*70)
    print("🚀 MobileSAM增强版交互式分割系统（跨平台兼容版）")
    print(f"   平台: {platform.system()} {platform.release()}")
    print(f"   Python: {platform.python_version()}")
    print("="*70)
    
    model_path = "models/mobile_sam.pt"
    device = "cpu"
    model_type = "vit_t"
    
    # macOS MPS支持检测
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
            print("✅ 检测到Apple Silicon MPS加速")
    except:
        pass
    
    interactive_system = PureMobileSAMInteractiveEnhanced(
        model_path=model_path,
        device=device,
        model_type=model_type
    )
    
    interactive_system.run_interactive_mode()


if __name__ == "__main__":
    main()