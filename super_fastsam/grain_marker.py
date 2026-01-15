"""
岩石颗粒标注模块
文件名：grain_marker.py
功能：在岩石分割结果图上添加颗粒编号和面积标注
特点：支持无背景标注，自动调整位置避免重叠
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import matplotlib.patheffects as path_effects  # 修复导入


def add_grain_labels(
    ax: plt.Axes,
    grain_data: pd.DataFrame,
    image_shape: Tuple[int, int],
    scale_factor: Optional[float] = None,
    font_size: int = 11,
    text_color: str = 'yellow',
    bg_color: Optional[str] = None,
    show_area: bool = True,
    max_labels: int = 1000,
    min_area: int = 0,
    text_outline: bool = True,
    outline_color: str = 'black',
    outline_width: float = 2.0
) -> plt.Axes:
    """
    在岩石分割图上添加颗粒编号和面积标注
    
    参数:
        ax: matplotlib坐标轴对象
        grain_data: 颗粒数据，必须包含['label', 'centroid-0', 'centroid-1', 'area']列
        image_shape: 图像尺寸 (高度, 宽度, 通道数)
        scale_factor: 比例因子 (μm/像素)，如果有则显示真实面积
        font_size: 字体大小
        text_color: 文字颜色
        bg_color: 背景框颜色，None或空字符串表示无背景
        show_area: 是否显示面积
        max_labels: 密集区域最大标注数
        min_area: 最小标注面积（像素）
        text_outline: 是否添加文字描边
        outline_color: 描边颜色
        outline_width: 描边宽度
        
    返回:
        更新后的坐标轴对象
    """
    
    # 1. 检查必需的数据列
    required_columns = ['label', 'centroid-0', 'centroid-1', 'area']
    for col in required_columns:
        if col not in grain_data.columns:
            print(f"⚠️ 警告: 颗粒数据缺少列 '{col}'，跳过标注")
            return ax
    
    # 2. 确保数据是数值类型
    for col in required_columns:
        grain_data[col] = pd.to_numeric(grain_data[col], errors='coerce')
    
    # 3. 过滤掉无效数据和小颗粒
    valid_data = grain_data.dropna(subset=required_columns)
    valid_data = valid_data[valid_data['area'] >= min_area]
    
    if len(valid_data) == 0:
        return ax
    
    # 4. 按面积从大到小排序，优先标注大颗粒
    sorted_data = valid_data.sort_values('area', ascending=False)
    
    # 5. 限制最大标注数
    if len(sorted_data) > max_labels:
        sorted_data = _filter_dense_areas(sorted_data, image_shape, max_labels)
    
    # 6. 记录已标注位置，避免重叠
    used_positions = []
    
    # 7. 为每个颗粒添加标注
    for _, row in sorted_data.iterrows():
        label_num = int(row['label'])
        centroid_y = row['centroid-0']  # 行坐标 (y)
        centroid_x = row['centroid-1']  # 列坐标 (x)
        
        # 构建标注文本
        text = _create_label_text(label_num, row['area'], scale_factor, show_area)
        
        # 自动调整位置避免重叠
        final_x, final_y = _find_available_position(
            centroid_x, centroid_y, used_positions, image_shape
        )
        
        # 如果找到合适位置，添加标注
        if final_x is not None and final_y is not None:
            _add_single_label(
                ax, final_x, final_y, text, font_size, text_color, bg_color,
                text_outline, outline_color, outline_width
            )
            used_positions.append((final_x, final_y))
    
    return ax


def _filter_dense_areas(
    grain_data: pd.DataFrame,
    image_shape: Tuple[int, int],
    max_labels: int
) -> pd.DataFrame:
    """
    筛选密集区域的颗粒，优先保留大面积和稀疏区域的颗粒
    """
    # 计算每个颗粒周围的密度
    densities = _calculate_grain_densities(grain_data, image_shape)
    grain_data['density'] = densities
    
    # 计算综合评分：大面积+低密度
    grain_data['score'] = (
        grain_data['area'] / grain_data['area'].max() * 0.7 +  # 面积权重70%
        (1 - grain_data['density']) * 0.3  # 稀疏度权重30%
    )
    
    # 按评分排序，取前max_labels个
    return grain_data.sort_values('score', ascending=False).head(max_labels)


def _calculate_grain_densities(
    grain_data: pd.DataFrame,
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    计算每个颗粒周围的密度
    """
    if len(grain_data) == 0:
        return np.array([])
    
    # 获取质心坐标 (x, y)
    centroids = grain_data[['centroid-1', 'centroid-0']].values
    
    # 设置密度计算半径（图像尺寸的5%）
    density_radius = min(image_shape[0], image_shape[1]) * 0.05
    
    densities = np.zeros(len(centroids))
    
    for i, (x, y) in enumerate(centroids):
        # 计算到所有其他颗粒的距离
        distances = np.sqrt(
            (centroids[:, 0] - x) ** 2 + 
            (centroids[:, 1] - y) ** 2
        )
        
        # 统计半径内的颗粒数（不包括自己）
        close_grains = np.sum(distances < density_radius) - 1
        densities[i] = close_grains
    
    # 归一化到0-1范围
    if np.max(densities) > 0:
        densities = densities / np.max(densities)
    
    return densities


def _create_label_text(
    label_num: int,
    area: float,
    scale_factor: Optional[float] = None,
    show_area: bool = True
) -> str:
    """
    创建标注文本
    """
    if not show_area:
        return f"{label_num}"
    
    if scale_factor:
        # 计算真实面积 (μm²)
        real_area = area * (scale_factor ** 2)
        if real_area > 1000:
            # 大于1000μm²时显示为mm²
            return f"{label_num}\n{real_area/1000:.1f}mm²"
        else:
            return f"{label_num}\n{real_area:.0f}μm²"
    else:
        # 显示像素面积
        return f"{label_num}\n{area:.0f}px"


def _find_available_position(
    x: float,
    y: float,
    used_positions: List[Tuple[float, float]],
    image_shape: Tuple[int, int],
    min_distance: float = 25.0  # 减小最小距离，允许更密集的标注
) -> Tuple[Optional[float], Optional[float]]:
    """
    寻找可用的标注位置，避免重叠
    """
    # 尝试的位置偏移
    position_offsets = [
        (0, 0),           # 原始位置
        (20, 0), (-20, 0), (0, 20), (0, -20),  # 上下左右
        (15, 15), (15, -15), (-15, 15), (-15, -15),  # 对角线
        (30, 0), (-30, 0), (0, 30), (0, -30),  # 更远的上下左右
        (10, 25), (10, -25), (-10, 25), (-10, -25),  # 斜向
    ]
    
    for dx, dy in position_offsets:
        new_x = x + dx
        new_y = y + dy
        
        # 检查是否在图像范围内
        if (0 <= new_x < image_shape[1] and 
            0 <= new_y < image_shape[0] and
            _is_position_available(new_x, new_y, used_positions, min_distance)):
            return new_x, new_y
    
    # 没有找到合适位置
    return None, None


def _is_position_available(
    x: float,
    y: float,
    used_positions: List[Tuple[float, float]],
    min_distance: float
) -> bool:
    """
    检查位置是否可用（与已有标注距离足够远）
    """
    for used_x, used_y in used_positions:
        distance = np.sqrt((x - used_x) ** 2 + (y - used_y) ** 2)
        if distance < min_distance:
            return False
    return True


def _add_single_label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    font_size: int,
    text_color: str,
    bg_color: Optional[str] = None,
    text_outline: bool = True,
    outline_color: str = 'black',
    outline_width: float = 2.0
) -> None:
    """
    在指定位置添加单个颗粒标注
    
    参数:
        ax: matplotlib坐标轴
        x: x坐标
        y: y坐标
        text: 标注文本
        font_size: 字体大小
        text_color: 文字颜色
        bg_color: 背景颜色，None表示无背景
        text_outline: 是否添加文字描边
        outline_color: 描边颜色
        outline_width: 描边宽度
    """
    if bg_color:
        # 有背景框的标注
        ax.text(
            x, y,
            text,
            fontsize=font_size,
            color=text_color,
            ha='center',
            va='center',
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=bg_color,
                edgecolor='black',
                alpha=0.8,
                linewidth=0.5
            ),
            zorder=10
        )
    else:
        # 无背景框，添加文字描边增强可读性
        text_obj = ax.text(
            x, y,
            text,
            fontsize=font_size,
            color=text_color,
            ha='center',
            va='center',
            zorder=10,
            weight='bold'  # 加粗字体
        )
        
        # 添加黑色描边
        if text_outline:
            text_obj.set_path_effects([
                path_effects.withStroke(  # 修复这里
                    linewidth=outline_width, 
                    foreground=outline_color
                )
            ])


def add_labels_with_config(
    ax: plt.Axes,
    grain_data: pd.DataFrame,
    image_shape: Tuple[int, int],
    config: dict
) -> plt.Axes:
    """
    使用配置字典添加颗粒标注（便捷函数）
    """
    # 提取配置参数
    font_size = config.get('font_size', 11)
    text_color = config.get('text_color', 'yellow')
    bg_color = config.get('bg_color', '')
    show_area = config.get('show_area', True)
    max_labels = config.get('max_labels', 1000)
    min_area = config.get('min_area', 0)
    text_outline = config.get('text_outline', True)
    outline_color = config.get('outline_color', 'black')
    outline_width = config.get('outline_width', 2.0)
    
    # 处理空字符串背景
    if bg_color == '':
        bg_color = None
    
    # 调用主函数
    return add_grain_labels(
        ax=ax,
        grain_data=grain_data,
        image_shape=image_shape,
        font_size=font_size,
        text_color=text_color,
        bg_color=bg_color,
        show_area=show_area,
        max_labels=max_labels,
        min_area=min_area,
        text_outline=text_outline,
        outline_color=outline_color,
        outline_width=outline_width
    )


# 测试代码
if __name__ == "__main__":
    print("✅ grain_marker.py 模块测试")
    print("功能: 为岩石颗粒添加编号和面积标注")
    print("使用方式:")
    print("1. 导入模块: from grain_marker import add_grain_labels")
    print("2. 调用函数: add_grain_labels(ax, grain_data, image_shape, ...)")
    print("\n配置说明:")
    print("  - bg_color: 'white' (有背景), ''或None (无背景)")
    print("  - text_color: 'yellow', 'white', 'black' 等")
    print("  - text_outline: True (无背景时建议开启)")
    print("\n模块已准备就绪！")