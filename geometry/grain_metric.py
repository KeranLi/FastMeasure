import numpy as np
import pandas as pd
from skimage.measure import regionprops
from skimage import measure
from scipy.spatial import ConvexHull

class GrainShapeMetrics:
    """
    颗粒轮廓计算工具类，计算各种颗粒形状的结构参数
    """
    
    def __init__(self, grain_data: pd.DataFrame):
        """
        初始化方法
        
        Args:
            grain_data (pd.DataFrame): 包含颗粒区域数据的DataFrame，必须包含 'area', 'perimeter', 'coordinates' 等列
        """
        self.grain_data = grain_data
        
    def calculate_circularity(self) -> pd.Series:
        """
        计算颗粒的圆形度
        
        公式: Circularity = 4 * π * Area / Perimeter^2
        """
        area = self.grain_data['area']
        perimeter = self.grain_data['perimeter']
        return 4 * np.pi * area / (perimeter ** 2)
    
    def calculate_aspect_ratio(self) -> pd.Series:
        """
        计算颗粒的长宽比
        
        公式: Aspect Ratio = Major Axis Length / Minor Axis Length
        """
        major_axis_length = self.grain_data['major_axis_length']
        minor_axis_length = self.grain_data['minor_axis_length']
        return major_axis_length / minor_axis_length
    
    def calculate_rectangularity(self) -> pd.Series:
        """
        计算颗粒的矩形度
        
        公式: Rectangularity = Area / (Major Axis Length * Minor Axis Length)
        """
        area = self.grain_data['area']
        major_axis_length = self.grain_data['major_axis_length']
        minor_axis_length = self.grain_data['minor_axis_length']
        return area / (major_axis_length * minor_axis_length)
    
    def calculate_compactness(self) -> pd.Series:
        """
        计算颗粒的压实度
        
        公式: Compactness = Perimeter^2 / (4 * π * Area)
        """
        perimeter = self.grain_data['perimeter']
        area = self.grain_data['area']
        return perimeter ** 2 / (4 * np.pi * area)
    
    def calculate_fractal_dimension(self) -> pd.Series:
        """
        计算颗粒的分形维数
        
        使用盒子计数法计算分形维数
        """
        def box_counting(coords):
            """计算盒子计数法的分形维数"""
            coords = np.array(coords)
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            box_size = max(max_coords - min_coords) / 10
            count = 0
            for box_x in np.arange(min_coords[0], max_coords[0], box_size):
                for box_y in np.arange(min_coords[1], max_coords[1], box_size):
                    if np.any((coords[:, 0] >= box_x) & (coords[:, 0] < box_x + box_size) &
                              (coords[:, 1] >= box_y) & (coords[:, 1] < box_y + box_size)):
                        count += 1
            return np.log(count) / np.log(1 / box_size)
        
            return np.array([box_counting(grain['coordinates']) for _, grain in self.grain_data.iterrows()])
        
    def calculate_angularity(self) -> pd.Series:
        """
        计算颗粒的棱角度
        
        通过计算颗粒轮廓的锐角程度来估算
        """
        angularity = []
        for _, grain in self.grain_data.iterrows():
            coords = grain['coordinates']
            hull = ConvexHull(coords)
            angularity.append(len(hull.vertices))  # 计算边界的角点数作为棱角度的简单估计
        return pd.Series(angularity)
    
    def calculate_roundness(self) -> pd.Series:
        """
        计算颗粒的磨圆度
        
        公式: Roundness = Perimeter^2 / (4 * π * Area)
        """
        perimeter = self.grain_data['perimeter']
        area = self.grain_data['area']
        return perimeter ** 2 / (4 * np.pi * area)
    
    def compute_all_metrics(self):
        """
        计算所有颗粒的形状参数
        """
        self.grain_data['circularity'] = self.calculate_circularity()
        self.grain_data['aspect_ratio'] = self.calculate_aspect_ratio()
        self.grain_data['rectangularity'] = self.calculate_rectangularity()
        self.grain_data['compactness'] = self.calculate_compactness()
        self.grain_data['fractal_dimension'] = self.calculate_fractal_dimension()
        self.grain_data['angularity'] = self.calculate_angularity()
        self.grain_data['roundness'] = self.calculate_roundness()
        
        return self.grain_data