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


    def calculate_2d_zingg_parameters(self) -> pd.DataFrame:
        """
        在 2D 场景下，L=主轴长, S=I=副轴长
        """
        l = self.grain_data['major_axis_length']
        s = self.grain_data['minor_axis_length']
        
        # 2D 适配版
        ei = s / l  # 2D 伸长率
        fi = 1.0    # 2D 无法体现扁平化，设为常数或忽略
        ar = s / l  # 2D 长宽比
        
        return pd.DataFrame({
            'EI_2d': ei,
            'FI_2d': fi,
            'AR_2d': ar
        })

    def calculate_fourier_descriptors(self, n_coeffs=25) -> pd.DataFrame:
        """
        2D 轮廓的傅里叶描述符（对应 3D 的球谐函数）
        用于描述从整体到细节的形状特征
        """
        results = []
        for _, grain in self.grain_data.iterrows():
            coords = np.array(grain['coordinates'])
            
            # 1. 将坐标转换为复数形式 x + iy
            complex_coords = coords[:, 0] + 1j * coords[:, 1]
            
            # 2. 离散傅里叶变换
            coeffs = np.fft.fft(complex_coords)
            
            # 3. 归一化（消除平移、旋转、缩放影响）
            # coeffs[0] 是直流分量（中心位置），忽略
            # 用 coeffs[1] 归一化以实现尺度不变性
            abs_coeffs = np.abs(coeffs)
            if abs_coeffs[1] != 0:
                normalized_coeffs = abs_coeffs / abs_coeffs[1]
            else:
                normalized_coeffs = np.zeros(len(abs_coeffs))

            # 4. 提取类似 Dn 的特征 (取前 n_coeffs 个)
            # D2 对应 normalized_coeffs[2], 依次类推
            d_vals = []
            for i in range(2, min(n_coeffs + 2, len(normalized_coeffs))):
                d_vals.append(normalized_coeffs[i])
            
            # 如果系数不够，补齐
            while len(d_vals) < 3: 
                d_vals.append(0)
                
            results.append(d_vals[:3]) # 仅取 D2, D3, D4 演示

        return pd.DataFrame(results, columns=['D2_2d', 'D3_2d', 'D4_2d'])

    def calculate_sh_equivalent_fd(self, beta):
        """
        FD = (6 + beta) / 2
        beta 通常通过 log(Fourier_Coeff) vs log(n) 的斜率获得
        """
        return (6 + beta) / 2

    def calculate_convexity(self) -> pd.Series:
        """
        计算颗粒的凸度 (Convexity)
        
        公式: C = A / A_hull
        其中 A 是颗粒面积，A_hull 是颗粒二维投影轮廓凸包的面积。
        """
        # 如果 grain_data 是通过 skimage.measure.regionprops 获得的，
        # 通常已经包含了 'solidity' 属性，它等同于此处的凸度。
        if 'solidity' in self.grain_data.columns:
            return self.grain_data['solidity']
        
        # 如果没有预计算好的 solidity，则通过坐标手动计算
        convexity_list = []
        for _, grain in self.grain_data.iterrows():
            coords = np.array(grain['coordinates'])
            area = grain['area']
            
            # 使用 scipy.spatial.ConvexHull 计算凸包
            try:
                hull = ConvexHull(coords)
                # hull.volume 在 2D 中代表面积 (Area)
                a_hull = hull.volume 
                convexity_list.append(area / a_hull)
            except Exception:
                # 针对坐标点不足以构成凸包的情况处理
                convexity_list.append(np.nan)
                
        return pd.Series(convexity_list)
        
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
        
            #return np.array([box_counting(grain['coordinates']) for _, grain in self.grain_data.iterrows()])
        
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
        self.grain_data['fractal_dimension'] = self.calculate_fractal_dimension()
        self.grain_data['convexity'] = self.calculate_convexity()
        
        return self.grain_data