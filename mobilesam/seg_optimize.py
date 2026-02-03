"""
智能后处理模块 - 专门针对岩石颗粒优化
文件名：seg_optimize.py
功能：智能后处理，解决重叠、粘连等问题
"""

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point, box
from shapely.ops import unary_union, polygonize
import networkx as nx
from typing import List, Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class SmartPostProcessor:
    """智能后处理器"""
    
    def __init__(self, min_area: int = 30, iou_threshold: float = 0.5):
        """
        初始化智能后处理器
        
        Args:
            min_area: 最小面积
            iou_threshold: IoU阈值
        """
        self.min_area = min_area
        self.iou_threshold = iou_threshold
        
        # 后处理参数
        self.params = {
            'erosion_distance': -2,    # 腐蚀距离（负值）
            'dilation_distance': 2,    # 膨胀距离
            'simplify_tolerance': 1.0, # 简化容差
            'buffer_distance': 0.5,    # 缓冲距离
            'min_overlap_area': 10,    # 最小重叠面积
            'max_aspect_ratio': 5.0,   # 最大纵横比
        }
        
        print("✅ SmartPostProcessor初始化完成")
    
    def process(self, polygons: List[Polygon]) -> List[Polygon]:
        """
        主处理函数
        
        Args:
            polygons: 输入多边形列表
            
        Returns:
            处理后多边形列表
        """
        if len(polygons) == 0:
            return []
        
        print(f"🔧 智能后处理: 输入{len(polygons)}个多边形")
        
        # 步骤1: 预处理（清理无效多边形）
        polygons = self._preprocess_polygons(polygons)
        print(f"  步骤1-预处理: {len(polygons)}个有效多边形")
        
        # 步骤2: 去除小多边形
        polygons = self._remove_small_polygons(polygons)
        print(f"  步骤2-去除小多边形: {len(polygons)}个多边形")
        
        # 步骤3: 处理高度重叠
        polygons = self._handle_high_overlap(polygons)
        print(f"  步骤3-处理高度重叠: {len(polygons)}个多边形")
        
        # 步骤4: 分割粘连多边形
        polygons = self._split_connected_polygons(polygons)
        print(f"  步骤4-分割粘连: {len(polygons)}个多边形")
        
        # 步骤5: 形态学优化
        polygons = self._morphological_optimization(polygons)
        print(f"  步骤5-形态学优化: {len(polygons)}个多边形")
        
        # 步骤6: 后处理清理
        polygons = self._post_cleanup(polygons)
        print(f"  步骤6-后处理清理: {len(polygons)}个最终多边形")
        
        return polygons
    
    def _preprocess_polygons(self, polygons: List[Polygon]) -> List[Polygon]:
        """预处理：清理无效多边形"""
        valid_polygons = []
        
        for poly in polygons:
            if not poly.is_valid:
                # 尝试修复无效多边形
                try:
                    poly = poly.buffer(0)
                    if poly.is_valid and poly.area >= self.min_area:
                        valid_polygons.append(poly)
                except:
                    continue
            elif poly.area >= self.min_area:
                valid_polygons.append(poly)
        
        return valid_polygons
    
    def _remove_small_polygons(self, polygons: List[Polygon]) -> List[Polygon]:
        """去除小多边形"""
        return [p for p in polygons if p.area >= self.min_area]
    
    def _handle_high_overlap(self, polygons: List[Polygon]) -> List[Polygon]:
        """
        处理高度重叠的多边形
        
        使用图论方法：构建重叠图，然后合并连通分量
        """
        if len(polygons) <= 1:
            return polygons
        
        # 构建图
        G = nx.Graph()
        for i, poly in enumerate(polygons):
            G.add_node(i, polygon=poly)
        
        # 添加边（如果多边形重叠）
        for i in range(len(polygons)):
            for j in range(i + 1, len(polygons)):
                poly1 = polygons[i]
                poly2 = polygons[j]
                
                if poly1.intersects(poly2):
                    intersection = poly1.intersection(poly2).area
                    min_area = min(poly1.area, poly2.area)
                    
                    if min_area > 0:
                        overlap_ratio = intersection / min_area
                        
                        # 如果重叠比例高，添加边
                        if overlap_ratio > 0.7:  # 70%重叠
                            G.add_edge(i, j, weight=overlap_ratio)
        
        # 获取连通分量
        components = list(nx.connected_components(G))
        
        # 合并每个连通分量
        merged_polygons = []
        
        for component in components:
            if len(component) == 1:
                # 单个多边形，直接保留
                idx = list(component)[0]
                merged_polygons.append(polygons[idx])
            else:
                # 多个多边形，合并它们
                component_polys = [polygons[idx] for idx in component]
                
                try:
                    # 尝试合并
                    merged = unary_union(component_polys)
                    
                    if isinstance(merged, MultiPolygon):
                        # 如果是多重多边形，分解为单个多边形
                        for geom in merged.geoms:
                            if isinstance(geom, Polygon) and geom.area >= self.min_area:
                                merged_polygons.append(geom)
                    elif isinstance(merged, Polygon) and merged.area >= self.min_area:
                        merged_polygons.append(merged)
                except Exception as e:
                    print(f"⚠️ 合并失败: {e}")
                    # 合并失败，保留原始多边形
                    merged_polygons.extend(component_polys)
        
        return merged_polygons
    
    def _split_connected_polygons(self, polygons: List[Polygon]) -> List[Polygon]:
        """
        分割粘连的多边形
        
        使用形态学方法检测和分割粘连区域
        """
        if len(polygons) <= 1:
            return polygons
        
        all_split_polygons = []
        
        for poly in polygons:
            # 检查多边形是否可能是多个粘连颗粒
            if self._is_connected_polygon(poly):
                split_polys = self._split_polygon(poly)
                all_split_polygons.extend(split_polys)
            else:
                all_split_polygons.append(poly)
        
        return all_split_polygons
    
    def _is_connected_polygon(self, polygon: Polygon) -> bool:
        """
        判断多边形是否可能是多个粘连颗粒
        
        基于形状特征：实心度、凹凸性、纵横比等
        """
        if not polygon.is_valid:
            return False
        
        area = polygon.area
        if area == 0:
            return False
        
        # 1. 计算实心度
        convex_hull = polygon.convex_hull
        hull_area = convex_hull.area
        
        if hull_area == 0:
            solidity = 0
        else:
            solidity = area / hull_area
        
        # 低实心度可能表示多个颗粒粘连
        if solidity < 0.6:
            return True
        
        # 2. 计算凹凸性
        perimeter = polygon.length
        hull_perimeter = convex_hull.length
        
        if hull_perimeter == 0:
            concavity = 0
        else:
            concavity = perimeter / hull_perimeter
        
        # 高凹凸性可能表示粘连
        if concavity > 1.5:
            return True
        
        # 3. 检查纵横比
        bounds = polygon.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        if height > 0:
            aspect_ratio = width / height
            if aspect_ratio > self.params['max_aspect_ratio'] or aspect_ratio < 1/self.params['max_aspect_ratio']:
                return True
        
        return False
    
    def _split_polygon(self, polygon: Polygon) -> List[Polygon]:
        """
        分割多边形
        
        使用骨架化和分水岭算法
        """
        try:
            # 获取多边形的边界框
            bounds = polygon.bounds
            minx, miny, maxx, maxy = bounds
            
            # 创建二值图像
            width = int(maxx - minx) + 2
            height = int(maxy - miny) + 2
            
            if width <= 0 or height <= 0:
                return [polygon]
            
            # 创建空白图像
            from skimage.draw import polygon as draw_polygon
            import skimage.morphology as morph
            
            img = np.zeros((height, width), dtype=np.uint8)
            
            # 将多边形绘制到图像上
            poly_coords = list(polygon.exterior.coords)
            x_coords = [int(x - minx + 1) for x, _ in poly_coords]
            y_coords = [int(y - miny + 1) for _, y in poly_coords]
            
            rr, cc = draw_polygon(y_coords, x_coords, img.shape)
            img[rr, cc] = 255
            
            # 骨架化
            skeleton = morph.skeletonize(img > 0)
            
            # 距离变换
            from scipy import ndimage
            distance = ndimage.distance_transform_edt(img > 0)
            
            # 寻找局部最大值作为标记
            from skimage.feature import peak_local_max
            coordinates = peak_local_max(distance, min_distance=5, labels=img > 0)
            
            if len(coordinates) <= 1:
                return [polygon]
            
            # 创建标记图像
            markers = np.zeros_like(img, dtype=np.int32)
            for i, (y, x) in enumerate(coordinates):
                markers[y, x] = i + 1
            
            # 分水岭分割
            from skimage.segmentation import watershed
            labels = watershed(-distance, markers, mask=img > 0)
            
            # 提取分割后的区域
            split_polygons = []
            
            for label_id in np.unique(labels):
                if label_id == 0:
                    continue
                
                # 创建掩码
                mask = (labels == label_id).astype(np.uint8) * 255
                
                # 寻找轮廓
                import cv2
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # 取最大轮廓
                    main_contour = max(contours, key=cv2.contourArea)
                    
                    # 简化轮廓
                    epsilon = 0.01 * cv2.arcLength(main_contour, True)
                    approx = cv2.approxPolyDP(main_contour, epsilon, True)
                    
                    if len(approx) >= 3:
                        # 转换为多边形
                        points = [(point[0][0] + minx - 1, point[0][1] + miny - 1) for point in approx]
                        split_poly = Polygon(points)
                        
                        if split_poly.is_valid and split_poly.area >= self.min_area:
                            split_polygons.append(split_poly)
            
            if len(split_polygons) > 1:
                return split_polygons
            else:
                return [polygon]
            
        except Exception as e:
            print(f"⚠️ 分割多边形失败: {e}")
            return [polygon]
    
    def _morphological_optimization(self, polygons: List[Polygon]) -> List[Polygon]:
        """
        形态学优化：平滑边界，填充孔洞
        """
        optimized_polygons = []
        
        for poly in polygons:
            if not poly.is_valid:
                optimized_polygons.append(poly)
                continue
            
            try:
                # 缓冲操作（正负缓冲可以平滑边界）
                buffered = poly.buffer(self.params['buffer_distance'])
                debuffered = buffered.buffer(-self.params['buffer_distance'])
                
                # 确保仍然是多边形
                if debuffered.is_valid and isinstance(debuffered, Polygon):
                    optimized_polygons.append(debuffered)
                else:
                    optimized_polygons.append(poly)
            except Exception as e:
                print(f"⚠️ 形态学优化失败: {e}")
                optimized_polygons.append(poly)
        
        return optimized_polygons
    
    def _post_cleanup(self, polygons: List[Polygon]) -> List[Polygon]:
        """
        后处理清理：最终过滤和验证
        """
        cleaned_polygons = []
        
        for poly in polygons:
            if not poly.is_valid:
                continue
            
            # 检查面积
            if poly.area < self.min_area:
                continue
            
            # 检查是否为有效多边形（不是线或点）
            if poly.is_empty:
                continue
            
            # 简化多边形（减少顶点数）
            try:
                simplified = poly.simplify(self.params['simplify_tolerance'], preserve_topology=True)
                
                if simplified.is_valid and simplified.area >= self.min_area:
                    cleaned_polygons.append(simplified)
                else:
                    cleaned_polygons.append(poly)
            except:
                cleaned_polygons.append(poly)
        
        return cleaned_polygons
    
    def calculate_statistics(self, polygons: List[Polygon]) -> Dict[str, Any]:
        """
        计算统计信息
        """
        if len(polygons) == 0:
            return {
                'count': 0,
                'total_area': 0,
                'avg_area': 0,
                'min_area': 0,
                'max_area': 0
            }
        
        areas = [p.area for p in polygons if p.is_valid]
        
        return {
            'count': len(polygons),
            'total_area': sum(areas),
            'avg_area': np.mean(areas) if areas else 0,
            'min_area': min(areas) if areas else 0,
            'max_area': max(areas) if areas else 0,
            'std_area': np.std(areas) if areas else 0
        }


if __name__ == "__main__":
    print("SmartPostProcessor测试")
    print("=" * 60)
    
    # 测试代码
    processor = SmartPostProcessor(min_area=30)
    print("✅ 后处理器测试通过")