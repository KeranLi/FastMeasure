"""
Smart Post-processing Module - Optimized for Rock Grain Segmentation
File: core/seg_optimize.py
Function: Smart post-processing to resolve overlap and adhesion issues
"""

import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import networkx as nx
from typing import List, Dict, Any, Optional
import warnings

warnings.filterwarnings('ignore')


class SmartPostProcessor:
    """Smart Post Processor"""
    
    def __init__(self, min_area: int = 30, iou_threshold: float = 0.5):
        """
        Initialize smart post processor
        
        Args:
            min_area: Minimum area
            iou_threshold: IoU threshold
        """
        self.min_area = min_area
        self.iou_threshold = iou_threshold
        
        # Post-processing parameters
        self.params = {
            'erosion_distance': -2,    # Erosion distance (negative)
            'dilation_distance': 2,    # Dilation distance
            'simplify_tolerance': 1.0, # Simplify tolerance
            'buffer_distance': 0.5,    # Buffer distance
            'min_overlap_area': 10,    # Minimum overlap area
            'max_aspect_ratio': 5.0,   # Maximum aspect ratio
        }
        
        print("SmartPostProcessor initialized")
    
    def process(self, polygons: List[Polygon]) -> List[Polygon]:
        """
        Main processing function
        
        Args:
            polygons: Input polygon list
            
        Returns:
            Processed polygon list
        """
        if len(polygons) == 0:
            return []
        
        print(f"Smart post-processing: {len(polygons)} polygons input")
        
        # Step 1: Preprocessing (clean invalid polygons)
        polygons = self._preprocess_polygons(polygons)
        print(f"  Step 1-Preprocessing: {len(polygons)} valid polygons")
        
        # Step 2: Remove small polygons
        polygons = self._remove_small_polygons(polygons)
        print(f"  Step 2-Remove small: {len(polygons)} polygons")
        
        # Step 3: Handle high overlap
        polygons = self._handle_high_overlap(polygons)
        print(f"  Step 3-Handle overlap: {len(polygons)} polygons")
        
        # Step 4: Split connected polygons
        polygons = self._split_connected_polygons(polygons)
        print(f"  Step 4-Split connected: {len(polygons)} polygons")
        
        # Step 5: Morphological optimization
        polygons = self._morphological_optimization(polygons)
        print(f"  Step 5-Morphological: {len(polygons)} polygons")
        
        # Step 6: Post cleanup
        polygons = self._post_cleanup(polygons)
        print(f"  Step 6-Cleanup: {len(polygons)} final polygons")
        
        return polygons
    
    def _preprocess_polygons(self, polygons: List[Polygon]) -> List[Polygon]:
        """Preprocessing: Clean invalid polygons"""
        valid_polygons = []
        
        for poly in polygons:
            if not poly.is_valid:
                # Try to fix invalid polygon
                try:
                    poly = poly.buffer(0)
                    if poly.is_valid and poly.area >= self.min_area:
                        valid_polygons.append(poly)
                except Exception:
                    continue
            elif poly.area >= self.min_area:
                valid_polygons.append(poly)
        
        return valid_polygons
    
    def _remove_small_polygons(self, polygons: List[Polygon]) -> List[Polygon]:
        """Remove small polygons"""
        return [p for p in polygons if p.area >= self.min_area]
    
    def _handle_high_overlap(self, polygons: List[Polygon]) -> List[Polygon]:
        """
        Handle highly overlapping polygons
        
        Use graph theory: Build overlap graph, then merge connected components
        """
        if len(polygons) <= 1:
            return polygons
        
        # Build graph
        G = nx.Graph()
        for i, poly in enumerate(polygons):
            G.add_node(i, polygon=poly)
        
        # Add edges (if polygons overlap)
        for i in range(len(polygons)):
            for j in range(i + 1, len(polygons)):
                poly1 = polygons[i]
                poly2 = polygons[j]
                
                if poly1.intersects(poly2):
                    intersection = poly1.intersection(poly2).area
                    min_area = min(poly1.area, poly2.area)
                    
                    if min_area > 0:
                        overlap_ratio = intersection / min_area
                        
                        # If overlap ratio is high, add edge
                        if overlap_ratio > 0.7:  # 70% overlap
                            G.add_edge(i, j, weight=overlap_ratio)
        
        # Get connected components
        components = list(nx.connected_components(G))
        
        # Merge each connected component
        merged_polygons = []
        
        for component in components:
            if len(component) == 1:
                # Single polygon, keep directly
                idx = list(component)[0]
                merged_polygons.append(polygons[idx])
            else:
                # Multiple polygons, merge them
                component_polys = [polygons[idx] for idx in component]
                
                try:
                    # Try to merge
                    merged = unary_union(component_polys)
                    
                    if isinstance(merged, MultiPolygon):
                        # If MultiPolygon, decompose to single polygons
                        for geom in merged.geoms:
                            if isinstance(geom, Polygon) and geom.area >= self.min_area:
                                merged_polygons.append(geom)
                    elif isinstance(merged, Polygon) and merged.area >= self.min_area:
                        merged_polygons.append(merged)
                except Exception as e:
                    print(f"Merge failed: {e}")
                    # Merge failed, keep original polygons
                    merged_polygons.extend(component_polys)
        
        return merged_polygons
    
    def _split_connected_polygons(self, polygons: List[Polygon]) -> List[Polygon]:
        """
        Split connected polygons
        
        Use morphological methods to detect and split adhesion regions
        """
        if len(polygons) <= 1:
            return polygons
        
        all_split_polygons = []
        
        for poly in polygons:
            # Check if polygon might be multiple adhered grains
            if self._is_connected_polygon(poly):
                split_polys = self._split_polygon(poly)
                all_split_polygons.extend(split_polys)
            else:
                all_split_polygons.append(poly)
        
        return all_split_polygons
    
    def _is_connected_polygon(self, polygon: Polygon) -> bool:
        """
        Determine if polygon might be multiple adhered grains
        
        Based on shape features: solidity, convexity, aspect ratio, etc.
        """
        if not polygon.is_valid:
            return False
        
        area = polygon.area
        if area == 0:
            return False
        
        # 1. Calculate solidity
        convex_hull = polygon.convex_hull
        hull_area = convex_hull.area
        
        if hull_area == 0:
            solidity = 0
        else:
            solidity = area / hull_area
        
        # Low solidity may indicate multiple adhered grains
        if solidity < 0.6:
            return True
        
        # 2. Calculate convexity
        perimeter = polygon.length
        hull_perimeter = convex_hull.length
        
        if hull_perimeter == 0:
            concavity = 0
        else:
            concavity = perimeter / hull_perimeter
        
        # High concavity may indicate adhesion
        if concavity > 1.5:
            return True
        
        # 3. Check aspect ratio
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
        Split polygon
        
        Use skeletonization and watershed algorithm
        """
        try:
            # Get polygon bounding box
            bounds = polygon.bounds
            minx, miny, maxx, maxy = bounds
            
            # Create binary image
            width = int(maxx - minx) + 2
            height = int(maxy - miny) + 2
            
            if width <= 0 or height <= 0:
                return [polygon]
            
            # Create blank image
            from skimage.draw import polygon as draw_polygon
            import skimage.morphology as morph
            
            img = np.zeros((height, width), dtype=np.uint8)
            
            # Draw polygon to image
            poly_coords = list(polygon.exterior.coords)
            x_coords = [int(x - minx + 1) for x, _ in poly_coords]
            y_coords = [int(y - miny + 1) for _, y in poly_coords]
            
            rr, cc = draw_polygon(y_coords, x_coords, img.shape)
            img[rr, cc] = 255
            
            # Skeletonization
            skeleton = morph.skeletonize(img > 0)
            
            # Distance transform
            from scipy import ndimage
            distance = ndimage.distance_transform_edt(img > 0)
            
            # Find local maxima as markers
            from skimage.feature import peak_local_max
            coordinates = peak_local_max(distance, min_distance=5, labels=img > 0)
            
            if len(coordinates) <= 1:
                return [polygon]
            
            # Create marker image
            markers = np.zeros_like(img, dtype=np.int32)
            for i, (y, x) in enumerate(coordinates):
                markers[y, x] = i + 1
            
            # Watershed segmentation
            from skimage.segmentation import watershed
            labels = watershed(-distance, markers, mask=img > 0)
            
            # Extract segmented regions
            split_polygons = []
            
            for label_id in np.unique(labels):
                if label_id == 0:
                    continue
                
                # Create mask
                mask = (labels == label_id).astype(np.uint8) * 255
                
                # Find contours
                import cv2
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Take largest contour
                    main_contour = max(contours, key=cv2.contourArea)
                    
                    # Simplify contour
                    epsilon = 0.01 * cv2.arcLength(main_contour, True)
                    approx = cv2.approxPolyDP(main_contour, epsilon, True)
                    
                    if len(approx) >= 3:
                        # Convert to polygon
                        points = [(point[0][0] + minx - 1, point[0][1] + miny - 1) for point in approx]
                        split_poly = Polygon(points)
                        
                        if split_poly.is_valid and split_poly.area >= self.min_area:
                            split_polygons.append(split_poly)
            
            if len(split_polygons) > 1:
                return split_polygons
            else:
                return [polygon]
            
        except Exception as e:
            print(f"Split polygon failed: {e}")
            return [polygon]
    
    def _morphological_optimization(self, polygons: List[Polygon]) -> List[Polygon]:
        """
        Morphological optimization: Smooth boundaries, fill holes
        """
        optimized_polygons = []
        
        for poly in polygons:
            if not poly.is_valid:
                optimized_polygons.append(poly)
                continue
            
            try:
                # Buffer operation (positive and negative buffer can smooth boundaries)
                buffered = poly.buffer(self.params['buffer_distance'])
                debuffered = buffered.buffer(-self.params['buffer_distance'])
                
                # Ensure still polygon
                if debuffered.is_valid and isinstance(debuffered, Polygon):
                    optimized_polygons.append(debuffered)
                else:
                    optimized_polygons.append(poly)
            except Exception as e:
                print(f"Morphological optimization failed: {e}")
                optimized_polygons.append(poly)
        
        return optimized_polygons
    
    def _post_cleanup(self, polygons: List[Polygon]) -> List[Polygon]:
        """
        Post cleanup: Final filtering and validation
        """
        cleaned_polygons = []
        
        for poly in polygons:
            if not poly.is_valid:
                continue
            
            # Check area
            if poly.area < self.min_area:
                continue
            
            # Check if valid polygon (not line or point)
            if poly.is_empty:
                continue
            
            # Simplify polygon (reduce vertices)
            try:
                simplified = poly.simplify(self.params['simplify_tolerance'], preserve_topology=True)
                
                if simplified.is_valid and simplified.area >= self.min_area:
                    cleaned_polygons.append(simplified)
                else:
                    cleaned_polygons.append(poly)
            except Exception:
                cleaned_polygons.append(poly)
        
        return cleaned_polygons
    
    def calculate_statistics(self, polygons: List[Polygon]) -> Dict[str, Any]:
        """
        Calculate statistics
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
    print("SmartPostProcessor Test")
    print("=" * 60)
    
    # Test code
    processor = SmartPostProcessor(min_area=30)
    print("Post processor test passed")
