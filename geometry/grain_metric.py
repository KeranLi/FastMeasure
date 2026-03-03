import numpy as np
import pandas as pd
from skimage.measure import regionprops
from skimage import measure
from scipy.spatial import ConvexHull

class GrainShapeMetrics:
    """
    Grain contour calculation tool class, computing various grain shape structural parameters
    """
    
    def __init__(self, grain_data: pd.DataFrame):
        """
        Initialization method
        
        Args:
            grain_data (pd.DataFrame): DataFrame containing grain region data, must include 'area', 'perimeter', 'coordinates' columns
        """
        self.grain_data = grain_data


    def calculate_2d_zingg_parameters(self) -> pd.DataFrame:
        """
        In 2D scenario, L=major axis length, S=I=minor axis length
        """
        l = self.grain_data['major_axis_length']
        s = self.grain_data['minor_axis_length']
        
        # 2D adapted version
        ei = s / l  # 2D elongation
        fi = 1.0    # 2D cannot represent flatness, set to constant or ignore
        ar = s / l  # 2D aspect ratio
        
        return pd.DataFrame({
            'EI_2d': ei,
            'FI_2d': fi,
            'AR_2d': ar
        })

    def calculate_fourier_descriptors(self, n_coeffs=25) -> pd.DataFrame:
        """
        Fourier descriptors for 2D contour (corresponding to 3D spherical harmonics)
        Used to describe shape features from global to detail
        """
        results = []
        for _, grain in self.grain_data.iterrows():
            coords = np.array(grain['coordinates'])
            
            # 1. Convert coordinates to complex form x + iy
            complex_coords = coords[:, 0] + 1j * coords[:, 1]
            
            # 2. Discrete Fourier Transform
            coeffs = np.fft.fft(complex_coords)
            
            # 3. Normalization (eliminate translation, rotation, scale effects)
            # coeffs[0] is DC component (center position), ignore
            # Normalize by coeffs[1] for scale invariance
            abs_coeffs = np.abs(coeffs)
            if abs_coeffs[1] != 0:
                normalized_coeffs = abs_coeffs / abs_coeffs[1]
            else:
                normalized_coeffs = np.zeros(len(abs_coeffs))

            # 4. Extract Dn-like features (take first n_coeffs)
            # D2 corresponds to normalized_coeffs[2], and so on
            d_vals = []
            for i in range(2, min(n_coeffs + 2, len(normalized_coeffs))):
                d_vals.append(normalized_coeffs[i])
            
            # If coefficients not enough, pad with zeros
            while len(d_vals) < 3: 
                d_vals.append(0)
                
            results.append(d_vals[:3]) # Only take D2, D3, D4 for demonstration

        return pd.DataFrame(results, columns=['D2_2d', 'D3_2d', 'D4_2d'])

    def calculate_sh_equivalent_fd(self, beta):
        """
        FD = (6 + beta) / 2
        beta is usually obtained from the slope of log(Fourier_Coeff) vs log(n)
        """
        return (6 + beta) / 2

    def calculate_convexity(self) -> pd.Series:
        """
        Calculate grain convexity
        
        Formula: C = A / A_hull
        Where A is grain area, A_hull is the area of convex hull of grain 2D projection contour.
        """
        # If grain_data is obtained via skimage.measure.regionprops,
        # it usually already contains 'solidity' attribute, which equals convexity here.
        if 'solidity' in self.grain_data.columns:
            return self.grain_data['solidity']
        
        # If pre-calculated solidity not available, calculate manually via coordinates
        convexity_list = []
        for _, grain in self.grain_data.iterrows():
            coords = np.array(grain['coordinates'])
            area = grain['area']
            
            # Use scipy.spatial.ConvexHull to calculate convex hull
            try:
                hull = ConvexHull(coords)
                # hull.volume represents area in 2D
                a_hull = hull.volume 
                convexity_list.append(area / a_hull)
            except Exception:
                # Handle cases where coordinates are insufficient to form convex hull
                convexity_list.append(np.nan)
                
        return pd.Series(convexity_list)
        
    def calculate_circularity(self) -> pd.Series:
        """
        Calculate grain circularity
        
        Formula: Circularity = 4 * π * Area / Perimeter^2
        """
        area = self.grain_data['area']
        perimeter = self.grain_data['perimeter']
        return 4 * np.pi * area / (perimeter ** 2)
    
    def calculate_aspect_ratio(self) -> pd.Series:
        """
        Calculate grain aspect ratio
        
        Formula: Aspect Ratio = Major Axis Length / Minor Axis Length
        """
        major_axis_length = self.grain_data['major_axis_length']
        minor_axis_length = self.grain_data['minor_axis_length']
        return major_axis_length / minor_axis_length
    
    def calculate_rectangularity(self) -> pd.Series:
        """
        Calculate grain rectangularity
        
        Formula: Rectangularity = Area / (Major Axis Length * Minor Axis Length)
        """
        area = self.grain_data['area']
        major_axis_length = self.grain_data['major_axis_length']
        minor_axis_length = self.grain_data['minor_axis_length']
        return area / (major_axis_length * minor_axis_length)
    
    def calculate_compactness(self) -> pd.Series:
        """
        Calculate grain compactness
        
        Formula: Compactness = Perimeter^2 / (4 * π * Area)
        """
        perimeter = self.grain_data['perimeter']
        area = self.grain_data['area']
        return perimeter ** 2 / (4 * np.pi * area)
    
    def calculate_fractal_dimension(self) -> pd.Series:
        """
        Calculate grain fractal dimension
        
        Calculate fractal dimension using box counting method
        """
        def box_counting(coords):
            """Calculate fractal dimension using box counting method"""
            coords = np.array(coords)
            if len(coords) < 2:
                return 1.0
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            size = max(max_coords - min_coords)
            if size == 0:
                return 1.0
            box_size = size / 10
            count = 0
            for box_x in np.arange(min_coords[0], max_coords[0], box_size):
                for box_y in np.arange(min_coords[1], max_coords[1], box_size):
                    if np.any((coords[:, 0] >= box_x) & (coords[:, 0] < box_x + box_size) &
                              (coords[:, 1] >= box_y) & (coords[:, 1] < box_y + box_size)):
                        count += 1
            if count <= 1:
                return 1.0
            return np.log(count) / np.log(1 / box_size)
        
        # Calculate fractal dimension for each grain
        results = []
        for _, grain in self.grain_data.iterrows():
            if 'coordinates' in grain and grain['coordinates'] is not None:
                try:
                    fd = box_counting(grain['coordinates'])
                    results.append(fd)
                except Exception:
                    results.append(1.0)
            else:
                results.append(1.0)
        return pd.Series(results, index=self.grain_data.index)
        
    def calculate_angularity(self) -> pd.Series:
        """
        Calculate grain angularity
        
        Estimate by calculating the sharpness of grain contour
        """
        angularity = []
        for _, grain in self.grain_data.iterrows():
            if 'coordinates' in grain and grain['coordinates'] is not None:
                try:
                    coords = np.array(grain['coordinates'])
                    if len(coords) >= 3:
                        hull = ConvexHull(coords)
                        angularity.append(len(hull.vertices))  # Count boundary corner points
                    else:
                        angularity.append(0)
                except Exception:
                    angularity.append(0)
            else:
                angularity.append(0)
        return pd.Series(angularity, index=self.grain_data.index)
    
    def calculate_roundness(self) -> pd.Series:
        """
        Calculate grain roundness
        
        Formula: Roundness = Perimeter^2 / (4 * π * Area)
        """
        perimeter = self.grain_data['perimeter']
        area = self.grain_data['area']
        return perimeter ** 2 / (4 * np.pi * area)
    
    def compute_all_metrics(self):
        """
        Calculate all grain shape parameters
        """
        # Basic shape parameters
        self.grain_data['circularity'] = self.calculate_circularity()
        self.grain_data['aspect_ratio'] = self.calculate_aspect_ratio()
        self.grain_data['rectangularity'] = self.calculate_rectangularity()
        self.grain_data['compactness'] = self.calculate_compactness()
        self.grain_data['roundness'] = self.calculate_roundness()
        self.grain_data['convexity'] = self.calculate_convexity()
        
        # Advanced parameters requiring coordinates
        self.grain_data['fractal_dimension'] = self.calculate_fractal_dimension()
        self.grain_data['angularity'] = self.calculate_angularity()
        
        # Zingg parameters (2D shape classification)
        zingg_df = self.calculate_2d_zingg_parameters()
        self.grain_data['EI_2d'] = zingg_df['EI_2d']
        self.grain_data['FI_2d'] = zingg_df['FI_2d']
        self.grain_data['AR_2d'] = zingg_df['AR_2d']
        
        # Fourier descriptors (requires coordinates)
        if 'coordinates' in self.grain_data.columns:
            fourier_df = self.calculate_fourier_descriptors()
            self.grain_data['D2_2d'] = fourier_df['D2_2d']
            self.grain_data['D3_2d'] = fourier_df['D3_2d']
            self.grain_data['D4_2d'] = fourier_df['D4_2d']
        
        return self.grain_data