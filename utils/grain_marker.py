"""
Rock Grain Labeling Module
File: grain_marker.py
Function: Add grain numbers and area labels to rock segmentation result images
Features: Support label without background, auto adjust position to avoid overlap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import matplotlib.patheffects as path_effects


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
    Add grain numbers and area labels to rock segmentation image
    
    Parameters:
        ax: matplotlib axes object
        grain_data: Grain data, must contain ['label', 'centroid-0', 'centroid-1', 'area'] columns
        image_shape: Image size (height, width, channels)
        scale_factor: Scale factor (um/pixel), if provided show real area
        font_size: Font size
        text_color: Text color
        bg_color: Background box color, None or empty string means no background
        show_area: Whether to show area
        max_labels: Maximum labels in dense areas
        min_area: Minimum label area (pixels)
        text_outline: Whether to add text outline
        outline_color: Outline color
        outline_width: Outline width
        
    Returns:
        Updated axes object
    """
    
    # 1. Check required data columns
    required_columns = ['label', 'centroid-0', 'centroid-1', 'area']
    for col in required_columns:
        if col not in grain_data.columns:
            print(f"Warning: Grain data missing column '{col}', skipping labeling")
            return ax
    
    # 2. Ensure data is numeric
    for col in required_columns:
        grain_data[col] = pd.to_numeric(grain_data[col], errors='coerce')
    
    # 3. Filter out invalid data and small grains
    valid_data = grain_data.dropna(subset=required_columns)
    valid_data = valid_data[valid_data['area'] >= min_area]
    
    if len(valid_data) == 0:
        return ax
    
    # 4. Sort by area from large to small, prioritize labeling large grains
    sorted_data = valid_data.sort_values('area', ascending=False)
    
    # 5. Limit maximum labels
    if len(sorted_data) > max_labels:
        sorted_data = _filter_dense_areas(sorted_data, image_shape, max_labels)
    
    # 6. Record labeled positions to avoid overlap
    used_positions = []
    
    # 7. Add label for each grain
    for _, row in sorted_data.iterrows():
        label_num = int(row['label'])
        centroid_y = row['centroid-0']  # Row coordinate (y)
        centroid_x = row['centroid-1']  # Column coordinate (x)
        
        # Build label text
        text = _create_label_text(label_num, row['area'], scale_factor, show_area)
        
        # Auto adjust position to avoid overlap
        final_x, final_y = _find_available_position(
            centroid_x, centroid_y, used_positions, image_shape
        )
        
        # If suitable position found, add label
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
    Filter grains in dense areas, prioritize keeping large area and sparse area grains
    """
    # Calculate density around each grain
    densities = _calculate_grain_densities(grain_data, image_shape)
    grain_data['density'] = densities
    
    # Calculate composite score: large area + low density
    grain_data['score'] = (
        grain_data['area'] / grain_data['area'].max() * 0.7 +  # Area weight 70%
        (1 - grain_data['density']) * 0.3  # Sparsity weight 30%
    )
    
    # Sort by score, take top max_labels
    return grain_data.sort_values('score', ascending=False).head(max_labels)


def _calculate_grain_densities(
    grain_data: pd.DataFrame,
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Calculate density around each grain
    """
    if len(grain_data) == 0:
        return np.array([])
    
    # Get centroid coordinates (x, y)
    centroids = grain_data[['centroid-1', 'centroid-0']].values
    
    # Set density calculation radius (5% of image size)
    density_radius = min(image_shape[0], image_shape[1]) * 0.05
    
    densities = np.zeros(len(centroids))
    
    for i, (x, y) in enumerate(centroids):
        # Calculate distance to all other grains
        distances = np.sqrt(
            (centroids[:, 0] - x) ** 2 + 
            (centroids[:, 1] - y) ** 2
        )
        
        # Count grains within radius (excluding self)
        close_grains = np.sum(distances < density_radius) - 1
        densities[i] = close_grains
    
    # Normalize to 0-1 range
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
    Create label text
    """
    if not show_area:
        return f"{label_num}"
    
    if scale_factor:
        # Calculate real area (um^2)
        real_area = area * (scale_factor ** 2)
        if real_area > 1000:
            # Greater than 1000um^2 show as mm^2
            return f"{label_num}\n{real_area/1000:.1f}mm2"
        else:
            return f"{label_num}\n{real_area:.0f}um2"
    else:
        # Show pixel area
        return f"{label_num}\n{area:.0f}px"


def _find_available_position(
    x: float,
    y: float,
    used_positions: List[Tuple[float, float]],
    image_shape: Tuple[int, int],
    min_distance: float = 25.0  # Reduce min distance, allow denser labels
) -> Tuple[Optional[float], Optional[float]]:
    """
    Find available label position to avoid overlap
    """
    # Try position offsets
    position_offsets = [
        (0, 0),           # Original position
        (20, 0), (-20, 0), (0, 20), (0, -20),  # Up down left right
        (15, 15), (15, -15), (-15, 15), (-15, -15),  # Diagonal
        (30, 0), (-30, 0), (0, 30), (0, -30),  # Further up down left right
        (10, 25), (10, -25), (-10, 25), (-10, -25),  # Diagonal
    ]
    
    for dx, dy in position_offsets:
        new_x = x + dx
        new_y = y + dy
        
        # Check if within image range
        if (0 <= new_x < image_shape[1] and 
            0 <= new_y < image_shape[0] and
            _is_position_available(new_x, new_y, used_positions, min_distance)):
            return new_x, new_y
    
    # No suitable position found
    return None, None


def _is_position_available(
    x: float,
    y: float,
    used_positions: List[Tuple[float, float]],
    min_distance: float
) -> bool:
    """
    Check if position is available (far enough from existing labels)
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
    Add single grain label at specified position
    
    Parameters:
        ax: matplotlib axes
        x: x coordinate
        y: y coordinate
        text: Label text
        font_size: Font size
        text_color: Text color
        bg_color: Background color, None means no background
        text_outline: Whether to add text outline
        outline_color: Outline color
        outline_width: Outline width
    """
    if bg_color:
        # Label with background box
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
        # No background box, add text outline for readability
        text_obj = ax.text(
            x, y,
            text,
            fontsize=font_size,
            color=text_color,
            ha='center',
            va='center',
            zorder=10,
            weight='bold'  # Bold font
        )
        
        # Add black outline
        if text_outline:
            text_obj.set_path_effects([
                path_effects.withStroke(
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
    Add grain labels using config dictionary (convenience function)
    """
    # Extract config parameters
    font_size = config.get('font_size', 11)
    text_color = config.get('text_color', 'yellow')
    bg_color = config.get('bg_color', '')
    show_area = config.get('show_area', True)
    max_labels = config.get('max_labels', 1000)
    min_area = config.get('min_area', 0)
    text_outline = config.get('text_outline', True)
    outline_color = config.get('outline_color', 'black')
    outline_width = config.get('outline_width', 2.0)
    
    # Handle empty string background
    if bg_color == '':
        bg_color = None
    
    # Call main function
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


# Test code
if __name__ == "__main__":
    print("grain_marker.py module test")
    print("Function: Add numbers and area labels for rock grains")
    print("Usage:")
    print("1. Import module: from grain_marker import add_grain_labels")
    print("2. Call function: add_grain_labels(ax, grain_data, image_shape, ...)")
    print("\nConfig description:")
    print("  - bg_color: 'white' (with background), '' or None (no background)")
    print("  - text_color: 'yellow', 'white', 'black', etc.")
    print("  - text_outline: True (recommended when no background)")
    print("\nModule ready!")
