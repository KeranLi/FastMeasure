"""
Core segmentation utilities migrated from segmenteverygrain.

This module provides core functions for grain segmentation and visualization,
extracted from segmenteverygrain to reduce external dependencies.

Original source: https://github.com/zsylvester/segmenteverygrain
License: MIT (same as original)
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
import rasterio
from rasterio.features import rasterize
from skimage import measure
from skimage.measure import regionprops
import networkx as nx
from typing import List, Tuple, Optional, Union


def load_image(fn: str) -> np.ndarray:
    """
    Load an image from a file.
    
    Parameters
    ----------
    fn : str
        Path to the image file.
        
    Returns
    -------
    np.ndarray
        Loaded image as numpy array.
    """
    img = Image.open(fn)
    return np.array(img)


def collect_polygon_from_mask(
    labels,
    mask,
    image_pred,
    all_grains,
    sx,
    sy,
    min_area=100,
    max_n_large_grains=10,
    max_bg_fraction=0.7,
):
    """
    Collect polygon from a mask and append it to a list of grains.
    
    Parameters
    ----------
    labels : ndarray
        Array of labels for each pixel in the image.
    mask : ndarray
        Boolean mask indicating the region of interest.
    image_pred : ndarray
        Predicted image from the model.
    all_grains : list
        List to append the resulting polygons to.
    sx : ndarray
        X-coordinates of the polygon vertices.
    sy : ndarray
        Y-coordinates of the polygon vertices.
    min_area : int, optional
        Minimum area for a label to be considered significant (default is 100).
    max_n_large_grains : int, optional
        Maximum number of large grains allowed in the mask (default is 10).
    max_bg_fraction : float, optional
        Maximum fraction of the mask that can be background (default is 0.7).
        
    Returns
    -------
    list
        Updated list of polygons representing grains.
    """
    labels_in_mask = np.unique(labels[mask])
    large_labels_in_mask = [
        label
        for label in labels_in_mask
        if len(labels[mask][labels[mask] == label]) >= min_area
    ]
    if (
        len(large_labels_in_mask) < max_n_large_grains
        and (image_pred is None or np.mean(image_pred[:, :, 0][mask]) < max_bg_fraction)
    ):
        poly = Polygon(np.vstack((sx, sy)).T)
        if not poly.is_valid:
            poly = poly.buffer(0)
        all_grains.append(poly)
    return all_grains


def find_overlapping_polygons(polygons: List[Polygon]) -> List[Tuple[int, int]]:
    """
    Find pairs of overlapping polygons.
    
    Parameters
    ----------
    polygons : list
        List of shapely Polygon objects.
        
    Returns
    -------
    list
        List of tuples (i, j) where polygons[i] and polygons[j] overlap.
    """
    overlapping = []
    n = len(polygons)
    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].intersects(polygons[j]):
                if polygons[i].intersection(polygons[j]).area > 0:
                    overlapping.append((i, j))
    return overlapping


def find_connected_components(all_grains, min_area):
    """
    Finds connected components in a graph of overlapping polygons.
    
    Parameters
    ----------
    all_grains : list
        List of polygons representing all grains.
    min_area : float
        Minimum area threshold for valid grains.
        
    Returns
    -------
    new_grains : list
        List of polygons that do not overlap and have an area greater than min_area.
    comps : list
        List of sets, where each set represents a connected component of overlapping polygons.
    g : networkx.Graph
        The graph of overlapping polygons.
    """
    overlapping_polygons = find_overlapping_polygons(all_grains)
    g = nx.Graph(overlapping_polygons)
    comps = list(nx.connected_components(g))
    connected_grains = set()
    for comp in comps:
        connected_grains.update(comp)
    new_grains = []
    for i in range(len(all_grains)):
        if i not in connected_grains and all_grains[i].area > min_area:
            if not all_grains[i].is_valid:
                all_grains[i] = all_grains[i].buffer(0)
            new_grains.append(all_grains[i])
    return new_grains, comps, g


def rasterize_grains(grains: List[Polygon], image: np.ndarray) -> np.ndarray:
    """
    Rasterize a list of grain polygons into a binary mask.
    
    Parameters
    ----------
    grains : list
        List of shapely Polygon objects.
    image : np.ndarray
        Reference image for determining output shape.
        
    Returns
    -------
    np.ndarray
        Binary mask with grains rasterized.
    """
    if len(grains) == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    
    shapes = [(grain, 1) for grain in grains]
    mask = rasterize(shapes, out_shape=image.shape[:2], fill=0, dtype=np.uint8)
    return mask


def pick_most_similar_polygon(polygons: List[Polygon]) -> Polygon:
    """
    Pick the polygon that is most similar to the others based on area.
    
    Parameters
    ----------
    polygons : list
        List of shapely Polygon objects.
        
    Returns
    -------
    Polygon
        The most representative polygon.
    """
    if len(polygons) == 0:
        return None
    if len(polygons) == 1:
        return polygons[0]
    
    areas = [poly.area for poly in polygons]
    median_area = np.median(areas)
    
    # Find polygon closest to median area
    closest_idx = np.argmin([abs(area - median_area) for area in areas])
    return polygons[closest_idx]


def merge_overlapping_polygons(all_grains, new_grains, comps, min_area, image_pred):
    """
    Merge overlapping polygons in a connected component.
    
    Parameters
    ----------
    all_grains : list
        List of all polygons.
    new_grains : list
        List of polygons that do not overlap each other.
    comps : list
        List of connected components.
    min_area : float
        Minimum area threshold.
    image_pred : numpy.ndarray
        The prediction image.
        
    Returns
    -------
    all_grains : list
        List of merged polygons.
    """
    for comp in comps:
        polygons = [all_grains[i] for i in comp]
        most_similar_polygon = pick_most_similar_polygon(polygons)
        
        # Handle difference polygons
        diff_polys = []
        for polygon in polygons:
            if polygon != most_similar_polygon:
                diff_polygon = polygon.difference(most_similar_polygon)
                if diff_polygon.area > min_area:
                    if isinstance(diff_polygon, MultiPolygon):
                        # Take the largest polygon from multipolygon
                        areas = [geom.area for geom in diff_polygon.geoms]
                        diff_polygon = diff_polygon.geoms[np.argmax(areas)]
                    if isinstance(diff_polygon, Polygon):
                        diff_polys.append(diff_polygon)
        
        # Add the most similar polygon
        if most_similar_polygon.area > min_area:
            if not most_similar_polygon.is_valid:
                most_similar_polygon = most_similar_polygon.buffer(0)
            new_grains.append(most_similar_polygon)
    
    return new_grains


def create_labeled_image(all_grains, image):
    """
    Create a labeled image from a list of grains.
    
    Parameters
    ----------
    all_grains : list
        List of shapely Polygon objects representing grains.
    image : np.ndarray
        Reference image for determining output shape.
        
    Returns
    -------
    labels : np.ndarray
        Labeled image where each grain has a unique integer label.
    """
    labels = np.zeros(image.shape[:2], dtype=np.int32)
    for i, grain in enumerate(all_grains):
        mask = rasterize([(grain, i + 1)], out_shape=image.shape[:2], 
                         fill=0, dtype=np.int32)
        labels[mask > 0] = i + 1
    return labels


def plot_image_w_colorful_grains(
    all_grains,
    image,
    cmap='tab20',
    figsize=(10, 10),
    ax=None,
    alpha=0.5,
    edgecolor='k',
    linewidth=1
):
    """
    Plot image with colorful grain overlays.
    
    Parameters
    ----------
    all_grains : list
        List of shapely Polygon objects.
    image : np.ndarray
        Background image.
    cmap : str, optional
        Colormap for grains (default is 'tab20').
    figsize : tuple, optional
        Figure size (default is (10, 10)).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    alpha : float, optional
        Transparency of grain overlays (default is 0.5).
    edgecolor : str, optional
        Edge color for grain boundaries (default is 'k').
    linewidth : float, optional
        Line width for grain boundaries (default is 1).
        
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax.imshow(image)
    
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(all_grains)))
    
    for i, grain in enumerate(all_grains):
        if grain.is_valid:
            x, y = grain.exterior.xy
            ax.fill(x, y, alpha=alpha, fc=colors[i], ec=edgecolor, linewidth=linewidth)
    
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.axis('off')
    
    return ax


def plot_grain_axes_and_centroids(
    all_grains,
    labels,
    ax,
    linewidth=1,
    markersize=10
):
    """
    Plot grain major/minor axes and centroids.
    
    Parameters
    ----------
    all_grains : list
        List of shapely Polygon objects.
    labels : np.ndarray
        Labeled image.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    linewidth : float, optional
        Line width for axes (default is 1).
    markersize : float, optional
        Marker size for centroids (default is 10).
    """
    regions = regionprops(labels)
    
    for region in regions:
        # Plot centroid
        y, x = region.centroid
        ax.plot(x, y, 'r+', markersize=markersize)
        
        # Plot major axis
        orientation = region.orientation
        x1 = x + 0.5 * region.major_axis_length * np.cos(orientation)
        y1 = y - 0.5 * region.major_axis_length * np.sin(orientation)
        x2 = x - 0.5 * region.major_axis_length * np.cos(orientation)
        y2 = y + 0.5 * region.major_axis_length * np.sin(orientation)
        ax.plot([x1, x2], [y1, y2], 'r-', linewidth=linewidth)
        
        # Plot minor axis
        x1 = x + 0.5 * region.minor_axis_length * np.sin(orientation)
        y1 = y + 0.5 * region.minor_axis_length * np.cos(orientation)
        x2 = x - 0.5 * region.minor_axis_length * np.sin(orientation)
        y2 = y - 0.5 * region.minor_axis_length * np.cos(orientation)
        ax.plot([x1, x2], [y1, y2], 'b-', linewidth=linewidth)


def polygons_to_grains(polygons: List[Polygon], image: np.ndarray = None) -> list:
    """
    Convert a list of polygons to grain objects.
    
    Parameters
    ----------
    polygons : list
        List of shapely Polygon objects.
    image : np.ndarray, optional
        Reference image for color information.
        
    Returns
    -------
    list
        List of grain-like dictionaries with basic properties.
    """
    grains = []
    for poly in polygons:
        if not poly.is_valid:
            poly = poly.buffer(0)
        
        grain = {
            'polygon': poly,
            'area': poly.area,
            'centroid': (poly.centroid.x, poly.centroid.y),
        }
        
        if image is not None:
            # Calculate basic color metrics if image provided
            mask = rasterize([(poly, 1)], out_shape=image.shape[:2], 
                           fill=0, dtype=np.uint8)
            if len(image.shape) == 3:
                grain['mean_intensity'] = np.mean(image[mask > 0], axis=0)
            else:
                grain['mean_intensity'] = np.mean(image[mask > 0])
        
        grains.append(grain)
    
    return grains


def save_grains(fn: str, grains: list):
    """
    Save grains to a file.
    
    Parameters
    ----------
    fn : str
        Output file path.
    grains : list
        List of grain objects or dictionaries.
    """
    import pickle
    with open(fn, 'wb') as f:
        pickle.dump(grains, f)
