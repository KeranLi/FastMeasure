"""
Scale Calibration Module - Manual scale bar calibration for interactive mode
File: core/scale_calibration.py
Function: Provide manual scale calibration by measuring line segments in images
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Optional, Tuple, Callable
import tkinter as tk
from tkinter import simpledialog, messagebox


class InteractiveScaleCalibrator:
    """
    Interactive scale calibrator for measuring line segments and calculating scale factors
    
    Usage:
        calibrator = InteractiveScaleCalibrator()
        scale_factor = calibrator.calibrate_scale(image, ax, fig)
        # Returns: scale factor in um/pixel, or None if calibration failed
    """
    
    def __init__(self):
        self.calibration_points = []
        self.temp_line = None
        self.is_calibrating = False
        self.callback = None
        self.result = None
        
    def calibrate_scale(self, image: np.ndarray, ax: plt.Axes, fig: plt.Figure,
                       callback: Optional[Callable[[float], None]] = None) -> Optional[float]:
        """
        Start scale calibration process
        
        Args:
            image: Current image array
            ax: Matplotlib axes object
            fig: Matplotlib figure object
            callback: Optional callback function to receive scale factor
            
        Returns:
            Scale factor (um/pixel) or None if cancelled
        """
        self.calibration_points = []
        self.temp_line = None
        self.is_calibrating = True
        self.callback = callback
        self.result = None
        
        # Show instructions
        messagebox.showinfo(
            "Scale Calibration",
            "Scale Calibration Mode:\n\n"
            "1. Click on the image to mark the START point of a known-length line\n"
            "2. Click again to mark the END point\n"
            "3. Enter the actual length in microns (um) when prompted\n\n"
            "Tips:\n"
            "- Use a clear, straight line (e.g., scale bar, ruler, or known object)\n"
            "- The longer the line, the more accurate the calibration\n"
            "- Press 'Escape' to cancel calibration"
        )
        
        # Store original title
        self.original_title = ax.get_title()
        ax.set_title(self.original_title + " [SCALE CALIBRATION MODE]", 
                    fontsize=14, color='red', fontweight='bold')
        fig.canvas.draw()
        
        return None  # Result will be set through callback
    
    def on_click(self, event) -> bool:
        """
        Handle mouse click during calibration
        
        Args:
            event: Matplotlib mouse event
            
        Returns:
            True if calibration is complete, False otherwise
        """
        if not self.is_calibrating:
            return False
            
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return False
            
        x, y = event.xdata, event.ydata
        
        if len(self.calibration_points) == 0:
            # First point - start of line
            self.calibration_points.append((x, y))
            print(f"Scale calibration: Start point at ({x:.1f}, {y:.1f})")
            print("Click the end point of the line...")
            
        elif len(self.calibration_points) == 1:
            # Second point - end of line
            self.calibration_points.append((x, y))
            print(f"Scale calibration: End point at ({x:.1f}, {y:.1f})")
            
            # Calculate pixel distance
            x1, y1 = self.calibration_points[0]
            x2, y2 = self.calibration_points[1]
            pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            print(f"Measured pixel distance: {pixel_distance:.2f} pixels")
            
            # Draw the measurement line
            self._draw_measurement_line(event.inaxes)
            
            # Ask for actual length
            self._ask_actual_length(pixel_distance)
            
            return True
            
        return False
    
    def _draw_measurement_line(self, ax):
        """Draw the measurement line on the axes"""
        if len(self.calibration_points) == 2:
            x1, y1 = self.calibration_points[0]
            x2, y2 = self.calibration_points[1]
            
            line = Line2D([x1, x2], [y1, y2], 
                         color='red', 
                         linewidth=2, 
                         linestyle='--',
                         marker='o',
                         markersize=8,
                         markerfacecolor='yellow',
                         markeredgecolor='red',
                         markeredgewidth=2,
                         zorder=100)
            ax.add_line(line)
            
            # Add text label at midpoint
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(mid_x, mid_y - 20, 'Scale Bar', 
                   color='red', fontsize=12, fontweight='bold',
                   ha='center', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='yellow', 
                            alpha=0.7,
                            edgecolor='red'))
            
            ax.figure.canvas.draw()
            self.temp_line = line
    
    def _ask_actual_length(self, pixel_distance: float):
        """Ask user for actual length and calculate scale factor"""
        try:
            # Create dialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            actual_length = simpledialog.askfloat(
                "Enter Actual Length",
                f"Measured pixel distance: {pixel_distance:.2f} pixels\n\n"
                f"Enter the actual length in microns (um):",
                minvalue=0.1,
                initialvalue=1000.0
            )
            
            root.destroy()
            
            if actual_length is None:
                print("Scale calibration cancelled by user")
                self.reset_calibration()
                return
            
            # Calculate scale factor (um/pixel)
            scale_factor = actual_length / pixel_distance
            self.result = scale_factor
            
            print(f"Scale calibration completed!")
            print(f"  Pixel distance: {pixel_distance:.2f} px")
            print(f"  Actual length: {actual_length:.2f} um")
            print(f"  Scale factor: {scale_factor:.4f} um/px")
            print(f"  (1 pixel = {scale_factor:.4f} microns)")
            
            # Show confirmation
            messagebox.showinfo(
                "Calibration Complete",
                f"Scale factor calculated: {scale_factor:.4f} um/px\n\n"
                f"This means:\n"
                f"  1 pixel = {scale_factor:.4f} microns\n"
                f"  1 micron = {1/scale_factor:.4f} pixels"
            )
            
            # Call callback if provided
            if self.callback:
                self.callback(scale_factor)
                
        except Exception as e:
            print(f"Error during scale calibration: {e}")
            self.reset_calibration()
    
    def reset_calibration(self):
        """Reset calibration state"""
        self.calibration_points = []
        self.is_calibrating = False
        if self.temp_line:
            try:
                self.temp_line.remove()
            except:
                pass
            self.temp_line = None
    
    def cancel_calibration(self, ax: Optional[plt.Axes] = None):
        """Cancel ongoing calibration"""
        print("Scale calibration cancelled")
        self.reset_calibration()
        if ax:
            ax.set_title(self.original_title if hasattr(self, 'original_title') else "")
            ax.figure.canvas.draw()
    
    def is_active(self) -> bool:
        """Check if calibration mode is active"""
        return self.is_calibrating
    
    def get_result(self) -> Optional[float]:
        """Get calibration result (scale factor)"""
        return self.result


def quick_scale_calibration(image: np.ndarray, 
                           known_length_um: float,
                           point1: Tuple[float, float], 
                           point2: Tuple[float, float]) -> float:
    """
    Quick scale calculation without GUI interaction
    
    Args:
        image: Image array (not used, kept for API consistency)
        known_length_um: Known actual length in microns
        point1: Start point (x, y) in pixels
        point2: End point (x, y) in pixels
        
    Returns:
        Scale factor (um/pixel)
    """
    x1, y1 = point1
    x2, y2 = point2
    pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    if pixel_distance == 0:
        raise ValueError("Points are identical, cannot calculate scale factor")
    
    scale_factor = known_length_um / pixel_distance
    
    print(f"Quick scale calibration:")
    print(f"  Pixel distance: {pixel_distance:.2f} px")
    print(f"  Actual length: {known_length_um:.2f} um")
    print(f"  Scale factor: {scale_factor:.4f} um/px")
    
    return scale_factor


# Example usage
if __name__ == "__main__":
    print("Scale Calibration Module")
    print("=" * 60)
    print("This module provides interactive scale calibration for images")
    print("\nUsage:")
    print("  from core.scale_calibration import InteractiveScaleCalibrator")
    print("  calibrator = InteractiveScaleCalibrator()")
    print("  calibrator.calibrate_scale(image, ax, fig, callback)")
    print("\nOr for quick calculation:")
    print("  from core.scale_calibration import quick_scale_calibration")
    print("  scale_factor = quick_scale_calibration(image, 1000.0, (x1,y1), (x2,y2))")
