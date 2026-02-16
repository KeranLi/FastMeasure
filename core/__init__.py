"""
Core Module - Provides shared tools and functionality for FastSAM and MobileSAM
"""

from .seg_tools import ImageProcessor, PolygonUtils, FileUtils, PerformanceMonitor
from .seg_optimize import SmartPostProcessor
from .cli_base import (
    SimpleArgs,
    create_base_parser,
    terminal_interactive_wizard,
    update_config_from_args,
    print_summary,
    print_welcome
)
from .scale_calibration import (
    InteractiveScaleCalibrator,
    quick_scale_calibration
)

__all__ = [
    # Tool classes
    'ImageProcessor',
    'PolygonUtils', 
    'FileUtils',
    'PerformanceMonitor',
    'SmartPostProcessor',
    # CLI classes
    'SimpleArgs',
    'create_base_parser',
    'terminal_interactive_wizard',
    'update_config_from_args',
    'print_summary',
    'print_welcome',
    # Scale calibration
    'InteractiveScaleCalibrator',
    'quick_scale_calibration',
]
