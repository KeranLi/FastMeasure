"""
MobileSAM Smart Post-processing Module - Compatibility Wrapper
File: mobilesam/seg_optimize.py
Function: Import SmartPostProcessor from core module for backward compatibility
"""

# Import SmartPostProcessor from core module
from core.seg_optimize import SmartPostProcessor

__all__ = ['SmartPostProcessor']
