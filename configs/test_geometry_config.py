#!/usr/bin/env python3
"""
Test script to verify geometry.yaml is loaded correctly
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("Testing geometry.yaml loading")
print("=" * 60)

# Test 1: Check if file exists
config_path = project_root / "configs" / "geometry.yaml"
print(f"\n1. Config file path: {config_path}")
print(f"   Exists: {config_path.exists()}")

if config_path.exists():
    print(f"   File size: {config_path.stat().st_size} bytes")
    
    # Test 2: Try to load using config_loader
    try:
        from geometry.config_loader import load_geometry_config
        config = load_geometry_config(str(config_path))
        print(f"\n2. Config loaded successfully")
        print(f"   Config content: {config}")
    except Exception as e:
        print(f"\n2. Failed to load config: {e}")

# Test 3: Check if fastsam system loads it
print("\n3. Testing fastsam system...")
try:
    from fastsam.rock_fastsam_system import RockUltraSystem
    # Note: This will try to load models, may fail but should show config loading
    print("   RockUltraSystem imported successfully")
except Exception as e:
    print(f"   Import error (expected if models not present): {e}")

# Test 4: Check if mobilesam system loads it
print("\n4. Testing mobilesam system...")
try:
    from mobilesam.rock_mobilesam_system import RockMobileSystem
    print("   RockMobileSystem imported successfully")
except Exception as e:
    print(f"   Import error (expected if models not present): {e}")

print("\n" + "=" * 60)
print("Test complete")
print("=" * 60)
