# -*- coding: utf-8 -*-
"""Check which methods in GrainShapeMetrics are actually used"""
import inspect
from geometry.grain_metric import GrainShapeMetrics

# Get all methods
all_methods = [m for m in dir(GrainShapeMetrics) 
               if not m.startswith('_') and callable(getattr(GrainShapeMetrics, m))]

# Get compute_all_metrics source
source = inspect.getsource(GrainShapeMetrics.compute_all_metrics)

print("=" * 60)
print("GrainShapeMetrics Method Usage Analysis")
print("=" * 60)

print("\n[1] Methods USED in compute_all_metrics():")
used = []
for m in all_methods:
    if m != 'compute_all_metrics' and m in source:
        print(f"  + {m}")
        used.append(m)

print("\n[2] Methods NOT used:")
unused = []
for m in all_methods:
    if m != 'compute_all_metrics' and m not in source:
        print(f"  - {m}")
        unused.append(m)

print("\n[3] Summary:")
print(f"  Total methods: {len(all_methods) - 1}")  # Exclude __init__
print(f"  Used: {len(used)}")
print(f"  Unused: {len(unused)}")

print("\n[4] Potential Issues:")
# Check for duplicate calls
if source.count('calculate_fractal_dimension') > 1:
    print("  ! calculate_fractal_dimension is called TWICE (redundant)")

# Check methods that need 'coordinates' column
needs_coords = ['calculate_fourier_descriptors', 'calculate_angularity']
for m in needs_coords:
    if m in used:
        print(f"  ! {m} requires 'coordinates' column (not in basic data)")
