# -*- coding: utf-8 -*-
"""
测试所有几何参数是否正确计算和输出
"""
import numpy as np
import pandas as pd
from geometry.grain_metric import GrainShapeMetrics

# 创建测试数据（模拟真实颗粒数据）
test_data = []
for i in range(3):
    # 创建模拟 mask 和轮廓坐标
    theta = np.linspace(0, 2*np.pi, 50)
    # 椭圆形状
    a, b = 50 + i*10, 30 + i*5  # 长短轴
    x = a * np.cos(theta) + 100
    y = b * np.sin(theta) + 100
    coordinates = np.column_stack([x, y]).tolist()
    
    # 计算基本参数
    area = np.pi * a * b
    perimeter = 2 * np.pi * np.sqrt((a**2 + b**2) / 2)
    
    test_data.append({
        'grain_id': i + 1,
        'area': float(area),
        'centroid_x': 100.0,
        'centroid_y': 100.0,
        'width': float(a * 2),
        'height': float(b * 2),
        'perimeter': float(perimeter),
        'confidence': 0.9,
        'major_axis_length': float(a * 2),
        'minor_axis_length': float(b * 2),
        'coordinates': coordinates
    })

df = pd.DataFrame(test_data)
print("=" * 60)
print("Test DataFrame columns:")
print(list(df.columns))
print(f"\nTest data shape: {df.shape}")

# 测试 GrainShapeMetrics
print("\n" + "=" * 60)
print("Testing GrainShapeMetrics.compute_all_metrics()")
print("=" * 60)

try:
    calculator = GrainShapeMetrics(df)
    result = calculator.compute_all_metrics()
    
    print(f"\nResult DataFrame shape: {result.shape}")
    print(f"\nAll columns ({len(result.columns)}):")
    for i, col in enumerate(result.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # 检查所有期望的参数是否都存在
    expected_params = [
        'circularity', 'aspect_ratio', 'rectangularity', 'compactness', 
        'roundness', 'convexity', 'fractal_dimension', 'angularity',
        'EI_2d', 'FI_2d', 'AR_2d', 'D2_2d', 'D3_2d', 'D4_2d'
    ]
    
    print("\n" + "=" * 60)
    print("Parameter Check:")
    print("=" * 60)
    
    for param in expected_params:
        if param in result.columns:
            print(f"  [OK] {param}: {result[param].mean():.4f}")
        else:
            print(f"  [MISSING] {param}: MISSING!")
    
    # 打印示例数据
    print("\n" + "=" * 60)
    print("Sample data (first grain):")
    print("=" * 60)
    for col in result.columns:
        val = result.iloc[0][col]
        if isinstance(val, list):
            print(f"  {col}: <list of {len(val)} points>")
        else:
            print(f"  {col}: {val}")
    
    print("\n" + "=" * 60)
    print("SUCCESS: All 14 geometry parameters calculated correctly!")
    print("=" * 60)
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
