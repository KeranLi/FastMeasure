# PyInstaller hook for OpenCV (cv2)
# 解决 OpenCV 配置文件缺失问题

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files, collect_submodules
import cv2
import os

# Get cv2 module path
cv2_path = os.path.dirname(cv2.__file__)

# Collect all OpenCV binaries
binaries = collect_dynamic_libs('cv2')

# Collect ALL cv2 data files
datas = collect_data_files('cv2', includes=['**/*'])

# 显式添加所有配置文件（关键修复）
config_files = [
    'config.py',
    'config-3.py', 
    'config-3.12.py',
    'load_config_py2.py',
    'load_config_py3.py'
]

for config_file in config_files:
    full_path = os.path.join(cv2_path, config_file)
    if os.path.exists(full_path):
        datas.append((full_path, 'cv2'))
        print(f"[hook-cv2] Added config: {config_file}")

# Collect all submodules
hiddenimports = collect_submodules('cv2')
hiddenimports += [
    'cv2.cv2',
    'cv2.config',
    'cv2.load_config_py2',
    'cv2.load_config_py3',
    'cv2.gapi',
    'cv2.gapi.wip',
    'cv2.gapi.wip.draw',
    'cv2.gapi.wip.gst',
    'cv2.gapi.wip.ovis',
    'cv2.gapi.wip.render',
    'cv2.mat_wrapper',
]

excludedimports = []
