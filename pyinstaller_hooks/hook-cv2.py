# PyInstaller hook for OpenCV (cv2)
# This resolves the recursion error and missing config during cv2 loading

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files, collect_submodules
import cv2
import os

# Get cv2 module path
cv2_path = os.path.dirname(cv2.__file__)

# Collect all OpenCV binaries
binaries = collect_dynamic_libs('cv2')

# Collect ALL cv2 data files (including config.py)
datas = collect_data_files('cv2', includes=['**/*'])

# Also explicitly add the config file if not captured
config_path = os.path.join(cv2_path, 'config.py')
if os.path.exists(config_path):
    datas.append((config_path, 'cv2'))

# Collect all submodules to ensure nothing is missing
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

# Exclude problematic imports
excludedimports = []
