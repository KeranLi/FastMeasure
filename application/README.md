# FastMeasure Application 打包目录

这个目录包含 FastMeasure 应用的打包相关文件。

## 目录结构

```
application/
├── build_exe_macos.py      # macOS 应用打包脚本
├── gui_launcher_simple.py  # GUI 启动器（主入口）
├── gui_interactive.py      # 交互式分割界面
├── main_launcher.py        # 命令行启动器
├── run_app.sh              # 应用运行脚本
├── pyinstaller_hooks/      # PyInstaller 钩子
│   └── hook-cv2.py        # OpenCV 钩子
└── README.md              # 本文件
```

## 使用方法

### 打包应用

```bash
cd application
python build_exe_macos.py
```

打包完成后，应用会生成在 `../dist/FastMeasure.app`。

### 复制模型文件

```bash
cp ../models/*.pt ../dist/FastMeasure.app/Contents/MacOS/models/
```

### 运行应用

```bash
open ../dist/FastMeasure.app
```

## 文件说明

### gui_launcher_simple.py
主 GUI 入口，提供模式选择（Auto/Batch/Interactive）和参数设置。

### gui_interactive.py
交互式分割界面，支持：
- FastSAM：点击选择预计算的掩码
- MobileSAM：点提示实时分割
- 比例尺标定（按 'm' 键）

### main_launcher.py
命令行启动器，用于 Auto/Batch 模式的后台处理。

### build_exe_macos.py
PyInstaller 打包脚本，生成独立的 macOS 应用。

## 依赖

打包需要安装 PyInstaller：

```bash
pip install pyinstaller
```

所有其他依赖都在 `efficientsam3` Conda 环境中。
