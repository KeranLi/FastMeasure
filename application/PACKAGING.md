# FastMeasure 打包指南

## 前置条件

### 1. 安装依赖

```bash
# 进入项目目录
cd f:\code\rock-yolo

# 激活 conda 环境
conda activate fastmeasure-clean

# 安装 PyInstaller
pip install pyinstaller
```

### 2. 准备模型文件（重要！）

打包**之前**需要下载模型文件：

```bash
# 检查模型文件
python utils/download_models.py
```

确保以下文件存在于 `models/` 目录：
- `best_yolo_20260107.pt` (YOLO 检测模型, ~100MB)
- `FastSAM-s.pt` (FastSAM 模型, ~150MB)
- `mobile_sam.pt` (MobileSAM 模型, ~450MB, 可选)

**注意**：模型文件**不**包含在打包中（太大），用户需要单独下载。

## 打包步骤

### 方式一：使用构建脚本（推荐）

```bash
# 运行构建脚本
python application/build_exe.py
```

构建过程：
1. 清理之前的构建文件
2. 检查依赖和目录
3. 运行 PyInstaller 打包
4. 创建 models/ 和 results/ 目录
5. 生成 README.txt
6. 生成 installer.iss（Windows 安装脚本）

### 方式二：手动打包

```bash
# Windows 命令
pyinstaller --name FastMeasure ^
    --windowed ^
    --onedir ^
    --add-data "core;core" ^
    --add-data "fastsam;fastsam" ^
    --add-data "mobilesam;mobilesam" ^
    --add-data "geometry;geometry" ^
    --add-data "configs;configs" ^
    --add-data "utils;utils" ^
    --add-data "run.py;." ^
    --add-data "run_fastsam.py;." ^
    --add-data "run_mobilesam.py;." ^
    --hidden-import "core" ^
    --hidden-import "core.segment_core" ^
    --hidden-import "geometry.grain_metric" ^
    --hidden-import "fastsam.rock_fastsam_system" ^
    --hidden-import "mobilesam.rock_mobilesam_system" ^
    --hidden-import "sklearn.cluster" ^
    --hidden-import "skimage.measure" ^
    --hidden-import "scipy.spatial" ^
    --hidden-import "shapely.geometry" ^
    --hidden-import "timm" ^
    --collect-all "ultralytics" ^
    --collect-all "torch" ^
    --collect-data "cv2" ^
    --exclude-module "pytest" ^
    --exclude-module "unittest" ^
    --clean ^
    utils/gui_launcher.py
```

## 打包输出

构建完成后，输出目录结构：

```
dist/
└── FastMeasure/                 # 主程序目录
    ├── FastMeasure.exe          # 可执行文件（入口）
    ├── models/                  # 模型目录（空，需用户放置模型）
    ├── results/                 # 结果输出目录
    ├── README.txt               # 使用说明
    ├── _internal/               # Python 库和依赖
    │   ├── core/                # 核心模块
    │   ├── fastsam/             # FastSAM 模块
    │   ├── mobilesam/           # MobileSAM 模块
    │   ├── geometry/            # 几何计算模块
    │   ├── configs/             # 配置文件
    │   └── ...                  # 其他依赖
    └── ...
```

## 分发准备

### 步骤 1：复制模型文件（测试用）

```bash
# 复制模型到打包目录（仅用于测试）
copy models\best_yolo_20260107.pt dist\FastMeasure\models\
copy models\FastSAM-s.pt dist\FastMeasure\models\
copy models\mobile_sam.pt dist\FastMeasure\models\
```

### 步骤 2：测试运行

```bash
# 运行测试
dist\FastMeasure\FastMeasure.exe
```

测试功能：
- [ ] 启动 GUI
- [ ] 选择图片/文件夹
- [ ] 运行 FastSAM 单图处理
- [ ] 运行 MobileSAM 单图处理
- [ ] 批量处理
- [ ] 交互模式

### 步骤 3：创建分发包

```bash
# 方式 1：压缩整个文件夹
cd dist
zip -r FastMeasure_v1.0.0.zip FastMeasure/

# 方式 2：Windows 压缩
# 右键 FastMeasure 文件夹 → 发送到 → 压缩文件夹
```

## 创建 Windows 安装程序（可选）

### 安装 Inno Setup

1. 下载并安装 [Inno Setup](https://jrsoftware.org/isinfo.php)

### 构建安装程序

```bash
# 构建脚本已生成 installer.iss
# 使用 Inno Setup 编译器打开并构建
```

或在命令行：

```bash
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
```

输出：`installer/FastMeasure_Setup.exe`

## 常见问题

### 1. 打包失败：找不到模块

**症状**：`ModuleNotFoundError: No module named 'xxx'`

**解决**：添加 `--hidden-import` 参数，编辑 `build_exe.py`：

```python
"--hidden-import", "缺失的模块名",
```

### 2. 打包后运行时崩溃

**症状**：程序启动后立即崩溃

**检查**：
1. 模型文件是否放置在 `models/` 目录
2. 检查 `configs/` 目录是否存在
3. 查看 `_internal/` 中的日志

### 3. 文件过大

**当前大小**：~500MB-1GB（包含 PyTorch）

**优化建议**：
- 使用 `--exclude-module` 排除不需要的模块
- 已经排除了：`pytest`, `unittest`, `matplotlib.tests`

### 4. 交互模式无法启动

**症状**：点击运行后无响应

**检查**：
- 打包时是否正确包含 `matplotlib` 后端
- 检查 `--hidden-import "matplotlib.backends.backend_tkagg"`

## 验证清单

打包完成后，请验证以下功能：

- [ ] 程序能正常启动
- [ ] 能选择图片文件
- [ ] 能选择文件夹
- [ ] FastSAM 单图处理正常
- [ ] MobileSAM 单图处理正常
- [ ] 批量处理正常
- [ ] 交互模式正常
- [ ] 结果保存到 results/ 目录
- [ ] CSV 文件包含所有几何参数

## 版本信息

- 打包脚本：`application/build_exe.py`
- GUI 入口：`utils/gui_launcher.py`
- 系统模块：`core/`, `fastsam/`, `mobilesam/`, `geometry/`
