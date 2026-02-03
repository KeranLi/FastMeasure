
# FastMeasure - 岩石颗粒自动分割系统

## 项目概述

岩石颗粒自动分割系统是一款用于处理岩石显微图像、自动检测并分割颗粒的专业工具。该系统基于深度学习技术，支持 **YOLO+FastSAM** 和 **YOLO+MobileSAM** 两种模型组合，结合智能比例尺检测和丰富的几何参数计算，能够从岩石显微图像中精确提取颗粒信息，并生成完整的统计分析报告。

系统支持三种使用模式：
- **自动处理模式**：YOLO检测 + SAM自动分割
- **批量处理模式**：对整个文件夹的图片进行批量处理
- **交互式模式**：通过GUI手动点选颗粒进行精细分割

## 核心功能

### 1. 双模型支持
| 模型组合 | 特点 | 适用场景 |
|---------|------|---------|
| YOLO + FastSAM | 速度快、轻量级 | 大批量快速处理 |
| YOLO + MobileSAM | 精度高、支持交互 | 高精度要求、交互式标注 |

### 2. 比例尺检测
- 自动识别图像右下角的红色比例尺条
- 计算像素与实际微米数的转换因子
- 支持自定义比例尺长度配置

### 3. 颗粒分割与标注
- 自动颗粒检测与分割
- 智能颗粒编号与面积标注
- 支持自定义标注样式（字体、颜色、描边等）

### 4. 几何参数计算
系统可计算多种颗粒几何参数：
- **基础参数**：面积、周长、质心坐标、外接矩形
- **形状参数**：圆度(Circularity)、长宽比(Aspect Ratio)、矩形度(Rectangularity)
- **结构参数**：压实度(Compactness)、磨圆度(Roundness)、凸度(Convexity)
- **高级参数**：分形维数(Fractal Dimension)、棱角度(Angularity)

### 5. 灵活的配置系统
- `config.yaml` / `config_mobilesam.yaml`：主配置文件（模型路径、处理参数、输出设置）
- `geometry_config.yaml`：几何参数配置文件（自定义CSV导出字段）

## 安装

### 环境要求
- Python 3.8+
- PyTorch
- CUDA（推荐，用于GPU加速）

### 安装步骤

1. 克隆本仓库到本地：
    ```bash
    git clone https://github.com/KeranLi/FastMeasure.git
    cd FastMeasure
    ```

2. 创建并激活虚拟环境（推荐使用 `conda`）：
    ```bash
    conda create -n rockseg python=3.8
    conda activate rockseg
    ```

3. 安装依赖：
    ```bash
    pip install torch torchvision opencv-python pandas matplotlib numpy pyyaml ultralytics shapely scikit-image pillow
    
    # MobileSAM交互模式需要额外安装
    pip install mobile_sam
    ```

4. 准备模型文件：
    - YOLO模型：`./models/best_yolo_20260107.pt`（FastSAM流程）或 `./models/best.pt`（MobileSAM流程）
    - FastSAM模型：`./models/FastSAM-s.pt`
    - MobileSAM模型：`./models/mobile_sam.pt`

## 使用指南

### 一、FastSAM 处理流程

#### 1. 处理单张图片
```bash
python run_fastsam.py --input path/to/image.tif
```

#### 2. 批量处理文件夹
```bash
python run_fastsam.py --input path/to/folder --batch
```

#### 3. 使用自定义配置
```bash
python run_fastsam.py --config custom_config.yaml --input image.tif
```

#### 4. 调整处理参数
```bash
python run_fastsam.py --input image.tif --conf 0.3 --min-area 50 --output my_results
```

### 二、MobileSAM 处理流程

#### 1. 终端交互模式（新手推荐）
```bash
python run_mobilesam.py
```
按提示选择处理模式和输入参数。

#### 2. 处理单张图片
```bash
python run_mobilesam.py --input path/to/image.tif
```

#### 3. 批量处理文件夹
```bash
python run_mobilesam.py --input path/to/folder --batch
```

#### 4. GUI交互式分割
```bash
python run_mobilesam.py --interactive
```

#### 5. 独立交互式标注工具
```bash
python mobilesam_interactive.py
```

### 三、交互式模式操作说明

| 按键/操作 | 功能 |
|----------|------|
| 左键点击 | 添加前景点（分割目标） |
| 右键点击 | 添加背景点（排除区域） |
| `X` | 删除最后一个颗粒 |
| `D` | 删除所有颗粒 |
| `S` | 保存结果 |
| `Shift+S` | 快速保存完整结果 |
| `C` | 清除所有点标记 |
| `R` | 重置界面 |
| `H` | 显示帮助 |
| `Q` | 退出 |

## 配置文件说明

### 主配置文件 (`config.yaml` / `config_mobilesam.yaml`)

```yaml
# 模型路径配置
model_paths:
  yolo: "./models/best_yolo_20260107.pt"    # YOLO模型路径
  fastsam: "./models/FastSAM-s.pt"          # FastSAM模型路径
  device: "cpu"                              # 运行设备: cpu 或 cuda

# 比例尺检测配置
scale_detection:
  enabled: true
  known_length_um: 1000.0                    # 比例尺实际长度（微米）

# 处理参数配置
processing:
  yolo_confidence: 0.25                      # YOLO检测置信度阈值
  min_area: 30                               # 最小颗粒面积（像素）
  remove_edge_grains: false                  # 是否移除边缘颗粒

# 输出配置
output:
  root_dir: "results"                        # 结果输出目录
  save_visualization: true                   # 保存可视化结果
  save_statistics: true                      # 保存CSV统计文件
  save_summary: true                         # 保存JSON摘要
```

### 几何参数配置文件 (`geometry_config.yaml`)

```yaml
grain_statistics_csv:
  enabled: true
  # 最终写入CSV的列（按此顺序输出）
  keep_columns:
    - label
    - area
    - perimeter
    - circularity
    - aspect_ratio
    - compactness
    - roundness
    - area_um2
    - diameter_um
```

## 输出文件说明

处理完成后，系统会在输出目录生成以下文件：

| 文件名 | 说明 |
|-------|------|
| `segmentation_result.png` | 分割结果可视化图（带颗粒轮廓） |
| `segmentation_labeled.png` | 标注结果图（带颗粒编号和面积） |
| `segmentation_mask.png` | 二值分割掩码图 |
| `grain_statistics.csv` | 颗粒统计数据表格 |
| `summary.json` | 处理摘要信息（JSON格式） |
| `performance.json` | 性能统计信息 |

## 项目结构

```
.
├── run_fastsam.py              # FastSAM启动脚本
├── run_mobilesam.py            # MobileSAM启动脚本（支持交互模式）
├── mobilesam_interactive.py    # MobileSAM独立交互式工具
├── config.yaml                 # FastSAM配置文件
├── config_mobilesam.yaml       # MobileSAM配置文件
├── geometry_config.yaml        # 几何参数配置文件
│
├── fastsam/                    # FastSAM模块
│   ├── rock_fastsam_system.py  # FastSAM主系统
│   ├── yolo_fastsam.py         # YOLO+FastSAM流水线
│   ├── seg_engine.py           # 分割引擎
│   ├── seg_optimize.py         # 分割优化
│   └── seg_tools.py            # 工具函数
│
├── mobilesam/                  # MobileSAM模块
│   ├── rock_mobilesam_system.py  # MobileSAM主系统
│   ├── yolo_mobilesam.py         # YOLO+MobileSAM流水线
│   ├── mobile_sam_engine.py      # MobileSAM引擎
│   ├── seg_optimize.py           # 分割优化
│   └── seg_tools.py              # 工具函数
│
├── geometry/                   # 几何参数计算模块
│   ├── grain_metric.py         # 颗粒形状参数计算
│   ├── config_loader.py        # 配置加载器
│   └── export_csv.py           # CSV导出工具
│
├── scale_detector.py           # 比例尺检测模块
├── grain_marker.py             # 颗粒标注模块
├── models/                     # 模型文件目录
├── results/                    # 默认输出目录
└── Boulder_20260107/           # 测试数据示例
```

## 性能参考

基于 RTX 3060 显卡的性能测试：

| 模型 | GPU推理 | CPU推理 | 速度对比 |
|------|--------|--------|---------|
| FastSAM | ~77ms | ~294ms | CPU约为GPU的4倍 |
| MobileSAM | ~3.7s | ~101s | CPU约为GPU的26倍 |

**建议**：对于大批量处理，推荐使用GPU加速；小批量或测试可使用CPU模式。

## 依赖列表

| 包名 | 用途 |
|------|------|
| `torch` | 深度学习框架 |
| `ultralytics` | YOLOv8和SAM模型 |
| `opencv-python` | 图像处理和比例尺检测 |
| `pandas` | 数据处理和统计 |
| `matplotlib` | 结果可视化 |
| `numpy` | 数值计算 |
| `pyyaml` | 配置文件解析 |
| `shapely` | 几何计算 |
| `scikit-image` | 图像处理工具 |
| `mobile_sam` | MobileSAM库（交互模式需要） |

## 常见问题

**Q: 比例尺检测失败怎么办？**  
A: 检查图片右下角是否有清晰的红色比例尺条，或在配置文件中调整`red_lower1/red_upper1`等颜色阈值参数。

**Q: 如何调整检测灵敏度？**  
A: 修改配置文件中的`yolo_confidence`参数（值越小检测越灵敏，但可能引入噪声）。

**Q: 交互模式无法启动GUI？**  
A: 确保系统有图形界面支持，或尝试设置环境变量`MPLBACKEND=TkAgg`。

## 更新日志

详见 [CHANGELOG.md](CHANGELOG.md) 了解项目各版本的详细更新内容。

## 贡献

欢迎贡献！如果你有改进建议或者发现问题，可以通过提交 `issue` 或 `pull request` 来贡献代码。

## 许可证

[LICENSE](LICENSE)
