
# 岩石颗粒自动分割系统

## 项目概述

岩石颗粒自动分割系统是一款用于处理岩石显微图像、自动检测并分割颗粒的工具。该系统使用YOLO和SAM模型结合比例尺检测，能够从岩石显微图像中提取颗粒信息，并通过图形标注展示颗粒编号、面积等信息。系统支持单张图像处理和批量图像处理功能。

## 功能

- **比例尺检测**：自动从图像中识别比例尺，并计算每个像素对应的实际微米数。
- **颗粒分割**：使用YOLO和SAM模型对岩石显微图像进行颗粒分割。
- **颗粒标注**：为分割出的颗粒自动添加编号和面积标注，支持定制样式（字体、颜色、背景等）。
- **批量处理**：支持对文件夹中的多个图像进行批量处理。
- **交互式模式**：提供图形界面选择文件进行处理。

## 安装

1. 克隆本仓库到本地：
    ```bash
    git clone https://github.com/yourusername/rock_segmentation_system.git
    cd rock_segmentation_system
    ```

2. 创建并激活虚拟环境（推荐使用`conda`）：
    ```bash
    conda create -n rockseg python=3.8
    conda activate rockseg
    ```

3. 安装依赖：
    ```bash
    pip install -r requirements.txt
    ```

4. 配置文件：
    - 默认配置文件 `config/config.yaml` 存储了所有的系统参数。你可以根据需要修改该文件的配置。

## 使用

### 启动脚本

#### 处理单张图片
```bash
python run.py --input path/to/image.tif
```

#### 批量处理文件夹中的所有图片
```bash
python run.py --input path/to/folder --batch
```

#### 交互式模式（图形界面选择文件）
```bash
python run.py --interactive
```

#### 自定义配置文件
```bash
python run.py --config custom_config.yaml --input path/to/image.tif
```

### 配置文件参数说明
你可以通过`config/config.yaml`文件修改以下参数：
- **模型路径**：`model_paths`（YOLO、SAM模型路径）
- **比例尺检测**：`scale_detection`（包括比例尺的参数设置）
- **输出目录**：`output`（保存结果的路径）
- **处理参数**：`processing`（颗粒检测的置信度、最小面积等）
- **日志设置**：`logging`（日志级别和输出方式）

### 输出文件

- **分割结果图**：保存分割后的图像（`segmentation_result.png`）。
- **标注结果图**：保存带有颗粒标注的图像（`segmentation_labeled.png`）。
- **颗粒统计数据**：保存颗粒统计的CSV文件（`grain_statistics.csv`）。
- **分割掩码**：保存图像分割的掩码文件（`segmentation_mask.png`）。
- **总结报告**：JSON格式的图像处理总结（`summary.json`）。

## 依赖

- `torch`：用于加载YOLO和SAM模型。
- `opencv`：用于图像处理和比例尺检测。
- `pandas`：用于数据处理和统计。
- `matplotlib`：用于结果可视化和图像标注。

## 贡献

欢迎贡献！如果你有改进建议或者发现问题，可以通过提交`issue`或`pull request`来贡献代码。
