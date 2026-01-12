# 岩石颗粒分割脚本（new文件夹内代码说明文件）
## 代码文件清单
以下是我的代码文件，对应岩石颗粒自动分割系统的核心功能实现：

| 代码文件 | 所在路径 | 核心作用 |
|----------|----------|----------|
| rock.py | ../new | 系统核心处理脚本，封装所有图像处理逻辑 |
| run.py | ../new | 脚本启动入口，解析命令行参数并调用rock.py |
| scale_detector.py | ../new | 比例尺检测专属脚本，实现像素-微米换算 |
| grain_marker.py | ../new | 颗粒标注专属脚本，实现编号/面积标注及防重叠 |
| yolo_sam_segmentation.py | ../segmenteverygrain/ | sam模型颗粒分割核心脚本，导入rock.py实现YOLO+SAM联合分割 |

## 核心文件功能说明
### 1. rock.py（核心处理脚本）
- 核心类：RockSegmentationSystem（封装全流程逻辑）
- 关键方法：
  - __init__：加载配置文件（对应项目的config/config.yaml），初始化比例尺检测器、日志
  - initialize_models：加载YOLO/SAM模型（读取config中model_paths参数），支持GPU/CPU自动适配
  - process_single_image：单张图片处理核心逻辑（调用scale_detector.py检测比例尺 → 调用yolo_sam_segmentation.py分割颗粒 → 调用grain_marker.py标注 → 保存结果）
  - process_batch：批量处理逻辑，循环调用process_single_image，自动跳过损坏图片

### 2. run.py（启动脚本）
- 解析命令行参数（--input/--batch/--interactive/--config），对应项目的“启动脚本”功能
- 关键逻辑：根据参数调用rock.py中的RockSegmentationSystem类，触发单张/批量/交互式处理
- 交互式模式：通过tkinter实现图形界面选文件，最终仍调用process_single_image处理

### 3. scale_detector.py（比例尺检测）
- 核心函数：
  - detect：识别图像中红色比例尺条，计算“微米/像素”换算因子（对应项目的“比例尺检测”功能）
  - extract_horizontal_line：精确提取比例尺线段长度，排除非水平干扰

### 4. grain_marker.py（颗粒标注）
- 核心函数：
  - add_grain_labels：为分割后的颗粒添加编号、面积标注（读取config中grain_labeling参数定制样式）
  - _find_available_position：标注位置自动调整，避免密集区域标注重叠（对应项目的“颗粒标注”功能）

### 5. yolo_sam_segmentation.py（sam进行分割）
- 核心函数：
  - detect_grains_yolo：YOLO检测颗粒并提取SAM提示点（读取config中processing参数的置信度/最小面积）
  - yolo_sam_segmentation：YOLO检测→SAM分割→重叠多边形合并（对应项目的“颗粒分割”功能）

## 代码调用关系
1. 启动流程：run.py → rock.py（初始化） → 加载config.yaml参数
2. 处理流程：rock.py → 调用scale_detector.py（比例尺） → 调用yolo_sam_segmentation.py（分割） → 调用grain_marker.py（标注） → 保存结果
3. 批量处理：run.py（--batch参数） → rock.py的process_batch方法 → 循环执行单图处理流程

## 与项目配置/运行的关联
- 所有可调参数（模型路径、检测阈值、标注样式）均读取config/config.yaml（对应项目的“配置文件”）
- 运行命令（单张/批量/交互式）均通过run.py触发，参数含义与项目说明一致：
  ```bash
  # 单张图片（调用rock.py的process_single_image）
  python run.py --input path/to/image.tif
  # 批量处理（调用rock.py的process_batch）
  python run.py --input path/to/folder --batch
  # 交互式模式（run.py的tkinter界面）
  python run.py --interactive
  ## 核心调用链路
  # 可视化调用流程
```mermaid
graph TD
A[run.py（启动入口）] -->|传入参数| B[rock.py（核心中枢）]
C[config.yaml（配置文件）] -->|加载参数| B
B -->|调用分割函数| D[yolo_sam_segmentation.py（分割核心）]
B -->|调用比例尺函数| E[scale_detector.py（比例尺计算）]
B -->|调用标注函数| F[grain_marker.py（颗粒标注）]
D -->|加载模型| G[YOLO模型]
D -->|加载模型| H[SAM模型]
D/E/F -->|返回结果| B
B -->|整合输出| I[CSV/PNG/JSON/报告]
