
# Change Log

## [Unreleased]

### Added

- 新增交互式模式，用户可以通过图形界面选择文件进行处理。
- 新增批量处理模式，支持对文件夹中的图像进行批量分割和处理。
- 添加了颗粒标注功能，自动在分割图上添加颗粒编号和面积标注。
- 增加了比例尺检测模块，自动从图像中识别比例尺并计算比例因子。

### Changed

- 改进了比例尺检测算法，优化了检测区域的定位和精确度。
- 改进了批量处理的日志输出，支持详细的处理进度和失败图像报告。
- 重构了代码结构，模块化了各个功能，方便未来扩展。

### Fixed

- 修复了在某些环境下YOLO模型加载失败的问题。
- 修复了图像中比例尺缺失时，比例因子计算不正确的问题。

## [1.0.0] - 2026-01-07

### Added

- 初始发布，包含单张图像处理、比例尺检测、颗粒分割、颗粒标注等功能。
- 支持配置文件调整模型路径、处理参数、输出目录等设置。
- 完成了YOLO和SAM模型的集成，可以进行高效的岩石颗粒分割。
- 输出文件包括分割结果图、带标注图、颗粒统计数据和JSON总结。

### Known Issues

- 在某些极端情况下，颗粒标注可能会重叠，需要进一步优化标注算法。
- 大量图像批量处理时，性能可能需要进一步优化。
- EfficientSAM3需要Triton推理架构，可能不适合window端

## [1.0.0] - 2026-01-08

### Added

- 加入FastSAM的测试脚本fastsam_inference_test.py。

### Known Issues

- 单图下，CPU推理时间是GPU的4倍数(97.8/405.9/379.5 ms in preprocess/inference/postprocess vs.61.1/102.1/68.0 ms, Device=RTX 3060)

## [1.0.0] - 2026-01-09

### Added

- 加入MoblieSAM的测试脚本mobilesam_inference_test.py。

### Known Issues

- 单图下，CPU推理时间是GPU的4倍数(11.5/101268.9/303.8ms in preprocess/inference/postprocess vs.128.7/3760.7/5.1 ms, Device=RTX 3060)

---

## 2026-01-10

- 修改人：李柯然
- 修改类型：修改
- 涉及文件：rock.py、rock_new.py、run.py、config.yaml
  
具体内容：

- 在rock_new.py中适配MobileSAM模型，同步修改run.py的调用逻辑；
- 更新config.yaml的对应配置参数。

---

## 2026-01-13

- 修改人：张立华
- 修改类型：新增
- 涉及文件：yolosection.md、segmenteverygrain删监督聚类.py、yolo_sam_segmentation删监督聚类.py
  
具体内容：

- 首次上传new文件夹的文件说明文档，补充项目文件结构的说明信息。
- 首次上传对代码文件中的无用函数进行了删减。
  
---

## 2026-01-15

- 修改人：张立华
- 修改类型：同步项目结构
- 涉及文件：yolo+sam相关代码和yolo+fastasam项目相关代码
  
具体内容：

- yolo+sam相关代码全都放入new文件夹中，yolo+fastasam项目相关代码放入了new子文件夹super_fastsam当中。

---

## 2026-01-15

- 修改人：李柯然
- 修改类型：同步项目结构
- 涉及文件：yolo+sam相关代码和yolo+fastasam项目相关代码
  
具体内容：

- 更改项目结构。

## 2026-01-19

- 修改人：张立华
- 修改类型：上传代码文件
- 涉及文件：把截至到1月18日最新fsatsam代码放入fastsam文件夹

具体内容：

- fsatsam0118文件夹

---

## 2026-01-19

- 修改人：李柯然
- 修改类型：重新整合代码结构，上传代码文件
- 涉及文件：0118fastsam文件夹

具体内容：

- fsatsam0118文件夹其实就是封装的fastsam，那么只需要和yolo封装好了以后封装成单独文件夹即可。mobile也是这个道理。同时，用户自然习惯在比0118fastsam更高一级的目录中直接使用run_xxx.py的脚本。用户运行完以后就可以直接在同一层级目录中寻找results。我已经预留好了mobilesam文件夹，这次开发可以直接参考当前fastsam的结构开发mobilesam的工具，这样其他脚本都可以直接快速复制改动。
- 注意run_XXX.py脚本输出最好就固定不动了，检查一下，别让大语言模型每次都输出不一样哈。
- "python run_fastsam.py --input ./Boulder_20260107/DSC08059.JPG "我只按照这个方式进行了单图像测试，CPU上效果也很好，目前先用Boulder_20260107文件夹进行测试吧。
- 晚上初步增加了更多的几何参数计算函数，放在了./geometry/grain_metric.py文件
- 修改了rock_fastsam_system.py和yolo_fastsam.py一些代码，做了一些修改，以适配新的集合参数。
- yolo_fastsam.py修改在步骤7计算颗粒属性中。

---

## 2026-01-21

- 修改人：李柯然
- 修改类型：增加根据yaml配置需要计算的几何参数的功能
- 涉及文件：geometry_config.yaml