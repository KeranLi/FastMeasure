# Changelog


## 2026-01-04
- 修改人：张立华
- 修改类型：首次上传
- 涉及文件：rock.py、run.py、scale_detector.py、grain_marker.py
- 具体内容：
1. 上传rock.py：封装核心处理类，实现单图/批量图像处理逻辑；
2. 上传run.py：作为启动入口，支持命令行参数、交互式选文件；
3. 上传scale_detector.py：实现比例尺识别、像素-微米换算；
4. 上传grain_marker.py：实现颗粒编号/面积标注、标注重叠规避。

## 2026-01-05
- 修改人：李老师
- 修改类型：Initial Commit（首次提交）
- 涉及文件：Boulder_20260107、efficientsam3 @ faba26f、results/images、segmenteverygrain、.gitignore、.gitmodules、effsam3_inference_test.py、env-KeranLiyi.yml、es-tv-mc-teaser.png
- 具体内容：
1. 提交项目基础文件（含初始代码、环境配置、示例资源）；
2. 未包含模型权重文件。

## 2026-01-08
- 修改人：李老师
- 修改类型：修改+新增
- 涉及文件：README.md、fastsam_inference_test.py
- 具体内容：
1. 将原SAM模型替换为FastSAM并更新README.md说明；
2. 新增FastSAM推理测试脚本；
3. 补充CPU与GPU的推理性能对比报告。

## 2026-01-09
- 修改人：李老师
- 修改类型：新增+更新
- 涉及文件：CHANGELOG.md、mobileam_inference_test.py、result.jpg
- 具体内容：
1. 新增MobileSAM推理测试脚本；
2. 补充推理结果示例图（result.jpg）；
3. 更新CHANGELOG.md记录MobileSAM的性能报告。


## 2026-01-10
- 修改人：李老师
- 修改类型：修改
- 涉及文件：rock.py、rock_new.py、run.py、config.yaml
- 具体内容：
1. 在rock_new.py中适配MobileSAM模型，同步修改run.py的调用逻辑；
2. 更新config.yaml的对应配置参数。


## 2026-01-13
- 修改人：张立华
- 修改类型：新增
- 涉及文件：yolosection.md、segmenteverygrain删监督聚类.py、yolo_sam_segmentation删监督聚类.py
- 具体内容：
1. 首次上传new文件夹的文件说明文档，补充项目文件结构的说明信息。
2.首次上传对代码文件中的无用函数进行了删减。

## 2026-01-15
- 修改人：张立华
- 修改类型：同步项目结构
- 涉及文件：yolo+sam相关代码和yolo+fastasam项目相关代码
- 具体内容：
1. yolo+sam相关代码全都放入new文件夹中，yolo+fastasam项目相关代码放入了new子文件夹super_fastsam当中。
