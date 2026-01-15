# 这里我准备测试MobileSAM
from ultralytics import SAM
import matplotlib.pyplot as plt

# Load model
model = SAM("./models/mobile_sam.pt")

# Process image and predict
source = "es-tv-mc-m-teaser.png"

# CPU
#everything_results = model(source, device="cpu", retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

# GPU
everything_results = model(source, device="0", retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

# 处理结果列表
for result in everything_results:
    boxes = result.boxes  # 边界框输出的Boxes对象
    masks = result.masks  # 分割掩码输出的Masks对象
    keypoints = result.keypoints  # 姿态输出的Keypoints对象
    probs = result.probs  # 分类输出的Probs对象
    obb = result.obb  # OBB输出的Oriented boxes对象
    result.show()  # 显示到屏幕
    result.save(filename="result.jpg")  # 保存到磁盘