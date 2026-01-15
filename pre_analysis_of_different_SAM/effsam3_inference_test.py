# 这里我是Windows的环境没有办法安装Triton

from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from PIL import Image

# Load model
model = build_efficientsam3_image_model(
  checkpoint_path="efficient_sam3_efficientvit_s.pt",
  backbone_type="efficientvit",
  model_name="b0",
  enable_inst_interactivity=True,
)

# Process image and predict
image = Image.open("es-tv-mc-m-teaser.png")
processor = Sam3Processor(model)
inference_state = processor.set_image(image)

# Single positive point prompt (x, y) in pixels
points = [[image.size[0] / 2, image.size[1] / 2]]
labels = [1]
masks, scores, _ = model.predict_inst(
    inference_state, 
    point_coords=points, 
    point_labels=labels
)