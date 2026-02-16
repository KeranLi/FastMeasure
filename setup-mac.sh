#!/bin/bash
# setup-mac.sh - Mac环境安装脚本 for FastMeasure

set -e  # 遇到错误停止

echo "🚀 开始安装 EfficientSAM3 Mac环境..."

# 检查conda
if ! command -v conda &> /dev/null; then
    echo "❌ 未找到conda，请先安装Anaconda或Miniconda"
    echo "   推荐: brew install miniconda"
    exit 1
fi

# 创建环境
echo "📦 创建conda环境: efficientsam3"
conda create -n efficientsam3 python=3.12 -y

# 激活环境
echo "🔧 激活环境"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate efficientsam3

# 安装PyTorch (Mac MPS版)
echo "🔥 安装PyTorch (Mac MPS支持)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装核心依赖
echo "📚 安装核心依赖"
pip install -q \
    opencv-python \
    matplotlib \
    tqdm \
    pyyaml \
    scipy \
    pandas \
    pillow \
    numpy \
    ultralytics \
    timm \
    transformers \
    accelerate \
    scikit-image \
    scikit-learn \
    tensorboard \
    hydra-core \
    omegaconf \
    einops \
    fairscale \
    fvcore \
    iopath \
    portalocker \
    tabulate \
    yacs \
    yapf \
    pycocotools \
    rasterio \
    shapely \
    segment-anything \
    decord \
    imageio \
    tifffile \
    addict \
    regex \
    ftfy \
    rich \
    typer-slim \
    submitit \
    polars \
    mmengine \
    mmcv

# 安装TensorFlow (Mac优化版)
echo "🧠 安装TensorFlow (Mac版)"
pip install -q tensorflow-macos
pip install -q tensorflow-metal  # MPS加速

# 检查安装
echo ""
echo "✅ 安装完成！验证中..."

python << EOF
import torch
import tensorflow as tf

print(f"PyTorch: {torch.__version__}")
print(f"MPS可用: {torch.backends.mps.is_available()}")
print(f"TensorFlow: {tf.__version__}")

if torch.backends.mps.is_available():
    print("🎉 MPS加速已启用！")
else:
    print("⚠️ MPS不可用，将使用CPU")
EOF

echo ""
echo "📝 使用方式:"
echo "   conda activate efficientsam3"
echo "   python your_script.py"