#!/bin/bash
# FastMeasure macOS 应用运行脚本

APP_PATH="dist/FastMeasure.app"

if [ ! -d "$APP_PATH" ]; then
    echo "错误: 找不到 $APP_PATH"
    echo "请先运行打包脚本: python build_exe_macos.py"
    exit 1
fi

# 检查模型文件
MODELS_DIR="$APP_PATH/Contents/MacOS/models"
if [ ! -f "$MODELS_DIR/best_yolo_20260107.pt" ]; then
    echo "正在复制模型文件..."
    mkdir -p "$MODELS_DIR"
    cp models/*.pt "$MODELS_DIR/" 2>/dev/null || echo "警告: 复制模型文件失败"
fi

echo "启动 FastMeasure..."

# 方法1: 直接打开
open "$APP_PATH"

# 如果方法1不行，取消下面这行的注释来使用方法2:
# "$APP_PATH/Contents/MacOS/FastMeasure"
