"""
简化的颗粒分割工具库
只保留核心功能，移除无监督聚类和复杂重叠处理
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from skimage import measure
from skimage.measure import regionprops
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import translate
import rasterio
from rasterio.features import rasterize
from PIL import Image
import tensorflow as tf
from keras import Model
from keras.layers import Input, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from keras.utils import load_img
from keras.saving import load_model
from keras.optimizers import Adam

# ==================== 核心分割函数 ====================

def one_point_prompt(x, y, image, predictor, ax=False):
    """
    简化的单点提示SAM分割
    """
    input_point = np.array([[x, y]])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    new_masks = []
    new_scores = []
    if len(masks) >= 1:
        for ind in range(len(masks)):
            # 过滤掉过大的mask（超过图像50%）
            if np.sum(masks[ind]) / (image.shape[0] * image.shape[1]) <= 0.5:
                new_scores.append(scores[ind])
                if masks.ndim > 2:
                    new_masks.append(masks[ind, :, :])
                else:
                    new_masks.append(masks[ind])
    
    if len(new_masks) > 0:
        masks = new_masks
        scores = new_scores
        ind = np.argmax(scores)
        mask = masks[ind]
        
        # 获取mask的轮廓
        contours = measure.find_contours(mask, 0.5)
        if len(contours) > 0:
            sx = contours[0][:, 1]
            sy = contours[0][:, 0]
        else:
            sx = []
            sy = []
        
        # 可视化（可选）
        if len(sx) > 0 and ax:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            ax.fill(sx, sy, facecolor=color, edgecolor="k", alpha=0.5)
    else:
        sx = []
        sy = []
        mask = np.zeros_like(image[:, :, 0])
    
    return sx, sy, mask


def collect_polygon_from_mask(
    labels,
    mask,
    image_pred,
    all_grains,
    sx,
    sy,
    min_area=100,
    max_n_large_grains=10,
    max_bg_fraction=0.7
):
    """
    简化的从mask收集多边形
    """
    # 简化的逻辑：只要有mask就创建多边形，跳过复杂的重叠检测
    if len(sx) > 0 and len(sy) > 0:
        try:
            poly = Polygon(np.vstack((sx, sy)).T)
            if not poly.is_valid:
                poly = poly.buffer(0)
            
            # 只检查最小面积
            if poly.area >= min_area:
                all_grains.append(poly)
        except Exception as e:
            # 如果创建多边形失败，跳过
            pass
    
    return all_grains


def rasterize_grains(all_grains, image):
    """
    光栅化多边形到图像
    """
    if len(all_grains) == 0:
        return np.zeros(image.shape[:2], dtype=np.int32)
    
    labels = np.arange(1, len(all_grains) + 1)
    shapes_with_labels = zip(all_grains, labels)
    
    out_shape = image.shape[:2]
    bounds = (-0.5, image.shape[0] - 0.5, image.shape[1] - 0.5, -0.5)
    transform = rasterio.transform.from_bounds(*bounds, out_shape[1], out_shape[0])
    
    rasterized = rasterize(
        ((poly, label) for poly, label in shapes_with_labels),
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="int32",
    )
    
    return rasterized


def create_labeled_image(all_grains, image):
    """
    创建带标签的图像
    """
    if len(all_grains) == 0:
        return np.zeros(image.shape[:2], dtype=np.int32), np.zeros(image.shape[:-1])
    
    # 光栅化颗粒
    rasterized = rasterize_grains(all_grains, image)
    
    # 创建边界
    boundaries = []
    for grain in all_grains:
        boundaries.append(grain.boundary.buffer(2))
    
    # 光栅化边界
    boundaries_rasterized = rasterize_grains(boundaries, image)
    
    # 合并掩码
    mask_all = np.zeros(image.shape[:-1])
    mask_all[rasterized > 0] = 1
    mask_all[boundaries_rasterized >= 1] = 2
    
    return rasterized.astype("int"), mask_all


# ==================== 可视化函数 ====================

def plot_image_w_colorful_grains(
    image, all_grains, ax, cmap="viridis", plot_image=True, im_alpha=1.0
):
    """
    用彩色显示颗粒分割结果
    """
    if plot_image:
        ax.imshow(image, alpha=im_alpha)
    
    # 为每个颗粒生成随机颜色
    cmap_obj = plt.cm.get_cmap(cmap)
    num_colors = len(all_grains)
    color_indices = np.random.randint(0, cmap_obj.N, num_colors)
    colors = [cmap_obj(i) for i in color_indices]
    
    for i, grain in enumerate(all_grains):
        color = colors[i]
        ax.fill(
            grain.exterior.xy[0],
            grain.exterior.xy[1],
            facecolor=color,
            edgecolor="none",
            linewidth=1,
            alpha=0.5,
        )
        ax.plot(
            grain.exterior.xy[0],
            grain.exterior.xy[1],
            color="k",
            linewidth=1,
        )
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    plt.tight_layout()


def plot_grain_axes_and_centroids(all_grains, labels, ax, linewidth=1, markersize=10):
    """
    绘制颗粒的轴线和质心
    """
    if len(all_grains) == 0:
        return
    
    # 获取区域属性
    regions = regionprops(labels.astype("int"))
    
    for region in regions:
        y0, x0 = region.centroid
        orientation = region.orientation
        
        # 短轴
        x1 = x0 + np.cos(orientation) * 0.5 * region.minor_axis_length
        y1 = y0 - np.sin(orientation) * 0.5 * region.minor_axis_length
        
        # 长轴
        x2 = x0 - np.sin(orientation) * 0.5 * region.major_axis_length
        y2 = y0 - np.cos(orientation) * 0.5 * region.major_axis_length
        
        # 绘制
        ax.plot((x0, x1), (y0, y1), "-k", linewidth=linewidth)
        ax.plot((x0, x2), (y0, y2), "-k", linewidth=linewidth)
        ax.plot(x0, y0, ".k", markersize=markersize)


# ==================== 简化的U-Net函数（备用） ====================

def Unet():
    """
    简化的U-Net模型（备用）
    """
    tf.keras.backend.clear_session()
    
    inputs = Input((256, 256, 3), name="input")
    
    # 编码器
    conv1 = Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(16, (3, 3), activation="relu", padding="same")(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(32, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # 瓶颈
    conv3 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv3)
    conv3 = BatchNormalization()(conv3)
    
    # 解码器
    up4 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(conv3)
    up4 = concatenate([up4, conv2])
    conv4 = Conv2D(32, (3, 3), activation="relu", padding="same")(up4)
    conv4 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv4)
    conv4 = BatchNormalization()(conv4)
    
    up5 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same")(conv4)
    up5 = concatenate([up5, conv1])
    conv5 = Conv2D(16, (3, 3), activation="relu", padding="same")(up5)
    conv5 = Conv2D(16, (3, 3), activation="relu", padding="same")(conv5)
    conv5 = BatchNormalization()(conv5)
    
    # 输出
    conv6 = Conv2D(3, (1, 1), activation="softmax")(conv5)
    
    model = Model(inputs=[inputs], outputs=[conv6])
    return model


def weighted_crossentropy(y_true, y_pred):
    """
    加权交叉熵损失函数
    """
    class_weights = tf.constant([[[[0.6, 1.0, 5.0]]]])
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(
        labels=y_true, logits=y_pred
    )
    weights = tf.reduce_sum(class_weights * y_true, axis=-1)
    weighted_losses = weights * unweighted_losses
    loss = tf.reduce_mean(weighted_losses)
    return loss


# ==================== 辅助函数 ====================

def predict_image_tile(im_tile, model):
    """
    预测单个图像块
    """
    im_tile = np.expand_dims(im_tile, axis=0)
    im_tile_pred = model.predict(im_tile, verbose=0)
    im_tile_pred = im_tile_pred[0]
    return im_tile_pred


def calculate_iou(poly1, poly2):
    """
    计算两个多边形的IoU（简化的备用函数）
    """
    if not poly1.is_valid:
        poly1 = poly1.buffer(0)
    if not poly2.is_valid:
        poly2 = poly2.buffer(0)
    
    if not poly1.intersects(poly2):
        return 0.0
    
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    
    return intersection_area / union_area if union_area > 0 else 0.0


# ==================== 文件保存/读取函数 ====================

import json
from shapely.geometry import mapping, shape

def save_polygons(polygons, fname):
    """
    保存多边形到GeoJSON文件
    """
    geojson_data = {"type": "FeatureCollection", "features": []}
    
    for polygon in polygons:
        feature = {
            "type": "Feature",
            "geometry": mapping(polygon),
            "properties": {},
        }
        geojson_data["features"].append(feature)
    
    with open(fname, "w") as f:
        json.dump(geojson_data, f)


def read_polygons(fname):
    """
    从GeoJSON文件读取多边形
    """
    with open(fname, "r") as f:
        geojson_data = json.load(f)
    
    polygons = []
    for feature in geojson_data["features"]:
        geometry = feature["geometry"]
        polygons.append(shape(geometry))
    
    return polygons