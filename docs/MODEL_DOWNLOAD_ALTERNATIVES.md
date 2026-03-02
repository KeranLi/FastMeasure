# 模型文件下载备选方案

由于模型文件较大（~700MB），如果 GitHub Releases 上传困难，可以使用以下备选方案：

---

## 方案 1: 百度网盘（推荐国内用户）

### 上传步骤

1. **登录百度网盘网页版**：https://pan.baidu.com/
2. **上传模型文件**：
   - 可以直接上传单个文件（支持大文件）
   - 或压缩成一个 ZIP 上传
3. **生成分享链接**：
   - 选择文件 → 分享 → 创建链接
   - 设置提取码（建议设置，防止失效）
4. **复制分享链接**，格式如下：
   ```
   链接: https://pan.baidu.com/s/1xxxxx
   提取码: xxxx
   ```

### 在 README 中添加下载说明

```markdown
## 模型文件下载

由于文件较大，模型文件存储在百度网盘：

- **链接**: https://pan.baidu.com/s/1xxxxx
- **提取码**: xxxx

下载后解压到 `models/` 文件夹即可。
```

### 优缺点

| 优点 | 缺点 |
|------|------|
| 国内下载速度快 | 需要提取码 |
| 支持大文件 | 非会员下载限速 |
| 分享稳定 | 国际用户访问不便 |

---

## 方案 2: Google Drive（推荐国际用户）

### 上传步骤

1. **上传文件到 Google Drive**
2. **设置分享权限**：
   - 右键文件 → 分享
   - 选择 "任何知道链接的人"
   - 权限设为 "查看者"
3. **获取直接下载链接**：
   - 分享链接格式：`https://drive.google.com/file/d/FILE_ID/view`
   - 转换为直接下载链接：`https://drive.google.com/uc?export=download&id=FILE_ID`

### 在 README 中添加

```markdown
## Model Download

Download model files from Google Drive:

- **Link**: https://drive.google.com/uc?export=download&id=YOUR_FILE_ID
- **Size**: ~700 MB total

Extract to `models/` folder after download.
```

---

## 方案 3: 分卷压缩上传 GitHub Releases

如果一定要使用 GitHub Releases，可以将大文件分卷压缩（每个 <100MB）：

### Windows 分卷压缩

```powershell
# 使用 7-Zip 分卷压缩（每个 90MB）
7z a -v90m mobile_sam.7z mobile_sam.pt

# 生成文件：
# mobile_sam.7z.001
# mobile_sam.7z.002
# mobile_sam.7z.003
# ...
```

### 用户合并

```bash
# 用户下载所有分卷后合并
7z x mobile_sam.7z.001
```

### 优缺点

| 优点 | 缺点 |
|------|------|
| 保持 GitHub 原生 | 用户操作复杂 |
| 无需第三方平台 | 需要安装 7-Zip |

---

## 方案 4: 阿里云盘 / 天翼云盘

类似百度网盘的流程，但可能速度更快：

- **阿里云盘**：https://www.aliyundrive.com/（不限速）
- **天翼云盘**：https://cloud.189.cn/

---

## 推荐方案

| 场景 | 推荐方案 |
|------|----------|
| 主要用户在国内 | 百度网盘 |
| 主要用户在国际 | Google Drive |
| 混合用户 | 同时提供百度网盘 + Google Drive |
| 坚持 GitHub 原生 | 分卷压缩上传 |

---

## README 示例（同时使用多个平台）

```markdown
## Model Files

Download pre-trained model files (~700 MB total):

### Option 1: Baidu Netdisk (Recommended for China users)
- **Link**: https://pan.baidu.com/s/1xxxxx
- **Extraction code**: xxxx

### Option 2: Google Drive (Recommended for international users)
- **Link**: https://drive.google.com/uc?export=download&id=xxxxx

### Option 3: GitHub Releases
- https://github.com/KeranLi/FastMeasure/releases

### Model File Structure

After downloading, extract to `models/` folder:
```
models/
├── best_yolo_20260107.pt    # ~100 MB
├── FastSAM-s.pt             # ~150 MB
└── mobile_sam.pt            # ~450 MB (optional)
```
```

---

## 当前项目推荐做法

基于你的情况，建议：

1. **主方案**：上传到百度网盘，README 放分享链接
2. **备选方案**：提供 Google Drive 链接（如果有国际用户）
3. **更新下载脚本**：支持从百度网盘下载（需要处理提取码）

或者简化处理：
- 不在代码里自动下载
- 只在 README 提供手动下载链接
- 用户下载后放入 models/ 文件夹即可
