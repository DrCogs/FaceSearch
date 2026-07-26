# FaceSearch

基于 **SeetaFace6** 的人脸识别与搜索工具，支持实时摄像头拍照、超分增强和特征比对识别。

## 功能特性

- **实时摄像头拍照**：打开摄像头实时预览，点击窗口拍照捕获人脸
- **超分辨率增强**：使用 ESPCN 模型对抓拍人脸做 2x 超分，提升小脸识别率
- **CLAHE 预处理**：在 Lab 色彩空间的 L 通道做直方图均衡化，增强光照鲁棒性
- **双重阈值识别**：余弦相似度阈值 + Top-1/Top-2 边距判定，输出"确认/不确定/未知"三级结果
- **按人聚合匹配**：同一人的多条特征记录取 Top-3 平均相似度，提高识别稳定性
- **JSON 配置文件**：摄像头 ID、分辨率、识别阈值等参数均可通过 `config.json` 配置
- **无窗口终端运行**：Windows 下以无控制台窗口模式运行，通过弹窗交互

## 项目结构

```
.
├── FaceSearch.cpp            # 主程序源码
├── config.json              # 运行配置文件
├── models/                  # 模型文件目录
│   ├── face_detector.csta           # 人脸检测模型
│   ├── face_landmarker_pts5.csta    # 5点人脸关键点模型
│   ├── face_recognizer.csta        # 人脸特征提取模型
│   └── ESPCN_x2.pb                 # 超分辨率模型（2x）
└── face_features.db         # 人脸特征数据库（由 FaceManager 生成）
```

## 环境要求

| 依赖项           | 说明                                              |
| ---------------- | ------------------------------------------------- |
| C++17 或更高     | 使用了 `std::filesystem` 等特性                  |
| OpenCV           | 图像处理、摄像头采集、DNN 超分辨率                |
| SeetaFace6       | 人脸检测、关键点检测、特征提取                    |
| nlohmann/json    | JSON 配置文件解析                                 |
| tinyfiledialogs  | 跨平台原生文件对话框与消息弹窗                    |
| CMake            | 构建系统（推荐）                                  |

### 运行时文件

程序运行前需确保以下文件就位：

- `models/face_detector.csta` — 人脸检测模型
- `models/face_landmarker_pts5.csta` — 人脸关键点检测模型
- `models/face_recognizer.csta` — 人脸特征提取模型
- `models/ESPCN_x2.pb` — ESPCN 超分辨率模型
- `config.json` — 运行配置文件（不存在时弹出文件选择对话框）
- `face_features.db` — 人脸特征数据库（由 FaceManager 生成）

### 配置文件示例（config.json）

```json
{
  "camera_id": 0,
  "cap_width": 1200,
  "cap_height": 800,
  "RECOGNITION_THRESHOLD": 0.7,
  "MARGIN_THRESHOLD": 0.15,
  "DETECT_THRESHOLD_HI": 0.9,
  "DETECT_THRESHOLD_LO": 0.7
}
```

## 使用流程

```
启动程序 → 加载配置与模型 → 打开摄像头实时预览
    → 点击拍照 → 超分增强 → 特征比对 → 显示识别结果
    → 确认身份（不确认则重新拍照）
```

## 依赖库

本项目依赖以下第三方库：

### SeetaFace6
- 开源人脸识别算法库（人脸检测 / 关键点 / 识别 / 活体 / 口罩检测等）
- 开发包下载地址见官方 README：
  https://github.com/seetafaceengine/SeetaFace6

### OpenCV contrib
- 需安装带 `contrib` 模块的 OpenCV
- 下载地址：https://github.com/opencv/opencv_contrib
