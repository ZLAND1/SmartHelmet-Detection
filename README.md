# 智行护盔 — 基于YOLOv11的电动车头盔佩戴智能检测预警系统

## 项目简介

本项目是一个基于深度学习的电动车头盔佩戴智能检测预警系统，使用 YOLOv11 目标检测模型实现实时推理。系统可检测图片和视频中的电动车骑行者，自动识别是否佩戴头盔，并在检测到未佩戴头盔时发出语音警报。

### 检测类别

| 类别 ID | 类别名称 | 说明 |
|---------|---------|------|
| 0 | two_wheeler | 两轮电动车/摩托车 |
| 1 | helmet | 佩戴头盔 |
| 2 | without_helmet | 未佩戴头盔 |

---

## 项目包含两个版本

| 版本 | 目录 | 适用平台 | 推理框架 |
|------|------|---------|---------|
| 🖥️ **PC 桌面版** | `pc_app/` | Windows / Linux / macOS | Ultralytics YOLO (PyTorch) |
| 🔌 **边缘部署版** | `edge_rk3588/` | RK3588 ARM64 Linux | Rockchip RKNN (NPU) |

---

## 🖥️ PC 桌面版（`pc_app/`）

> ✅ **可在普通 PC 上直接运行，支持 GPU/CPU 推理。**

### 环境要求

- Python 3.8+
- Windows / Linux / macOS
- 推荐 NVIDIA GPU（CPU 也可运行）

### 安装

```bash
cd SmartHelmet-Detection
pip install -r requirements-pc.txt
```

### 运行

```bash
cd pc_app/src
python app.py
```

### GUI 功能

- 📷 **打开图片** — 选择图片进行目标检测
- 📁 **打开文件夹** — 批量检测文件夹内所有图片
- 🎬 **打开视频** — 对视频文件逐帧分析
- 📹 **打开摄像头** — 实时摄像头检测
- 💾 **保存结果** — 将检测结果导出为 CSV
- ⚙️ **阈值调节** — 可调节置信度和 IoU 阈值

### 模型说明

默认使用 `yolov8n.pt`（COCO 预训练模型）作为演示。
**若要进行头盔佩戴检测**，请将自定义训练的头盔检测 `.pt` 模型放入 `pc_app/model/` 目录，
并修改 `app.py` 中 `CLASS_NAMES` 字典和模型路径。

---

## 🔌 边缘部署版（`edge_rk3588/`）

> ⚠️ **必须在搭载 RK3588 NPU 的 ARM64 Linux 设备上运行。**

### 硬件要求

- **开发板**: Orange Pi 5 / Rock 5 / 其他搭载 RK3588 的设备
- **操作系统**: Linux (Debian/Ubuntu), **ARM64 架构**
- **NPU 驱动**: 已安装 RKNN SDK (rknnlite)

### 安装

```bash
# 1. 系统依赖
sudo apt update
sudo apt install python3-pip python3-opencv libsdl2-mixer-2.0-0

# 2. Python 依赖
cd SmartHelmet-Detection
pip install -r requirements-edge.txt

# 3. RKNN SDK（从瑞芯微官方获取）
# 下载地址: https://github.com/airockchip/rknn-toolkit2
```

### 运行

```bash
# GUI 界面
cd edge_rk3588/src
python ui.py

# 命令行 — 批量图片检测
python main.py --model_path ../model/yolo11_best.rknn --img_folder ../image

# 命令行 — 视频检测（含 ByteTrack 跟踪）
python new_vedio.py --model_path ../model/yolo11_best.rknn --input_stream ../image/test.mp4
```

---

## 项目结构

```
SmartHelmet-Detection/
├── README.md
├── requirements-pc.txt          # PC 桌面版依赖
├── requirements-edge.txt        # 边缘部署版依赖
├── .gitignore
│
├── pc_app/                      # 🖥️ PC 桌面版
│   ├── src/
│   │   └── app.py               # PyQt5 GUI 主程序
│   ├── model/
│   │   └── yolov8n.pt           # PyTorch 模型（可替换为自定义头盔模型）
│   └── test_data/               # 测试图片
│
└── edge_rk3588/                 # 🔌 边缘部署版
    ├── src/
    │   ├── main.py              # CLI 批量图片检测入口
    │   ├── ui.py                # PySide6 GUI 主程序
    │   ├── image_detect.py      # 图片检测模块
    │   ├── video_detect.py      # 视频检测模块
    │   ├── new_vedio.py         # 视频检测 + ByteTrack 跟踪
    │   ├── rknn_executor.py     # RKNN 模型封装
    │   ├── dataset_utils.py     # 图像预处理
    │   ├── waring.py            # 语音警报模块
    │   ├── test.py              # 警报测试
    │   ├── alarm.wav            # 警报音频
    │   ├── jingao_3d.wav        # 3D 警报音频
    │   └── track/               # ByteTrack 多目标跟踪
    ├── model/                   # RKNN 模型文件
    │   ├── yolo11_best.rknn
    │   └── yolo11_new.rknn
    └── image/                   # 测试图片/视频
```

## 模型说明

### PC 版模型
- `yolov8n.pt` — Ultralytics YOLOv8n 标准模型（COCO 80类预训练）
- 可替换为自定义头盔检测 `.pt` 模型

### 边缘版模型
- `yolo11_best.rknn` — 推荐使用的头盔检测模型（RK3588 NPU 编译）
- `yolo11_new.rknn` — 更新版本

模型转换流程：**PyTorch (.pt) → ONNX → RKNN (.rknn)**
（使用瑞芯微 `rknn-toolkit2` 在 x86 Linux 上完成转换）

## 技术栈

- **目标检测**: YOLOv11 / YOLOv8
- **PC 推理框架**: Ultralytics YOLO (PyTorch)
- **边缘推理框架**: Rockchip RKNN SDK (rknnlite)
- **目标跟踪**: ByteTrack
- **GUI (PC 版)**: PyQt5
- **GUI (边缘版)**: PySide6 (Qt for Python)
- **音频**: Pygame
- **图像处理**: OpenCV, NumPy

## 常见问题

### Q: PC 版检测不到头盔？
A: 默认使用 COCO 预训练模型，需要替换为自定义训练的头盔检测模型才能识别头盔类别。

### Q: import rknnlite 失败？
A: 需要在 RK3588 设备上安装 RKNN SDK，`rknnlite` 不通过 pip 分发。

### Q: PC 版运行报错 CUDA out of memory？
A: 降低输入图片分辨率，或在代码中指定 `device='cpu'`。

## 许可

本项目为竞赛作品，仅供学习和研究使用。
