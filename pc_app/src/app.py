#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智行护盔 — PC 桌面版（Windows / Linux / macOS）
基于 Ultralytics YOLO 的头盔佩戴智能检测预警系统

运行方式:
    python app.py

依赖安装:
    pip install -r ../requirements-pc.txt
"""

import sys, os, time, glob
import cv2, numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QFrame,
    QFileDialog, QMessageBox, QSlider, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QStatusBar, QProgressBar, QSplitter, QScrollArea,
    QComboBox, QCheckBox, QLineEdit
)
from PyQt5.QtGui import (
    QPixmap, QImage, QFont, QColor, QPalette, QIcon,
    QLinearGradient, QBrush, QPainter, QPen
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize

from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════════
#  全局样式常量
# ═══════════════════════════════════════════════════════════════

# 主题色
COLOR_PRIMARY    = "#2563EB"   # 主蓝色
COLOR_PRIMARY_DK = "#1D4ED8"   # 深蓝
COLOR_ACCENT     = "#F59E0B"   # 强调琥珀色
COLOR_SUCCESS    = "#10B981"   # 绿色
COLOR_DANGER     = "#EF4444"   # 红色
COLOR_WARNING    = "#F59E0B"   # 黄色
COLOR_BG         = "#F1F5F9"   # 页面背景
COLOR_SURFACE    = "#FFFFFF"   # 卡片背景
COLOR_TEXT       = "#1E293B"   # 文字
COLOR_TEXT_SEC   = "#64748B"   # 次要文字
COLOR_BORDER     = "#E2E8F0"   # 边框

# 检测框颜色 (BGR)
BOX_COLORS = [
    (56, 189, 248),    # 天蓝 - 类别0
    (34, 197, 94),     # 翠绿 - 类别1
    (239, 68, 68),     # 红色 - 类别2
    (168, 85, 247),    # 紫色 - 类别3
    (251, 146, 60),    # 橙色 - 类别4
    (236, 72, 153),    # 粉色 - 类别5
]

# 类别名称（默认 COCO，可按需替换为头盔检测类别）
CLASS_NAMES = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    4:  "airplane",
    5:  "bus",
    6:  "train",
    7:  "truck",
    8:  "boat",
    9:  "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

# 头盔检测模型的类别（替换自定义模型后启用）
HELMET_CLASSES = {
    0: "两轮车 (two_wheeler)",
    1: "戴头盔 (helmet)",
    2: "未戴头盔 (without_helmet)",
}


# ═══════════════════════════════════════════════════════════════
#  视频处理线程
# ═══════════════════════════════════════════════════════════════

class VideoProcessThread(QThread):
    """在后台线程中处理视频/摄像头，避免阻塞 UI"""
    frame_ready = pyqtSignal(np.ndarray, list, float)  # frame, detections, fps
    finished = pyqtSignal()

    def __init__(self, source, model, conf=0.25, iou=0.7):
        super().__init__()
        self.source = source       # 文件路径或摄像头索引
        self.model = model
        self.conf = conf
        self.iou = iou
        self._running = True

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.finished.emit()
            return

        prev_time = time.time()
        while self._running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            detections = []
            if self.model is not None:
                try:
                    results = self.model(frame, conf=self.conf, iou=self.iou,
                                         verbose=False)
                    for r in results:
                        if r.boxes is not None:
                            boxes = r.boxes.data.cpu().numpy()
                            for box in boxes:
                                x1, y1, x2, y2 = map(int, box[:4])
                                conf = float(box[4])
                                cls_id = int(box[5])
                                detections.append((x1, y1, x2, y2, conf, cls_id))
                except Exception:
                    pass

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 0.001)
            prev_time = curr_time

            self.frame_ready.emit(frame, detections, fps)

        cap.release()
        self.finished.emit()

    def stop(self):
        self._running = False


# ═══════════════════════════════════════════════════════════════
#  主窗口
# ═══════════════════════════════════════════════════════════════

class HelmetDetectionApp(QMainWindow):
    """智行护盔 — 头盔佩戴检测系统"""

    def __init__(self):
        super().__init__()
        self.model = None
        self.model_path = None
        self.video_thread = None
        self.detection_history = []
        self.current_image = None
        self.is_helmet_model = False

        self.init_ui()
        self.apply_stylesheet()
        self.init_model()  # 必须在 init_ui 之后，因为要更新UI控件

    # ── 模型初始化 ──────────────────────────────────────────

    def init_model(self):
        """尝试加载默认模型"""
        default_model = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "model", "yolov8n.pt"
        )
        if os.path.exists(default_model):
            self.load_model_file(default_model)

    def load_model_file(self, path):
        """加载指定的模型文件"""
        try:
            self.model = YOLO(path)
            self.model_path = path
            model_name = os.path.basename(path)
            self.lbl_model_name.setText(f"📦 {model_name}")

            # 检测是否为头盔模型（3类）
            try:
                nc = getattr(self.model.model, 'nc', 80)
                self.is_helmet_model = (nc == 3)
            except Exception:
                self.is_helmet_model = False

            tag = "🪖 头盔检测" if self.is_helmet_model else "📷 COCO 通用"
            self.lbl_model_type.setText(tag)
            self.status_bar.showMessage(f"✅ 模型加载成功: {model_name}", 5000)
            return True
        except Exception as e:
            QMessageBox.critical(self, "模型加载失败", str(e))
            self.status_bar.showMessage("❌ 模型加载失败", 5000)
            return False

    # ── UI 初始化 ────────────────────────────────────────────

    def init_ui(self):
        """构建完整用户界面"""
        self.setWindowTitle("智行护盔 — 头盔佩戴智能检测预警系统")
        self.setMinimumSize(1400, 850)
        self.resize(1600, 950)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("🚀 就绪 — 请加载图片/视频或打开摄像头")

        # 中央部件
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ── 左侧：图像显示区 ──
        left_panel = QWidget()
        left_panel.setObjectName("leftPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 标题栏
        header = QWidget()
        header.setObjectName("headerBar")
        hh = QHBoxLayout(header)
        hh.setContentsMargins(16, 12, 16, 12)
        title = QLabel("🪖  智行护盔 · 智能检测预警系统")
        title.setObjectName("appTitle")
        hh.addWidget(title)
        hh.addStretch()
        self.lbl_status = QLabel("⚪ 待机中")
        self.lbl_status.setObjectName("statusBadge")
        hh.addWidget(self.lbl_status)
        left_layout.addWidget(header)

        # 图像显示
        self.img_display = QLabel()
        self.img_display.setObjectName("imageDisplay")
        self.img_display.setAlignment(Qt.AlignCenter)
        self.img_display.setMinimumSize(800, 500)
        self.img_display.setText(
            "<div style='color:#94A3B8;font-size:18px;'>"
            "📂 点击右侧按钮选择图片/视频<br>或打开摄像头开始检测"
            "</div>"
        )
        left_layout.addWidget(self.img_display, 1)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progressBar")
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        root.addWidget(left_panel, 3)

        # ── 右侧：控制面板 ──
        right_scroll = QScrollArea()
        right_scroll.setObjectName("rightScroll")
        right_scroll.setWidgetResizable(True)
        right_scroll.setFixedWidth(380)

        right_panel = QWidget()
        right_panel.setObjectName("rightPanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(14)
        right_layout.setContentsMargins(16, 16, 16, 16)

        # 卡片1: 模型管理
        card_model, c1 = self._make_card("🧠 模型管理")
        cm = QVBoxLayout(c1)
        cm.setSpacing(8)

        self.lbl_model_name = QLabel("📦 未加载模型")
        self.lbl_model_name.setObjectName("modelNameLabel")
        self.lbl_model_name.setWordWrap(True)
        cm.addWidget(self.lbl_model_name)

        row_model = QHBoxLayout()
        self.lbl_model_type = QLabel("—")
        self.lbl_model_type.setObjectName("modelTypeLabel")
        row_model.addWidget(self.lbl_model_type, 1)

        btn_load = self._make_btn("📂 选择模型", COLOR_PRIMARY)
        btn_load.clicked.connect(self.on_select_model)
        row_model.addWidget(btn_load)
        cm.addLayout(row_model)
        right_layout.addWidget(card_model)

        # 卡片2: 检测设置
        card_settings, c2 = self._make_card("⚙️ 检测参数")
        cs = QGridLayout(c2)
        cs.setSpacing(8)

        cs.addWidget(QLabel("置信度阈值"), 0, 0)
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.25)
        self.spin_conf.setDecimals(2)
        cs.addWidget(self.spin_conf, 0, 1)

        cs.addWidget(QLabel("IoU 阈值"), 1, 0)
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.01, 1.0)
        self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setValue(0.45)
        self.spin_iou.setDecimals(2)
        cs.addWidget(self.spin_iou, 1, 1)

        self.chk_show_labels = QCheckBox("显示标签与置信度")
        self.chk_show_labels.setChecked(True)
        cs.addWidget(self.chk_show_labels, 2, 0, 1, 2)

        right_layout.addWidget(card_settings)

        # 卡片3: 实时统计
        card_stats, c3 = self._make_card("📊 实时统计")
        cst = QGridLayout(c3)
        cst.setSpacing(8)

        for label, key, row in [("目标总数", "count", 0), ("FPS", "fps", 1),
                                ("推理耗时", "infer", 2)]:
            cst.addWidget(QLabel(label), row, 0)

        self.lbl_count = QLabel("0")
        self.lbl_count.setObjectName("statValue")
        cst.addWidget(self.lbl_count, 0, 1)

        self.lbl_fps = QLabel("—")
        self.lbl_fps.setObjectName("statValue")
        cst.addWidget(self.lbl_fps, 1, 1)

        self.lbl_infer_time = QLabel("—")
        self.lbl_infer_time.setObjectName("statValue")
        cst.addWidget(self.lbl_infer_time, 2, 1)

        right_layout.addWidget(card_stats)

        # 卡片4: 检测发现
        card_findings, c4 = self._make_card("🔍 最近检测")
        cf = QVBoxLayout(c4)
        cf.setSpacing(4)
        self.lbl_findings = QLabel("等待检测…")
        self.lbl_findings.setObjectName("findingsLabel")
        self.lbl_findings.setWordWrap(True)
        self.lbl_findings.setMinimumHeight(60)
        cf.addWidget(self.lbl_findings)
        right_layout.addWidget(card_findings)

        # 卡片5: 操作按钮
        card_actions, c5 = self._make_card("🎬 操作")
        ca = QVBoxLayout(c5)
        ca.setSpacing(8)

        btn_image = self._make_btn("🖼️  打开图片", "#6366F1")
        btn_image.clicked.connect(self.on_open_image)
        ca.addWidget(btn_image)

        btn_folder = self._make_btn("📁  批量处理", "#8B5CF6")
        btn_folder.clicked.connect(self.on_open_folder)
        ca.addWidget(btn_folder)

        btn_video = self._make_btn("🎬  打开视频", "#0EA5E9")
        btn_video.clicked.connect(self.on_open_video)
        ca.addWidget(btn_video)

        btn_camera = self._make_btn("📹  打开摄像头", COLOR_SUCCESS)
        btn_camera.clicked.connect(self.on_open_camera)
        ca.addWidget(btn_camera)

        self.btn_stop = self._make_btn("⏹  停止", COLOR_DANGER)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_stop.setEnabled(False)
        ca.addWidget(self.btn_stop)

        btn_clear = self._make_btn("🗑️  清除结果", "#64748B")
        btn_clear.clicked.connect(self.on_clear)
        ca.addWidget(btn_clear)

        btn_save = self._make_btn("💾  导出 CSV", COLOR_ACCENT)
        btn_save.clicked.connect(self.on_save_csv)
        ca.addWidget(btn_save)

        right_layout.addWidget(card_actions)

        right_layout.addStretch()
        right_scroll.setWidget(right_panel)
        root.addWidget(right_scroll)

        # ── 底部：检测表格 ──
        # (通过 central 的垂直布局嵌套)
        # 实际放在左侧面板底部

    # ── 卡片工厂 ────────────────────────────────────────────

    def _make_card(self, title: str):
        """创建统一样式的卡片，返回 (card_frame, content_layout)"""
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 12, 14, 12)
        card_layout.setSpacing(8)

        lbl = QLabel(title)
        lbl.setObjectName("cardTitle")
        card_layout.addWidget(lbl)

        content = QWidget()
        content.setObjectName("cardContent")
        card_layout.addWidget(content)

        return card, content

    def _make_btn(self, text: str, color: str) -> QPushButton:
        """创建统一样式的按钮"""
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setMinimumHeight(42)
        btn.setProperty("btnColor", color)
        return btn

    # ── 全局样式 ────────────────────────────────────────────

    def apply_stylesheet(self):
        """应用 QSS 样式表"""
        self.setStyleSheet(f"""

            QMainWindow {{
                background-color: {COLOR_BG};
            }}

            /* 左侧面板 */
            QWidget#leftPanel {{
                background: {COLOR_SURFACE};
                border-radius: 16px;
                border: 1px solid {COLOR_BORDER};
            }}

            /* 标题栏 */
            QWidget#headerBar {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLOR_PRIMARY}, stop:1 #7C3AED);
                border-top-left-radius: 16px;
                border-top-right-radius: 16px;
            }}

            QLabel#appTitle {{
                color: white;
                font-size: 18px;
                font-weight: bold;
            }}

            QLabel#statusBadge {{
                color: rgba(255,255,255,0.9);
                font-size: 13px;
                background: rgba(255,255,255,0.2);
                border-radius: 12px;
                padding: 4px 14px;
            }}

            /* 图像显示区 */
            QLabel#imageDisplay {{
                background: #0F172A;
                border: 2px dashed #334155;
                border-radius: 8px;
                margin: 8px;
            }}

            /* 右侧面板 */
            QWidget#rightPanel {{
                background: transparent;
            }}

            QScrollArea#rightScroll {{
                border: none;
                background: transparent;
            }}

            /* 卡片 */
            QFrame#card {{
                background: {COLOR_SURFACE};
                border: 1px solid {COLOR_BORDER};
                border-radius: 14px;
            }}

            QLabel#cardTitle {{
                font-size: 15px;
                font-weight: bold;
                color: {COLOR_TEXT};
                padding-bottom: 4px;
            }}

            QWidget#cardContent {{
                background: transparent;
            }}

            /* 模型信息 */
            QLabel#modelNameLabel {{
                font-size: 13px;
                color: {COLOR_PRIMARY};
                font-weight: bold;
                padding: 6px 10px;
                background: #EFF6FF;
                border-radius: 8px;
            }}

            QLabel#modelTypeLabel {{
                font-size: 12px;
                color: {COLOR_TEXT_SEC};
                padding: 4px 0;
            }}

            /* 统计值 */
            QLabel#statValue {{
                font-size: 22px;
                font-weight: bold;
                color: {COLOR_PRIMARY};
            }}

            /* 发现 */
            QLabel#findingsLabel {{
                font-size: 12px;
                color: {COLOR_TEXT};
                background: #F8FAFC;
                border-radius: 8px;
                padding: 8px;
            }}

            /* 按钮基础（颜色由代码逐个设置） */
            QPushButton {{
                color: white;
                border: none;
                border-radius: 10px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: 600;
            }}

            QPushButton:hover {{
                filter: brightness(1.1);
            }}

            QPushButton:pressed {{
                filter: brightness(0.9);
            }}

            QPushButton:disabled {{
                background-color: #CBD5E1 !important;
                color: #94A3B8;
            }}

            /* 输入控件 */
            QDoubleSpinBox, QSpinBox {{
                border: 2px solid {COLOR_BORDER};
                border-radius: 8px;
                padding: 6px 10px;
                font-size: 13px;
                background: white;
            }}

            QDoubleSpinBox:focus, QSpinBox:focus {{
                border-color: {COLOR_PRIMARY};
            }}

            QCheckBox {{
                font-size: 13px;
                spacing: 8px;
            }}

            /* 进度条 */
            QProgressBar {{
                border: none;
                border-radius: 6px;
                background: {COLOR_BORDER};
                height: 6px;
                text-align: center;
                font-size: 11px;
            }}

            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLOR_PRIMARY}, stop:1 #7C3AED);
                border-radius: 6px;
            }}

            /* 状态栏 */
            QStatusBar {{
                background: {COLOR_SURFACE};
                border-top: 1px solid {COLOR_BORDER};
                color: {COLOR_TEXT_SEC};
                font-size: 12px;
            }}

            /* 滚动条 */
            QScrollBar:vertical {{
                border: none;
                background: transparent;
                width: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: #CBD5E1;
                border-radius: 3px;
            }}
        """)

        # 为每个按钮设置独立的颜色变量
        for btn in self.findChildren(QPushButton):
            color = btn.property("btnColor")
            if color:
                btn.setStyleSheet(btn.styleSheet() +
                    f"background-color: {color}; color: white; border: none; "
                    f"border-radius: 10px; padding: 8px 16px; "
                    f"font-size: 13px; font-weight: 600;")

    # ── 事件处理 ────────────────────────────────────────────

    def on_select_model(self):
        """选择模型文件"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO 模型文件", "",
            "模型文件 (*.pt *.pth *.engine *.onnx);;所有文件 (*)"
        )
        if path:
            self.load_model_file(path)

    def on_open_image(self):
        """打开单张图片"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;所有文件 (*)"
        )
        if path:
            self.process_image(path)

    def on_open_folder(self):
        """批量处理文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return

        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp')
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder, '**', ext), recursive=True))
            files.extend(glob.glob(os.path.join(folder, '**', ext.upper()), recursive=True))
        files = sorted(set(files))

        if not files:
            QMessageBox.information(self, "提示", "文件夹中没有找到图片文件")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(files))
        for i, f in enumerate(files):
            self.progress_bar.setValue(i + 1)
            self.process_image(f)
            QApplication.processEvents()
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage(f"✅ 批量处理完成，共 {len(files)} 张图片", 5000)

    def on_open_video(self):
        """打开视频文件"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;所有文件 (*)"
        )
        if path:
            self.start_video_stream(path)

    def on_open_camera(self):
        """打开摄像头"""
        self.start_video_stream(0)  # 默认摄像头

    def on_stop(self):
        """停止视频/摄像头"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait(2000)
            self.video_thread = None
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("⚪ 待机中")
        self.status_bar.showMessage("⏹ 已停止", 3000)

    def on_clear(self):
        """清除检测结果"""
        self.detection_history.clear()
        self.lbl_count.setText("0")
        self.lbl_fps.setText("—")
        self.lbl_infer_time.setText("—")
        self.lbl_findings.setText("等待检测…")
        self.img_display.setText(
            "<div style='color:#94A3B8;font-size:18px;'>"
            "📂 点击右侧按钮选择图片/视频<br>或打开摄像头开始检测"
            "</div>"
        )
        self.status_bar.showMessage("🗑️ 结果已清除", 3000)

    def on_save_csv(self):
        """导出 CSV"""
        if not self.detection_history:
            QMessageBox.information(self, "提示", "没有可保存的检测结果")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "导出 CSV", "detection_results.csv", "CSV 文件 (*.csv)"
        )
        if not path:
            return

        import csv
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(["序号", "来源", "类别", "置信度", "x1", "y1", "x2", "y2"])
            for i, det in enumerate(self.detection_history, 1):
                writer.writerow([
                    i,
                    det.get("source", ""),
                    det.get("class", ""),
                    f"{det.get('conf', 0):.2%}",
                    det.get("x1", ""), det.get("y1", ""),
                    det.get("x2", ""), det.get("y2", ""),
                ])

        self.status_bar.showMessage(f"💾 已导出: {path}", 5000)

    # ── 检测逻辑 ────────────────────────────────────────────

    def process_image(self, path):
        """处理单张图片"""
        if self.model is None:
            QMessageBox.warning(self, "提示", "请先加载模型！")
            return

        img = cv2.imread(path)
        if img is None:
            return

        self.current_image = img.copy()
        conf = self.spin_conf.value()
        iou = self.spin_iou.value()

        t0 = time.time()
        detections = self._run_inference(img, conf, iou)
        elapsed = (time.time() - t0) * 1000

        annotated = self._draw_boxes(img, detections)
        self.lbl_count.setText(str(len(detections)))
        self.lbl_infer_time.setText(f"{elapsed:.0f} ms")
        self.lbl_fps.setText("—")
        self.lbl_status.setText("🟢 检测完成")
        self.status_bar.showMessage(
            f"✅ {os.path.basename(path)} — 检测到 {len(detections)} 个目标 "
            f"({elapsed:.0f}ms)", 5000
        )

        self._update_findings(detections)
        self._display_frame(annotated)
        self._store_history(detections, path)

    def start_video_stream(self, source):
        """启动视频/摄像头流"""
        if self.model is None:
            QMessageBox.warning(self, "提示", "请先加载模型！")
            return

        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait(2000)

        self.lbl_status.setText("🔴 检测中…")
        self.btn_stop.setEnabled(True)
        src_name = "摄像头" if source == 0 else os.path.basename(str(source))
        self.status_bar.showMessage(f"🎬 正在处理: {src_name}")

        self.video_thread = VideoProcessThread(
            source, self.model,
            conf=self.spin_conf.value(),
            iou=self.spin_iou.value()
        )
        self.video_thread.frame_ready.connect(self._on_video_frame)
        self.video_thread.finished.connect(lambda: self.btn_stop.setEnabled(False))
        self.video_thread.finished.connect(lambda: self.lbl_status.setText("⚪ 待机中"))
        self.video_thread.start()

    def _on_video_frame(self, frame, detections, fps):
        """接收视频线程的帧"""
        self.current_image = frame.copy()
        annotated = self._draw_boxes(frame, detections)
        self._display_frame(annotated)
        self.lbl_count.setText(str(len(detections)))
        self.lbl_fps.setText(f"{fps:.1f}")
        self._update_findings(detections[:10])
        self._store_history(detections, "视频流")

    def _run_inference(self, img, conf, iou):
        """执行模型推理，返回标准化检测列表"""
        try:
            results = self.model(img, conf=conf, iou=iou, verbose=False)
        except Exception as e:
            print(f"推理错误: {e}")
            return []

        detections = []
        names = HELMET_CLASSES if self.is_helmet_model else CLASS_NAMES

        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                c = float(box[4])
                cls_id = int(box[5])
                cls_name = names.get(cls_id, f"class_{cls_id}")
                detections.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": c, "class_id": cls_id, "class": cls_name,
                })
        return detections

    def _draw_boxes(self, img, detections):
        """在图像上绘制检测框和标签"""
        img = img.copy()
        show_label = self.chk_show_labels.isChecked()

        for det in detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            cls_id = det["class_id"] % len(BOX_COLORS)
            color = BOX_COLORS[cls_id]

            # 绘制矩形框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            if show_label:
                label = f"{det['class']} {det['conf']:.2f}"
                # 文字背景
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(img, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return img

    def _display_frame(self, frame):
        """将 OpenCV 图像显示到 QLabel"""
        h, w, ch = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # 缩放适配显示区域
        display_size = self.img_display.size()
        scaled = pixmap.scaled(display_size, Qt.KeepAspectRatio,
                               Qt.SmoothTransformation)
        self.img_display.setPixmap(scaled)

    def _update_findings(self, detections):
        """更新最近检测摘要"""
        if not detections:
            self.lbl_findings.setText("未检测到目标")
            return

        # 按类别计数
        from collections import Counter
        counts = Counter(d["class"] for d in detections)
        lines = []
        for cls_name, cnt in counts.most_common(6):
            # 颜色标记
            if "头盔" in cls_name:
                icon = "🟢" if "戴" in cls_name and "未" not in cls_name else \
                       "🔴" if "未" in cls_name else "🟡"
            else:
                icon = "🔵"
            lines.append(f"{icon} {cls_name}: <b>{cnt}</b> 个")

        if len(counts) > 6:
            lines.append(f"… 还有 {len(counts) - 6} 个类别")

        self.lbl_findings.setText("<br>".join(lines))

    def _store_history(self, detections, source):
        """将检测结果存入历史"""
        for d in detections:
            self.detection_history.append({
                "source": os.path.basename(source) if source != "视频流" else source,
                "class": d["class"],
                "conf": d["conf"],
                "x1": d["x1"], "y1": d["y1"],
                "x2": d["x2"], "y2": d["y2"],
            })
        # 只保留最近 5000 条
        if len(self.detection_history) > 5000:
            self.detection_history = self.detection_history[-5000:]

    # ── 关闭事件 ────────────────────────────────────────────

    def closeEvent(self, event):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait(2000)
        event.accept()


# ═══════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import traceback

    # 错误日志文件
    LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'error.log')

    try:
        # Windows 高分屏适配
        if hasattr(Qt, 'AA_EnableHighDpiScaling'):
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

        app = QApplication(sys.argv)
        app.setFont(QFont("Microsoft YaHei", 10))

        window = HelmetDetectionApp()
        window.show()
        sys.exit(app.exec_())

    except Exception as e:
        err_msg = f"[启动错误] {type(e).__name__}: {e}\n{traceback.format_exc()}"
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(err_msg)
        print(err_msg)
        print(f"\n详细错误已写入: {LOG_FILE}")
        input("按 Enter 退出...")
        sys.exit(1)
