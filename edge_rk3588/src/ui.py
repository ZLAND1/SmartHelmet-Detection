import cv2, sys
import numpy as np
from image_detect import start
from video_detect import start_video
from rknn_executor import RKNN_model_container 
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QMessageBox, QFileDialog, QGroupBox
from PySide6.QtCore import Qt, QEvent, QTimer, QThread, Signal

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
 
        self.xiaolian_ui()
        self.model = None
        self.timer = QTimer()
        self.cap = None
        self.result = None
        self.video_thread = None  # 新增视频线程引用
 
    def xiaolian_ui(self):
        self.setFixedSize(1200, 630)
        self.setWindowTitle('@author：Further')
        self.move(400, 300)
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)
        mainlayout = QVBoxLayout(centralWidget)
        layout = QHBoxLayout()
        
        # 左侧显示原始视频
        self.label1 = QLabel(self)
        self.label1.setMinimumSize(580, 550)
        self.label1.setStyleSheet('border:3px solid #6950a1; background-color: black;')
        self.label1.setAlignment(Qt.AlignCenter)
        self.label1.setText("原始视频")
        layout.addWidget(self.label1)
        
        # 右侧显示处理后的视频
        self.label2 = QLabel(self)
        self.label2.setMinimumSize(580, 550)
        self.label2.setStyleSheet('border:3px solid #6950a1; background-color: black;')
        self.label2.setAlignment(Qt.AlignCenter)
        self.label2.setText("处理后视频")
        layout.addWidget(self.label2)
        
        mainlayout.addLayout(layout)
        
        # 界面下半部分： 输出框 和 按钮
        groupbox = QGroupBox(self)
        bottomlayout = QVBoxLayout(groupbox)
        mainlayout.addWidget(groupbox)
 
        hbox_buttons = QHBoxLayout()
        self.load_model_btn = QPushButton('👆选择模型')
        self.load_model_btn.setFixedSize(120, 30)
        self.load_model_btn.clicked.connect(self.load_model)
        hbox_buttons.addWidget(self.load_model_btn)

        self.img_detect_btn = QPushButton('🖼️选择图片')
        self.img_detect_btn.setFixedSize(120, 30)
        self.img_detect_btn.setEnabled(False)
        self.img_detect_btn.clicked.connect(self.select_image)
        hbox_buttons.addWidget(self.img_detect_btn)
        
        # 视频按钮
        self.video_detect_btn = QPushButton('📹选择视频')  
        self.video_detect_btn.setFixedSize(120, 30)
        self.video_detect_btn.setEnabled(False)
        self.video_detect_btn.clicked.connect(self.select_video)  # 绑定新方法
        hbox_buttons.addWidget(self.video_detect_btn)

        # self.display_btn = QPushButton("🔍显示检测物体")
        # self.display_btn.clicked.connect(self.show_detected_objects)
        # self.display_btn.setEnabled(False)
        # self.display_btn.setFixedSize(120, 30)
        # hbox_buttons.addWidget(self.display_btn)

        self.stop_detect_btn = QPushButton('🛑停止')
        self.stop_detect_btn.setFixedSize(120, 30)
        self.stop_detect_btn.setEnabled(False)
        self.stop_detect_btn.clicked.connect(self.stop_detect)
        hbox_buttons.addWidget(self.stop_detect_btn)

        self.exit_btn = QPushButton('❌退出')
        self.exit_btn.setFixedSize(120, 30)
        self.exit_btn.clicked.connect(self.close)
        hbox_buttons.addWidget(self.exit_btn)
        
        bottomlayout.addLayout(hbox_buttons)
 
    # 加载模型
    def load_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", filter='*.rknn')
        platform = 'rk3588'
        if model_path:
            self.model = RKNN_model_container(model_path, 0)
        else:
            print("请重选模型！")
        print('Model-{} is {} model, starting val'.format(model_path, platform))
        # 模型加载成功后启用图片按钮
        self.img_detect_btn.setEnabled(True)
        # 模型加载成功后启用视频按钮
        self.video_detect_btn.setEnabled(True)
        self.stop_detect_btn.setEnabled(True)
 
    def select_image(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        image_path, fileType = QFileDialog.getOpenFileName(self, "选择图片文件", filter='*.jpg *.png *.bmp')
        if image_path:
            img = cv2.imread(image_path)
            self.detect_image(img, image_path)

    def select_video(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        try:
            # 验证视频文件有效性
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                QMessageBox.warning(self, "错误", "无法打开视频文件")
                return
            cap.release()

            # 更新界面状态
            self.video_detect_btn.setEnabled(False)
            self.stop_detect_btn.setEnabled(True)
            # self.display_btn.setEnabled(False)
            self.statusBar().showMessage("正在处理视频...")

            # 启动视频处理线程
            self.video_thread = VideoThread(video_path, self.model)
            self.video_thread.original_frame_ready.connect(self.update_original_frame)
            self.video_thread.processed_frame_ready.connect(self.update_processed_frame)
            self.video_thread.finished.connect(self.on_video_finished)
            self.video_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"视频处理失败: {str(e)}")
            self.reset_video_ui()

    def update_original_frame(self, qimage):
        """更新原始视频帧显示"""
        pixmap = QPixmap.fromImage(qimage)
        self.label1.setPixmap(pixmap.scaled(
            self.label1.size(), 
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def update_processed_frame(self, qimage):
        """更新处理后的视频帧显示"""
        pixmap = QPixmap.fromImage(qimage)
        self.label2.setPixmap(pixmap.scaled(
            self.label2.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        self.result = qimage  # 存储结果用于显示检测物体

    def on_video_finished(self):
        """视频处理完成回调"""
        self.reset_video_ui()
        self.statusBar().showMessage("视频处理完成", 3000)
        QMessageBox.information(self, "完成", "视频分析已完成！") 

    def reset_video_ui(self):
        """重置视频相关UI状态"""
        self.video_detect_btn.setEnabled(True)
        self.stop_detect_btn.setEnabled(False)
        # self.display_btn.setEnabled(True)
        self.statusBar().clearMessage()   
 #后面修改
    def show_detected_objects(self):
        frame = self.result
        if frame:
            det_info = []
            # 注意：这里需要根据实际的results格式进行调整
            # 假设frame是处理后的图像，实际的检测信息可能需要从其他地方获取
            # 由于原始代码中没有明确的获取检测信息的方法，这里提供一个示例
            try:
                if hasattr(frame, 'boxes') and hasattr(frame, 'names'):
                    class_ids = frame.boxes.cls
                    class_names_dict = frame.names
                    for class_id in class_ids:
                        class_name = class_names_dict[int(class_id)]
                        det_info.append(class_name)
                if det_info:
                    object_count = len(det_info)
                    object_info = f"识别到的物体总个数：{object_count}\n"
                    object_dict = {}
                    for obj in det_info:
                        if obj in object_dict:
                            object_dict[obj] += 1
                        else:
                            object_dict[obj] = 1
                    sorted_objects = sorted(object_dict.items(), key=lambda x: x[1], reverse=True)
     
                    for obj_name, obj_count in sorted_objects:
                        object_info += f"{obj_name}: {obj_count}\n"
                    self.show_message_box("识别结果", object_info)
                else:
                    self.show_message_box("识别结果", "未检测到物体")
            except Exception as e:
                self.show_message_box("识别结果", f"无法解析检测结果: {str(e)}")
        else:
            self.show_message_box("识别结果", "没有可用的检测结果")

 
    def stop_detect(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.result = None
        img = cv2.cvtColor(np.zeros((580, 550), np.uint8), cv2.COLOR_BGR2RGB)
        img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        self.label1.setPixmap(QPixmap.fromImage(img))
        self.label2.setPixmap(QPixmap.fromImage(img))
        # self.display_btn.setEnabled(False)
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.quit()
            self.video_thread.wait()
        self.reset_video_ui()
 
    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.timer.isActive():
            self.timer.stop()
        exit()
 
    def detect_image(self, img, image_path):
        if self.model is not None:
            frame = img            
            # Use model for inference
            results = start(image_path, self.model)
            
            # 左侧显示原始图像
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height1, width1, channel1 = frame.shape
            bytesPerLine1 = 3 * width1
            qimage1 = QImage(image_rgb.data, width1, height1, bytesPerLine1, QImage.Format_RGB888)
            pixmap1 = QPixmap.fromImage(qimage1)
            self.label1.setPixmap(pixmap1.scaled(self.label1.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            # 右侧显示处理后的图像
            annotated_image = results
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)  # 转换为 RGB
            height2, width2, channel2 = annotated_image.shape
            bytesPerLine2 = 3 * width2
            qimage2 = QImage(annotated_image.data, width2, height2, bytesPerLine2, QImage.Format_RGB888)
            pixmap2 = QPixmap.fromImage(qimage2)
            self.label2.setPixmap(pixmap2.scaled(self.label2.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            self.result = results  # 存储结果用于显示检测物体
            # self.display_btn.setEnabled(True)
 
    def show_message_box(self, title, message):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec()
 
    def closeEvent(self, event: QEvent):
        # 关闭应用时清除缓存
        if self.cap:
            self.cap.release()
        self.timer.stop()
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.quit()
            self.video_thread.wait()
        event.accept()

class VideoThread(QThread):
    original_frame_ready = Signal(QImage)
    processed_frame_ready = Signal(QImage)

    def __init__(self, video_path, model):
        super().__init__()
        self.video_path = video_path
        self.model = model
        self._is_running = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        flag=False
        try:
            while self._is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 保存原始帧
                rgb_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_original.shape
                bytes_per_line = ch * w
                qimage_original = QImage(rgb_original.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.original_frame_ready.emit(qimage_original)

                # 使用RKNN模型处理帧
                processed_frame,flag = start_video(frame, self.model,flag)
                
                # 转换处理后的帧为QImage
                rgb_processed = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_processed.shape
                bytes_per_line = ch * w
                qimage_processed = QImage(rgb_processed.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.processed_frame_ready.emit(qimage_processed)

                # 控制处理速度
                self.msleep(30)  # 约30FPS
        finally:
            cap.release()

    def stop(self):
        self._is_running = False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())