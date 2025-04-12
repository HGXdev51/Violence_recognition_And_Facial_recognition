import sys
import json
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                           QStackedWidget, QSlider, QComboBox, QFileDialog,
                           QMessageBox, QListWidget, QFrame)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QLinearGradient
from detection_utils import ViolenceDetector, FaceDetector
import os
from playsound import playsound
import threading
import datetime
from PIL import Image, ImageDraw, ImageFont

class LoginWindow(QWidget):
    def __init__(self, app_instance):
        super().__init__()
        self.app = app_instance  # 保存App实例的引用
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('校园欺凌终结者')
        self.setFixedSize(1600, 1200)  # 调整窗口尺寸
        
        # 创建渐变背景
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 #1a2a6c, stop:1 #b21f1f);
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 25px 50px;
                border-radius: 14px;
                font-size: 38px;
                font-family: 'Microsoft YaHei';
                min-height: 90px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QLineEdit {
                padding: 25px;
                border: 3px solid rgba(255, 255, 255, 0.3);
                border-radius: 14px;
                background: rgba(255, 255, 255, 0.1);
                color: white;
                font-family: 'Microsoft YaHei';
                min-height: 95px;
                font-size: 42px;
            }
            QLineEdit:focus {
                border: 3px solid #4CAF50;
                background: rgba(255, 255, 255, 0.15);
            }
            QLineEdit::placeholder {
                color: rgba(255, 255, 255, 0.5);
                font-size: 42px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(60)  # 调整组件间距
        layout.setContentsMargins(180, 200, 180, 200)  # 调整边距
        
        # 标题容器
        title_container = QWidget()
        title_container.setFixedHeight(250)  # 调整标题容器高度
        title_container.setStyleSheet("background: transparent;")  # 设置透明背景
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        
        # 标题
        title = QLabel("校园欺凌终结者")
        title.setStyleSheet("""
            color: white; 
            font-size: 72px; 
            font-weight: bold;
            font-family: 'Microsoft YaHei';
            padding: 40px;
            letter-spacing: 5px;
        """)
        title.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title)
        
        layout.addWidget(title_container)
        
        # 用户名输入
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("用户名")
        self.username_input.setMinimumHeight(95)  # 调整输入框高度
        layout.addWidget(self.username_input)
        
        # 密码输入
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("密码")
        self.password_input.setMinimumHeight(95)  # 调整输入框高度
        layout.addWidget(self.password_input)
        
        # 登录按钮
        self.login_button = QPushButton("登录")
        self.login_button.clicked.connect(self.check_login)
        self.login_button.setMinimumHeight(95)  # 调整按钮高度
        layout.addWidget(self.login_button)
        
        # 添加制作信息
        author_label = QLabel("由黄国轩制作")
        author_label.setStyleSheet("""
            color: rgba(255, 255, 255, 0.7);
            font-size: 38px;
            font-family: 'Microsoft YaHei';
            margin-top: 40px;
        """)
        author_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(author_label)
        
        self.setLayout(layout)
        
    def check_login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        
        if not username or not password:
            QMessageBox.warning(self, "错误", "用户名和密码不能为空")
            return
            
        try:
            # 确保user.json文件存在
            if not os.path.exists('user.json'):
                # 创建默认用户
                default_user = {
                    "users": [
                        {
                            "username": "admin",
                            "password": "hgx0501",
                            "is_admin": True
                        }
                    ]
                }
                with open('user.json', 'w') as f:
                    json.dump(default_user, f, indent=4)
                    
            # 读取用户数据
            with open('user.json', 'r') as f:
                users_data = json.load(f)
                if 'users' not in users_data:
                    QMessageBox.warning(self, "错误", "用户数据格式错误")
                    return
                    
                for user in users_data['users']:
                    if user['username'] == username and user['password'] == password:
                        self.app.show_main_window()  # 使用保存的app实例
                        return
                        
            # 如果循环结束还没有返回，说明验证失败
            QMessageBox.warning(self, "错误", "用户名或密码错误")
            self.username_input.setText("")
            self.password_input.setText("")
            
        except json.JSONDecodeError:
            QMessageBox.warning(self, "错误", "用户数据文件格式错误")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"发生错误: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.violence_detector = ViolenceDetector()
        self.face_detector = FaceDetector()
        self.initUI()
        # 添加框选相关变量
        self.is_selecting = False
        self.selection_start = None
        self.selection_end = None
        self.current_selection = None
        self.face_display.mousePressEvent = self.mousePressEvent
        self.face_display.mouseMoveEvent = self.mouseMoveEvent
        self.face_display.mouseReleaseEvent = self.mouseReleaseEvent
        
        # 添加警告声音相关变量
        self.warning_thread = None
        self.is_warning = False
        self.warning_played = False  # 添加标志位
        self.setFocusPolicy(Qt.StrongFocus)  # 确保窗口可以接收键盘事件
        
    def initUI(self):
        self.setWindowTitle('校园欺凌终结者')
        self.showMaximized()
        
        # 设置全局样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QWidget {
                color: #ffffff;
                font-family: 'Microsoft YaHei';
                font-size: 24px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 8px;
                font-size: 24px;
                transition: background-color 0.3s;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1c6ea4;
            }
            QLabel {
                color: #ffffff;
                font-size: 24px;
            }
            QListWidget {
                background-color: #2c3e50;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
                font-size: 22px;
            }
            QListWidget::item {
                padding: 12px;
                border-bottom: 1px solid #34495e;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                border-radius: 4px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #34495e;
                height: 12px;
                background: #2c3e50;
                border-radius: 6px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: none;
                width: 24px;
                margin: -6px 0;
                border-radius: 12px;
            }
            QComboBox {
                background-color: #2c3e50;
                color: white;
                border: 1px solid #34495e;
                border-radius: 8px;
                padding: 10px;
                font-size: 22px;
                min-height: 40px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 20px;
                height: 20px;
            }
        """)
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(30)
        
        # 创建左侧控制面板
        control_panel = QWidget()
        control_panel.setFixedWidth(400)  # 增加控制面板宽度
        control_panel.setStyleSheet("""
            QWidget {
                background-color: #2c3e50;
                border-radius: 20px;
                padding: 25px;
            }
        """)
        
        control_layout = QVBoxLayout()
        control_layout.setSpacing(25)  # 增加间距
        
        # 添加标题
        title = QLabel("控制面板")
        title.setStyleSheet("font-size: 36px; font-weight: bold; color: #3498db;")
        title.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(title)
        
        # 页面切换按钮
        button_layout = QHBoxLayout()
        self.violence_button = QPushButton("暴力检测")
        self.face_button = QPushButton("人脸识别")
        self.violence_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                min-width: 150px;
                font-size: 26px;
            }
        """)
        self.face_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                min-width: 150px;
                font-size: 26px;
            }
        """)
        self.violence_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        self.face_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        
        button_layout.addWidget(self.violence_button)
        button_layout.addWidget(self.face_button)
        control_layout.addLayout(button_layout)
        
        # 添加分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #34495e;")
        control_layout.addWidget(line)
        
        # 暴力检测相关控件
        violence_controls = QWidget()
        violence_layout = QVBoxLayout()
        violence_layout.setSpacing(10)
        
        camera_label = QLabel("选择摄像头:")
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["摄像头 0", "摄像头 1"])
        
        sensitivity_label = QLabel("检测灵敏度:")
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setMinimum(1)
        self.sensitivity_slider.setMaximum(10)
        self.sensitivity_slider.setValue(5)
        
        self.detect_button = QPushButton("开始检测")
        self.detect_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                min-width: 180px;
                font-size: 26px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        self.detect_button.clicked.connect(self.toggle_detection)
        
        violence_layout.addWidget(camera_label)
        violence_layout.addWidget(self.camera_combo)
        violence_layout.addWidget(sensitivity_label)
        violence_layout.addWidget(self.sensitivity_slider)
        violence_layout.addWidget(self.detect_button)
        violence_controls.setLayout(violence_layout)
        
        # 人脸识别相关控件
        face_controls = QWidget()
        face_layout = QVBoxLayout()
        face_layout.setSpacing(10)
        
        # 显示暴力图片列表
        self.image_list = QListWidget()
        self.image_list.setFixedHeight(200)
        self.refresh_image_list()
        
        # 选择图片按钮
        self.select_image_button = QPushButton("选择图片")
        self.select_image_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                min-width: 180px;
                font-size: 26px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.select_image_button.clicked.connect(self.select_image)
        
        # 识别按钮
        self.recognize_button = QPushButton("开始识别")
        self.recognize_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                min-width: 180px;
                font-size: 26px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.recognize_button.clicked.connect(self.recognize_face)
        
        # 结果显示
        self.result_label = QLabel("识别结果将显示在这里")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 24px;
                padding: 20px;
                background-color: #34495e;
                border-radius: 10px;
            }
        """)
        
        face_layout.addWidget(QLabel("暴力图片列表:"))
        face_layout.addWidget(self.image_list)
        face_layout.addWidget(self.select_image_button)
        face_layout.addWidget(self.recognize_button)
        face_layout.addWidget(self.result_label)
        face_controls.setLayout(face_layout)
        
        # 将控件添加到堆叠布局
        self.controls_stack = QStackedWidget()
        self.controls_stack.addWidget(violence_controls)
        self.controls_stack.addWidget(face_controls)
        
        control_layout.addWidget(self.controls_stack)
        
        # 添加版权信息
        copyright_label = QLabel("Copyright © HGXdev 2025")
        copyright_label.setStyleSheet("color: #7f8c8d; font-size: 20px;")
        copyright_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(copyright_label)
        
        control_panel.setLayout(control_layout)
        main_layout.addWidget(control_panel)
        
        # 创建堆叠窗口部件
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setStyleSheet("""
            QWidget {
                background-color: #1a1a1a;
                border-radius: 15px;
            }
        """)
        
        # 暴力检测页面
        self.violence_display = QLabel()
        self.violence_display.setStyleSheet("background-color: black; border-radius: 15px;")
        self.stacked_widget.addWidget(self.violence_display)
        
        # 人脸识别页面
        self.face_display = QLabel()
        self.face_display.setStyleSheet("background-color: black; border-radius: 15px;")
        self.stacked_widget.addWidget(self.face_display)
        
        main_layout.addWidget(self.stacked_widget)
        main_widget.setLayout(main_layout)
        
        # 初始化变量
        self.cap = None
        self.is_detecting = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_image = None
        
        # 连接页面切换信号
        self.stacked_widget.currentChanged.connect(self.on_page_changed)
        
        # 添加页面切换动画
        self.animation = QPropertyAnimation(self.stacked_widget, b"pos")
        self.animation.setDuration(300)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        
    def on_page_changed(self, index):
        self.controls_stack.setCurrentIndex(index)
        if index == 1:  # 人脸识别页面
            self.refresh_image_list()
            
    def refresh_image_list(self):
        self.image_list.clear()
        if os.path.exists("violent_figures"):
            for filename in os.listdir("violent_figures"):
                if filename.endswith((".jpg", ".png")):
                    self.image_list.addItem(filename)
                    
    def select_image(self):
        if self.image_list.currentItem() is None:
            QMessageBox.warning(self, "警告", "请先选择一张图片")
            return
            
        filename = self.image_list.currentItem().text()
        image_path = os.path.join("violent_figures", filename)
        self.current_image = cv2.imread(image_path)
        if self.current_image is not None:
            self.display_image(self.current_image, self.face_display)
            self.result_label.setText("请在图片上框选人脸区域")
            
    def recognize_face(self):
        """识别选中图片中的人脸"""
        try:
            # 获取选中的图片路径
            selected_items = self.image_list.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "警告", "请先选择一张图片")
                return
                
            # 获取完整的图片路径
            filename = selected_items[0].text()
            image_path = os.path.join("violent_figures", filename)
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"找不到图片文件: {image_path}")
            
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("无法读取图片")
            
            # 转换为RGB格式用于显示
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 转换为PIL图像用于绘制
            pil_image = Image.fromarray(image_rgb)
            draw = ImageDraw.Draw(pil_image)
            
            # 加载中文字体
            try:
                font = ImageFont.truetype("simhei.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # 检测和识别所有人脸
            recognized_faces = self.face_detector.recognize_face(image)
            
            # 用于存储识别结果
            recognition_results = []
            
            # 绘制每个人脸的框和名字
            for (x, y, w, h, name) in recognized_faces:
                # 绘制人脸框
                draw.rectangle([(x, y), (x+w, y+h)], outline="green", width=2)
                
                # 计算文本位置，确保在图像范围内
                text_y = max(0, y - 25)
                # 绘制名字
                draw.text((x, text_y), name, fill="green", font=font)
                
                # 添加到识别结果
                recognition_results.append(f"在位置 ({x}, {y}) 检测到: {name}")
            
            # 将PIL图像转回OpenCV格式
            image_with_boxes = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # 直接在右侧显示结果
            self.display_image(image_with_boxes, self.face_display)
            
            # 更新结果标签
            if recognition_results:
                self.result_label.setText("\n".join(recognition_results))
            else:
                self.result_label.setText("未检测到人脸")
                
        except FileNotFoundError as e:
            self.result_label.setText(f"错误: {str(e)}")
            print(f"错误: {str(e)}")
        except Exception as e:
            self.result_label.setText(f"识别失败: {str(e)}")
            print(f"识别失败: {str(e)}")
            
    def display_image(self, frame, display_widget):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        
        # 获取显示区域的尺寸
        display_rect = display_widget.contentsRect()
        display_width = display_rect.width()
        display_height = display_rect.height()
        
        # 计算缩放后的尺寸，保持宽高比
        display_ratio = display_width / display_height
        image_ratio = w / h
        
        if display_ratio > image_ratio:
            # 高度适配，宽度按比例缩放
            target_height = display_height
            target_width = int(image_ratio * target_height)
        else:
            # 宽度适配，高度按比例缩放
            target_width = display_width
            target_height = int(target_width / image_ratio)
            
        # 缩放图像
        scaled_image = cv2.resize(rgb_image, (target_width, target_height))
        bytes_per_line = ch * target_width
        
        # 创建QImage
        qt_image = QImage(scaled_image.data, target_width, target_height, 
                         bytes_per_line, QImage.Format_RGB888)
        
        # 创建QPixmap并设置到QLabel
        pixmap = QPixmap.fromImage(qt_image)
        display_widget.setPixmap(pixmap)
        display_widget.setAlignment(Qt.AlignCenter)

    def toggle_detection(self):
        if not self.is_detecting:
            self.start_detection()
        else:
            self.stop_detection()
            
    def start_detection(self):
        camera_index = self.camera_combo.currentIndex()
        self.cap = cv2.VideoCapture(camera_index)
        self.is_detecting = True
        self.detect_button.setText("停止检测")
        self.timer.start(30)  # 约30fps
        
    def stop_detection(self):
        self.is_detecting = False
        self.detect_button.setText("开始检测")
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            
    def update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.stacked_widget.currentIndex() == 0:  # 暴力检测页面
                    try:
                        violence_score, boxes, keypoints = self.violence_detector.detect_violence(frame)
                        
                        # 转换为PIL图像用于绘制中文
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        draw = ImageDraw.Draw(pil_image)
                        
                        # 加载中文字体
                        try:
                            font = ImageFont.truetype("simhei.ttf", 32)
                        except:
                            font = ImageFont.load_default()
                        
                        # 保存所有超过阈值的图片
                        if violence_score > 0.5:  # 降低阈值以保存更多图片
                            self.save_violent_frame(frame.copy(), violence_score, boxes, keypoints)
                        
                        # 绘制关键点和骨架（不包含面部关键点）
                        for kpts in keypoints:
                            if len(kpts) < 13:  # 确保有足够的非面部关键点
                                continue
                                
                            # 绘制非面部关键点（不包含耳朵点和1号点）
                            body_keypoints = kpts[1:13]  # 跳过1号点，使用2-13号点
                            for i, (x, y) in enumerate(body_keypoints):
                                if x > 0 and y > 0:
                                    # 绘制关键点
                                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                                    # 显示序号（从2开始）
                                    cv2.putText(frame, str(i+2), (int(x)-5, int(y)-5),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            # 绘制骨架连接（不包含面部连接）
                            skeleton = [(0,2), (2,4), (1,3), (3,5),  # 手臂
                                      (6,8), (8,10), (7,9), (9,11),  # 腿
                                      (0,1), (0,6), (1,7), (6,7)]  # 躯干
                            for i, j in skeleton:
                                if (0 <= i < len(body_keypoints) and 0 <= j < len(body_keypoints) and
                                    body_keypoints[i][0] > 0 and body_keypoints[i][1] > 0 and 
                                    body_keypoints[j][0] > 0 and body_keypoints[j][1] > 0):
                                    cv2.line(frame, 
                                           (int(body_keypoints[i][0]), int(body_keypoints[i][1])),
                                           (int(body_keypoints[j][0]), int(body_keypoints[j][1])),
                                           (0, 255, 0), 2)
                        
                        # 根据暴力倾向值设置颜色
                        for box, conf, cls in boxes:
                            x1, y1, x2, y2 = box
                            color = (0, 255, 0)  # 绿色
                            if violence_score > 0.4:
                                color = (0, 255, 255)  # 黄色
                            if violence_score > 0.5:
                                color = (0, 0, 255)  # 红色
                                self.start_warning()  # 开始循环播放警告声音
                            
                            # 绘制整体人物框
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            
                            # 显示置信度
                            label = f"Person: {conf:.2f}"
                            cv2.putText(frame, label, (int(x1), int(y1)-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # 将frame转换回PIL图像以绘制中文
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        draw = ImageDraw.Draw(pil_image)
                        
                        # 显示总体暴力倾向值
                        draw.text((10, 30), f"Violence Score: {violence_score:.2f}", 
                                fill=(0, 0, 255) if violence_score > 0.7 else (0, 255, 0),
                                font=font)
                        
                        # 显示空格键提示（使用中文）
                        if self.is_warning:
                            draw.text((10, 70), "按空格键停止警告声音", 
                                    fill=(0, 0, 255),
                                    font=font)
                        
                        # 将PIL图像转回OpenCV格式
                        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                        
                    except Exception as e:
                        print(f"Error in update_frame: {str(e)}")
                        # 显示错误信息
                        cv2.putText(frame, "Error in detection", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    self.display_image(frame, self.violence_display)
                    
    def save_violent_frame(self, frame, violence_score, boxes, keypoints):
        """保存暴力图片"""
        try:
            # 确保目录存在
            if not os.path.exists("violent_figures"):
                os.makedirs("violent_figures")
                
            # 在保存的图片上绘制所有内容
            # 绘制关键点和骨架
            for kpts in keypoints:
                if len(kpts) < 13:
                    continue
                    
                body_keypoints = kpts[1:13]  # 跳过1号点
                for i, (x, y) in enumerate(body_keypoints):
                    if x > 0 and y > 0:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                        cv2.putText(frame, str(i+2), (int(x)-5, int(y)-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                skeleton = [(0,2), (2,4), (1,3), (3,5), (6,8), (8,10), (7,9), (9,11),
                          (0,1), (0,6), (1,7), (6,7)]
                for i, j in skeleton:
                    if (0 <= i < len(body_keypoints) and 0 <= j < len(body_keypoints) and
                        body_keypoints[i][0] > 0 and body_keypoints[i][1] > 0 and 
                        body_keypoints[j][0] > 0 and body_keypoints[j][1] > 0):
                        cv2.line(frame, 
                               (int(body_keypoints[i][0]), int(body_keypoints[i][1])),
                               (int(body_keypoints[j][0]), int(body_keypoints[j][1])),
                               (0, 255, 0), 2)
            
            # 绘制框和标签
            for box, conf, cls in boxes:
                x1, y1, x2, y2 = box
                color = (0, 255, 0)
                if violence_score > 0.5:
                    color = (0, 255, 255)
                if violence_score > 0.7:
                    color = (0, 0, 255)
                
                # 绘制整体人物框
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # 显示置信度
                label = f"Person: {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1)-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 显示暴力倾向值
            cv2.putText(frame, f"Violence Score: {violence_score:.2f}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                      (0, 0, 255) if violence_score > 0.7 else (0, 255, 0), 2)
                
            # 生成文件名（使用时间戳）
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"violent_figures/violence_{timestamp}_score_{violence_score:.2f}.jpg"
            
            # 保存图片
            cv2.imwrite(filename, frame)
            print(f"已保存暴力图片: {filename}")
            
            # 刷新图片列表
            self.refresh_image_list()
            
        except Exception as e:
            print(f"保存暴力图片时出错: {str(e)}")

    def play_warning(self):
        """播放警告声音的方法"""
        try:
            while self.is_warning:
                if os.path.exists("warning.mp3"):
                    playsound("warning.mp3")
                    if not self.is_warning:  # 检查是否需要停止
                        break
                else:
                    print("找不到警告音频文件: warning.mp3")
                    self.is_warning = False
                    break
        except Exception as e:
            print(f"播放警告声音出错: {str(e)}")
            self.is_warning = False
        finally:
            self.warning_played = False  # 重置标志位
            
    def start_warning(self):
        """开始播放警告声音"""
        if not self.warning_played:  # 检查是否已经在播放
            self.is_warning = True
            self.warning_played = True  # 设置标志位
            if self.warning_thread is not None and self.warning_thread.is_alive():
                self.warning_thread.join(0)  # 等待之前的线程结束
            self.warning_thread = threading.Thread(target=self.play_warning)
            self.warning_thread.daemon = True
            self.warning_thread.start()
            
    def stop_warning(self):
        """停止播放警告声音"""
        self.is_warning = False
        self.warning_played = False
        if self.warning_thread is not None and self.warning_thread.is_alive():
            self.warning_thread.join(0)  # 等待线程结束
            
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.stop_warning()
            print("警告声音已停止")
            
    def closeEvent(self, event):
        """窗口关闭时的处理"""
        self.stop_warning()  # 确保停止声音播放
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.face_display.underMouse():
            # 获取相对于face_display的坐标
            pos = self.face_display.mapFromParent(event.pos())
            self.is_selecting = True
            self.selection_start = pos
            self.selection_end = pos
            
    def mouseMoveEvent(self, event):
        if self.is_selecting and self.face_display.underMouse():
            # 获取相对于face_display的坐标
            pos = self.face_display.mapFromParent(event.pos())
            self.selection_end = pos
            self.update_selection_display()
            
    def mouseReleaseEvent(self, event):
        if self.is_selecting:
            # 获取相对于face_display的坐标
            pos = self.face_display.mapFromParent(event.pos())
            self.is_selecting = False
            self.selection_end = pos
            self.update_selection_display()
            # 开始识别
            self.recognize_selected_face()
            
    def update_selection_display(self):
        if self.current_image is not None and self.selection_start and self.selection_end:
            # 创建临时图像用于显示
            temp_image = self.current_image.copy()
            # 转换为RGB格式
            temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
            # 转换为PIL图像
            pil_image = Image.fromarray(temp_image)
            draw = ImageDraw.Draw(pil_image, 'RGBA')  # 使用RGBA模式支持透明度
            
            # 获取显示区域和原始图片的尺寸
            display_rect = self.face_display.contentsRect()
            display_width = display_rect.width()
            display_height = display_rect.height()
            image_height, image_width = self.current_image.shape[:2]
            
            # 计算实际显示的图片尺寸（考虑长宽比）
            display_ratio = display_width / display_height
            image_ratio = image_width / image_height
            
            if display_ratio > image_ratio:
                # 高度适配，宽度按比例缩放
                scaled_height = display_height
                scaled_width = int(image_ratio * scaled_height)
                x_offset = (display_width - scaled_width) // 2
                y_offset = 0
            else:
                # 宽度适配，高度按比例缩放
                scaled_width = display_width
                scaled_height = int(scaled_width / image_ratio)
                x_offset = 0
                y_offset = (display_height - scaled_height) // 2
                
            # 计算缩放比例
            scale_x = image_width / scaled_width
            scale_y = image_height / scaled_height
            
            # 将鼠标坐标转换为图片坐标（考虑偏移）
            x1 = int((self.selection_start.x() - x_offset) * scale_x)
            y1 = int((self.selection_start.y() - y_offset) * scale_y)
            x2 = int((self.selection_end.x() - x_offset) * scale_x)
            y2 = int((self.selection_end.y() - y_offset) * scale_y)
            
            # 确保坐标在图片范围内
            x1 = max(0, min(x1, image_width))
            y1 = max(0, min(y1, image_height))
            x2 = max(0, min(x2, image_width))
            y2 = max(0, min(y2, image_height))
            
            # 绘制半透明绿色填充
            draw.rectangle([(x1, y1), (x2, y2)], fill=(0, 255, 0, 64))  # 半透明绿色
            # 绘制绿色边框
            draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=2)
            
            # 转回OpenCV格式并显示
            image_with_selection = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            self.display_image(image_with_selection, self.face_display)
            
    def recognize_selected_face(self):
        """识别用户框选的人脸区域"""
        try:
            if not self.selection_start or not self.selection_end:
                return
                
            # 获取显示区域和原始图片的尺寸
            display_rect = self.face_display.contentsRect()
            display_width = display_rect.width()
            display_height = display_rect.height()
            image_height, image_width = self.current_image.shape[:2]
            
            # 计算实际显示的图片尺寸（考虑长宽比）
            display_ratio = display_width / display_height
            image_ratio = image_width / image_height
            
            if display_ratio > image_ratio:
                scaled_height = display_height
                scaled_width = int(image_ratio * scaled_height)
                x_offset = (display_width - scaled_width) // 2
                y_offset = 0
            else:
                scaled_width = display_width
                scaled_height = int(scaled_width / image_ratio)
                x_offset = 0
                y_offset = (display_height - scaled_height) // 2
                
            # 计算缩放比例
            scale_x = image_width / scaled_width
            scale_y = image_height / scaled_height
            
            # 将鼠标坐标转换为图片坐标（考虑偏移）
            x1 = int((self.selection_start.x() - x_offset) * scale_x)
            y1 = int((self.selection_start.y() - y_offset) * scale_y)
            x2 = int((self.selection_end.x() - x_offset) * scale_x)
            y2 = int((self.selection_end.y() - y_offset) * scale_y)
            
            # 确保坐标在图片范围内
            x1 = max(0, min(x1, image_width))
            y1 = max(0, min(y1, image_height))
            x2 = max(0, min(x2, image_width))
            y2 = max(0, min(y2, image_height))
            
            # 获取框选区域
            selected_region = self.current_image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
            
            # 识别选中的人脸
            recognized_faces = self.face_detector.recognize_face(selected_region)
            
            if recognized_faces:
                # 获取第一个识别结果
                _, _, _, _, name = recognized_faces[0]
                
                # 在原图上绘制结果
                image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                draw = ImageDraw.Draw(pil_image, 'RGBA')  # 使用RGBA模式支持透明度
                
                # 加载中文字体
                try:
                    font = ImageFont.truetype("simhei.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                # 绘制半透明绿色填充
                draw.rectangle([(x1, y1), (x2, y2)], fill=(0, 255, 0, 64))  # 半透明绿色
                # 绘制绿色边框
                draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=2)
                
                # 绘制名字
                text_y = max(0, min(y1, y2) - 25)
                draw.text((min(x1, x2), text_y), name, fill="green", font=font)
                
                # 显示结果
                image_with_result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                self.display_image(image_with_result, self.face_display)
                
                # 更新结果标签
                self.result_label.setText(f"识别结果: {name}")
            else:
                self.result_label.setText("未识别到人脸")
                
        except Exception as e:
            self.result_label.setText(f"识别失败: {str(e)}")
            print(f"识别失败: {str(e)}")

class App(QApplication):
    def __init__(self, sys_argv):
        super().__init__(sys_argv)
        self.login_window = LoginWindow(self)  # 传递self给LoginWindow
        self.main_window = None
        
    def show_main_window(self):
        self.login_window.hide()
        self.main_window = MainWindow()
        self.main_window.show()

if __name__ == '__main__':
    app = App(sys.argv)
    app.login_window.show()
    sys.exit(app.exec_()) 