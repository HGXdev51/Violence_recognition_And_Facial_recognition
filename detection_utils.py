import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

class ViolenceDetector:
    def __init__(self):
        # 加载YOLOv8姿态检测模型
        self.model = YOLO('yolov8n-pose.pt')
        self.violence_threshold = 0.7
        self.frame_count = 0
        self.save_interval = 30
        
        # 定义暴力动作的关键点组合（不包含面部关键点）
        self.violent_poses = {
            'punch': [(5, 6, 7), (6, 7, 8)],  # 右手臂
            'kick': [(11, 12, 13), (12, 13, 14)],  # 右腿
            'strike': [(5, 6, 7), (6, 7, 8), (11, 12, 13), (12, 13, 14)]  # 组合动作
        }
        
        # 定义可能表示暴力的物品
        self.violence_objects = {
            44: 0.8,   # bottle
            45: 0.8,   # wine glass
            46: 0.8,   # cup
            47: 0.8,   # fork
            48: 0.8,   # knife
            49: 0.8,   # spoon
            50: 0.8,   # bowl
            63: 0.9,   # baseball bat
            64: 0.9,   # baseball glove
            65: 0.9,   # skateboard
            66: 0.9,   # surfboard
            67: 0.9,   # tennis racket
        }
        
    def calculate_angle(self, p1, p2, p3):
        """计算三个点形成的角度"""
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
        
    def detect_violence(self, frame):
        try:
            # 进行YOLO姿态检测
            results = self.model(frame, conf=0.3)
            
            violence_score = 0
            boxes = []
            keypoints = []
            
            for result in results:
                # 获取边界框
                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if cls == 0:  # 只处理人物类别
                        boxes.append((box.xyxy[0], conf, cls))
                
                # 获取关键点（跳过面部关键点）
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    for kpts in result.keypoints:
                        if kpts is not None and kpts.xy is not None:
                            kpts_np = kpts.xy[0].cpu().numpy()
                            if len(kpts_np) > 4:  # 确保有足够的非面部关键点
                                # 跳过前4个面部关键点
                                keypoints.append(kpts_np[4:])
            
            # 分析每个检测到的人物的姿态
            for kpts in keypoints:
                if len(kpts) < 13:  # 17-4=13个非面部关键点
                    continue
                    
                person_violence = 0
                
                # 分析暴力动作
                for pose_name, keypoint_groups in self.violent_poses.items():
                    for group in keypoint_groups:
                        try:
                            # 检查关键点是否有效
                            if all(0 <= idx < len(kpts) for idx in group):
                                p1, p2, p3 = kpts[group[0]], kpts[group[1]], kpts[group[2]]
                                if all(p[0] > 0 and p[1] > 0 for p in [p1, p2, p3]):
                                    angle = self.calculate_angle(p1, p2, p3)
                                    
                                    # 根据角度判断动作
                                    if pose_name == 'punch' and 60 < angle < 120:
                                        person_violence += 0.3
                                    elif pose_name == 'kick' and 30 < angle < 90:
                                        person_violence += 0.4
                                    elif pose_name == 'strike' and 45 < angle < 135:
                                        person_violence += 0.5
                        except Exception as e:
                            print(f"Error analyzing pose {pose_name}: {str(e)}")
                            continue
                
                # 分析物体交互
                for box, conf, cls in boxes:
                    if cls in self.violence_objects:
                        try:
                            # 计算物体与人的距离
                            person_center = np.mean(kpts, axis=0)
                            box_center = np.array([(box[0] + box[2])/2, (box[1] + box[3])/2])
                            distance = np.linalg.norm(person_center - box_center)
                            
                            if distance < 100:  # 像素距离阈值
                                person_violence += self.violence_objects[cls] * conf
                        except Exception as e:
                            print(f"Error calculating distance: {str(e)}")
                            continue
                
                # 考虑多人场景
                if len(keypoints) > 1:
                    person_violence = min(1.0, person_violence + 0.2)
                
                violence_score = max(violence_score, person_violence)
            
            return violence_score, boxes, keypoints
            
        except Exception as e:
            print(f"Error in detect_violence: {str(e)}")
            return 0, [], []
    
    def save_violent_frame(self, frame, violence_score):
        if self.frame_count % self.save_interval == 0 and violence_score > self.violence_threshold:
            # 创建帧的副本用于保存
            save_frame = frame.copy()
            
            # 获取检测结果
            results = self.model(save_frame, conf=0.3)
            
            # 在保存的帧上绘制关键点和骨架（不包含面部关键点）
            for result in results:
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    for kpts in result.keypoints:
                        if kpts is not None and kpts.xy is not None:
                            kpts_np = kpts.xy[0].cpu().numpy()
                            if len(kpts_np) > 4:  # 确保有足够的非面部关键点
                                # 跳过前4个面部关键点
                                kpts_np = kpts_np[4:]
                                
                                # 绘制非面部关键点
                                for i, (x, y) in enumerate(kpts_np):
                                    if x > 0 and y > 0:
                                        cv2.circle(save_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                                
                                # 绘制骨架连接（不包含面部连接）
                                skeleton = [(0,2), (2,4), (1,3), (3,5),  # 手臂
                                          (6,8), (8,10), (7,9), (9,11),  # 腿
                                          (0,1), (0,6), (1,7), (6,7)]  # 躯干
                                for i, j in skeleton:
                                    if (0 <= i < len(kpts_np) and 0 <= j < len(kpts_np) and
                                        kpts_np[i][0] > 0 and kpts_np[i][1] > 0 and 
                                        kpts_np[j][0] > 0 and kpts_np[j][1] > 0):
                                        cv2.line(save_frame, 
                                               (int(kpts_np[i][0]), int(kpts_np[i][1])),
                                               (int(kpts_np[j][0]), int(kpts_np[j][1])),
                                               (0, 255, 0), 2)
            
            # 保存处理后的帧
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"violent_figures/violence_{timestamp}_{violence_score:.2f}.jpg"
            os.makedirs("violent_figures", exist_ok=True)
            cv2.imwrite(filename, save_frame)
        self.frame_count += 1

class FaceDetector:
    def __init__(self):
        # 加载OpenCV的人脸检测器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.known_faces = {}
        self.load_known_faces()
        
    def load_known_faces(self):
        """加载known_faces目录下的人脸图片"""
        face_dir = "known_faces"
        os.makedirs(face_dir, exist_ok=True)
        
        for filename in os.listdir(face_dir):
            if filename.endswith((".jpg", ".png")):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(face_dir, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    # 转换为灰度图
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # 直方图均衡化
                    gray = cv2.equalizeHist(gray)
                    # 检测人脸
                    faces = self.face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.05,
                        minNeighbors=3,
                        minSize=(50, 50)
                    )
                    if len(faces) > 0:
                        # 只保存第一个检测到的人脸
                        x, y, w, h = faces[0]
                        # 扩大人脸区域
                        x = max(0, x - w//4)
                        y = max(0, y - h//4)
                        w = min(gray.shape[1] - x, w + w//2)
                        h = min(gray.shape[0] - y, h + h//2)
                        face_roi = gray[y:y+h, x:x+w]
                        # 调整大小并标准化
                        face_roi = cv2.resize(face_roi, (100, 100))
                        face_roi = cv2.equalizeHist(face_roi)
                        # 保存原始图像和预处理后的图像
                        self.known_faces[name] = {
                            'original': face_roi,
                            'processed': self.preprocess_face(face_roi)
                        }
                    
    def preprocess_face(self, face):
        """预处理人脸图像"""
        # 高斯模糊
        face = cv2.GaussianBlur(face, (3, 3), 0)
        # 直方图均衡化
        face = cv2.equalizeHist(face)
        # 归一化
        face = cv2.normalize(face, None, 0, 255, cv2.NORM_MINMAX)
        return face
    
    def calculate_similarity(self, face1, face2):
        """计算两张人脸的相似度"""
        # 使用多种方法计算相似度
        methods = [
            cv2.TM_CCOEFF_NORMED,
            cv2.TM_CCORR_NORMED,
            cv2.TM_SQDIFF_NORMED
        ]
        
        scores = []
        for method in methods:
            result = cv2.matchTemplate(face1, face2, method)
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                score = 1 - result[0][0]  # 对于SQDIFF，值越小越相似
            else:
                score = result[0][0]  # 对于其他方法，值越大越相似
            scores.append(score)
        
        # 返回平均相似度
        return sum(scores) / len(scores)
    
    def detect_faces(self, frame):
        """检测图像中的人脸"""
        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 直方图均衡化
        gray = cv2.equalizeHist(gray)
        
        # 检测人脸
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(50, 50)
        )
        
        return faces
    
    def recognize_face(self, frame):
        """识别图像中的人脸"""
        faces = self.detect_faces(frame)
        recognized_faces = []
        
        for (x, y, w, h) in faces:
            # 扩大人脸区域
            x = max(0, x - w//4)
            y = max(0, y - h//4)
            w = min(frame.shape[1] - x, w + w//2)
            h = min(frame.shape[0] - y, h + h//2)
            
            # 提取人脸区域
            face_roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            # 调整大小
            face_roi = cv2.resize(face_roi, (100, 100))
            # 预处理
            processed_face = self.preprocess_face(face_roi)
            
            # 计算相似度
            best_match = None
            best_score = 0
            
            for name, known_face in self.known_faces.items():
                # 计算相似度
                score = self.calculate_similarity(processed_face, known_face['processed'])
                
                if score > best_score:
                    best_score = score
                    best_match = name
            
            # 降低相似度阈值，并打印调试信息
            print(f"Best match: {best_match}, Score: {best_score}")
            if best_match and best_score > 0.3:  # 进一步降低阈值
                recognized_faces.append((x, y, w, h, best_match))
            else:
                recognized_faces.append((x, y, w, h, "Unknown"))
                
        return recognized_faces 