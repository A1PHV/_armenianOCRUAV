import cv2
import numpy as np
import time
import logging

class CameraHandler:
    def __init__(self, config):
        self.config = config
        self.camera = None
        self.logger = logging.getLogger('CameraHandler')
        self.initialize_camera()
        
    def initialize_camera(self):
        """Инициализация камеры"""
        try:
            self.camera = cv2.VideoCapture(self.config.get('device_id', 0))
            
            if not self.camera.isOpened():
                raise Exception("Cannot open camera")
                
            # Настройка параметров камеры
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get('width', 1920))
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get('height', 1080))
            self.camera.set(cv2.CAP_PROP_FPS, self.config.get('fps', 30))
            
            # Настройка фокуса
            if self.config.get('auto_focus', True):
                self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            else:
                self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                self.camera.set(cv2.CAP_PROP_FOCUS, self.config.get('focus', 50))
                
            self.logger.info("Camera initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            raise
            
    def capture_frame(self):
        """Захват кадра"""
        try:
            ret, frame = self.camera.read()
            if not ret:
                self.logger.warning("Failed to capture frame")
                return None
                
            # Предварительная обработка кадра
            if self.config.get('apply_preprocessing', True):
                frame = self.preprocess_frame(frame)
                
            return frame
            
        except Exception as e:
            self.logger.error(f"Frame capture error: {e}")
            return None
            
    def preprocess_frame(self, frame):
        """Предварительная обработка кадра"""
        try:
            # Коррекция экспозиции
            if self.config.get('auto_exposure', True):
                frame = self.auto_exposure_correction(frame)
                
            # Улучшение резкости
            if self.config.get('enhance_sharpness', True):
                frame = self.enhance_sharpness(frame)
                
            return frame
            
        except Exception as e:
            self.logger.error(f"Preprocessing error: {e}")
            return frame
            
    def auto_exposure_correction(self, frame):
        """Автоматическая коррекция экспозиции"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result
        
    def enhance_sharpness(self, frame):
        """Улучшение резкости изображения"""
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        
        sharpened = cv2.filter2D(frame, -1, kernel)
        result = cv2.addWeighted(frame, 0.5, sharpened, 0.5, 0)
        
        return result
        
    def close(self):
        """Закрытие камеры"""
        if self.camera:
            self.camera.release()
            self.logger.info("Camera closed")
