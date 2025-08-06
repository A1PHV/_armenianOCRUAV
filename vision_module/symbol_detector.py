import cv2
import numpy as np
import logging

class SymbolDetector:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('SymbolDetector')
        
        # Параметры для символов 3x3 метра на разных высотах
        self.min_symbol_area = config.get('min_symbol_area', 2000)
        self.max_symbol_area = config.get('max_symbol_area', 100000)
        
    def detect_symbols(self, frame):
        """Детекция потенциальных символов на кадре"""
        try:
            # Предобработка для детекции
            processed = self.preprocess_for_detection(frame)
            
            # Поиск контуров
            contours = self.find_symbol_contours(processed)
            
            # Фильтрация и анализ контуров
            detections = self.analyze_contours(contours, frame.shape)
            
            self.logger.debug(f"Found {len(detections)} potential symbols")
            return detections
            
        except Exception as e:
            self.logger.error(f"Symbol detection error: {e}")
            return []
            
    def preprocess_for_detection(self, frame):
        """Предобработка кадра для детекции символов"""
        # Конвертация в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Размытие для уменьшения шума
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Адаптивная пороговая обработка для контраста черного на белом
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2  # INV для черных символов на белом фоне
        )
        
        # Морфологические операции
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return closed
        
    def find_symbol_contours(self, processed_frame):
        """Поиск контуров потенциальных символов"""
        contours, _ = cv2.findContours(
            processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours
        
    def analyze_contours(self, contours, frame_shape):
        """Анализ контуров и создание детекций"""
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Фильтрация по размеру (символы 3x3 метра)
            if area < self.min_symbol_area or area > self.max_symbol_area:
                continue
                
            # Ограничивающий прямоугольник
            x, y, w, h = cv2.boundingRect(contour)
            
            # Проверка пропорций (символы должны быть примерно квадратными)
            aspect_ratio = w / h
            if aspect_ratio < 0.4 or aspect_ratio > 2.5:
                continue
                
            # Проверка заполненности контура
            contour_area = cv2.contourArea(contour)
            bbox_area = w * h
            fill_ratio = contour_area / bbox_area
            
            if fill_ratio < 0.1 or fill_ratio > 0.9:
                continue
                
            # Создание детекции
            detection = {
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'fill_ratio': fill_ratio,
                'confidence': self.calculate_detection_confidence(
                    area, aspect_ratio, fill_ratio
                )
            }
            
            detections.append(detection)
            
        # Сортировка по уверенности
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
        
    def calculate_detection_confidence(self, area, aspect_ratio, fill_ratio):
        """Расчет уверенности детекции"""
        # Нормализация для символов 3x3 метра
        area_score = min(area / 20000, 1.0)  
        
        # Предпочтение квадратных символов
        ratio_score = 1.0 - abs(1.0 - aspect_ratio) if aspect_ratio <= 2.0 else 0.5
        
        # Предпочтение хорошо заполненных контуров
        fill_score = fill_ratio if fill_ratio <= 0.6 else 1.0 - fill_ratio
        
        confidence = (area_score + ratio_score + fill_score) / 3.0
        return confidence
        
    def extract_symbol_region(self, frame, detection):
        """Извлечение области символа из кадра"""
        x, y, w, h = detection['bbox']
        
        # Добавляем отступы для лучшего OCR
        padding = 15
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        region = frame[y1:y2, x1:x2]
        return region
