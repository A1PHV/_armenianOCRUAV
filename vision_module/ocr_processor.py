import cv2
import numpy as np
import logging
from pathlib import Path

# Импорт pytesseract с fallback
try:
    import pytesseract
except ImportError:
    pytesseract = None
    logging.warning("pytesseract not available, OCR will not work")

class ArmenianOCRProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('ArmenianOCR')
        
        # Конфигурация Tesseract для армянского языка
        self.base_config = '--oem 1 --psm 8 -l hye'
        
        # Нумерация слева направо, сверху вниз, начиная с 0
        self.armenian_symbols = [
            # Строка 0 (ID 0-8): Ա Բ Գ Դ Ե Զ Է Ը Թ
            'Ա', 'Բ', 'Գ', 'Դ', 'Ե', 'Զ', 'Է', 'Ը', 'Թ',
            # Строка 1 (ID 9-17): Ժ Ի Լ Խ Ծ Կ Հ Ձ Ղ  
            'Ժ', 'Ի', 'Լ', 'Խ', 'Ծ', 'Կ', 'Հ', 'Ձ', 'Ղ',
            # Строка 2 (ID 18-26): Ճ Մ Յ Ն Շ Ո Չ Պ Ջ
            'Ճ', 'Մ', 'Յ', 'Ն', 'Շ', 'Ո', 'Չ', 'Պ', 'Ջ',
            # Строка 3 (ID 27-35): Ռ Ս Վ Տ Ր Ց Ւ Փ Ք
            'Ռ', 'Ս', 'Վ', 'Տ', 'Ր', 'Ց', 'Ւ', 'Փ', 'Ք'
        ]
        
        # Создание словаря для быстрого поиска ID символа
        self.symbol_to_id = {symbol: idx for idx, symbol in enumerate(self.armenian_symbols)}
        
        # Проверка доступности модели
        self.check_tesseract_model()
        
    def check_tesseract_model(self):
        """Проверка доступности модели Tesseract"""
        if pytesseract is None:
            return
            
        try:
            available_languages = pytesseract.get_languages()
            if 'hye' in available_languages:
                self.logger.info("Armenian Tesseract model is available")
            else:
                self.logger.error("Armenian Tesseract model not found!")
                
        except Exception as e:
            self.logger.error(f"Error checking Tesseract models: {e}")
            
    def recognize_armenian_text(self, image_region):
        """Распознавание армянского текста в области изображения"""
        try:
            if pytesseract is None:
                return {'text': '', 'confidence': 0.0, 'symbol_id': None}
                
            if image_region is None or image_region.size == 0:
                return {'text': '', 'confidence': 0.0, 'symbol_id': None}
                
            # Предобработка для OCR
            processed = self.preprocess_for_ocr(image_region)
            
            # Динамическая настройка параметров OCR
            config = self.get_dynamic_config(processed)
            
            # OCR распознавание
            text = pytesseract.image_to_string(processed, config=config)
            
            # Получение данных с confidence
            data = pytesseract.image_to_data(
                processed, config=config, output_type=pytesseract.Output.DICT
            )
            
            # Расчет среднего confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            # Очистка и валидация текста
            cleaned_text = self.clean_armenian_text(text)
            
            # Определение ID символа для конкурса СКАТ
            symbol_id = self.get_symbol_id(cleaned_text)
            
            result = {
                'text': cleaned_text,
                'confidence': avg_confidence / 100.0,
                'raw_text': text,
                'word_confidences': confidences,
                'symbol_id': symbol_id
            }
            
            self.logger.debug(
                f"OCR result: '{cleaned_text}' (ID: {symbol_id}) (confidence: {avg_confidence:.1f}%)"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"OCR processing error: {e}")
            return {'text': '', 'confidence': 0.0, 'symbol_id': None, 'error': str(e)}
            
    def preprocess_for_ocr(self, image):
        """Предобработка изображения для OCR"""
        # Конвертация в оттенки серого
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Масштабирование для улучшения качества OCR (символы 3x3м)
        scale_factor = 4
        height, width = gray.shape
        scaled = cv2.resize(gray, (width * scale_factor, height * scale_factor), 
                          interpolation=cv2.INTER_CUBIC)
        
        # Повышение контраста
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(scaled)
        
        # Бинаризация (черные символы на белом фоне)
        _, binary = cv2.threshold(contrast_enhanced, 0, 255, 
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Морфологические операции для очистки
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
        
    def get_dynamic_config(self, processed_image):
        """Динамическая настройка конфигурации OCR"""
        height, width = processed_image.shape
        
        # Выбор PSM режима в зависимости от размера
        if width * height > 100000:  # Большая область
            psm = 6  # Единый блок текста
        elif width * height > 20000:  # Средняя область
            psm = 8  # Одно слово
        else:  # Маленькая область
            psm = 10  # Один символ
            
        # Настройка DPI для больших символов
        dpi = min(300, max(150, int(np.sqrt(width * height) / 20)))
        
        config = f'--oem 1 --psm {psm} --dpi {dpi} -l hye'
        
        return config
        
    def clean_armenian_text(self, text):
        """Очистка и валидация армянского текста"""
        if not text:
            return ""
            
        # Удаление лишних пробелов и переносов строк
        cleaned = text.strip().replace('\n', ' ').replace('\r', '')
        cleaned = ' '.join(cleaned.split())
        
        # Фильтрация только армянских символов
        armenian_chars = []
        for char in cleaned:
            # Армянские символы: U+0530-U+058F
            if '\u0530' <= char <= '\u058F':
                armenian_chars.append(char)
                
        result = ''.join(armenian_chars).strip()
        return result
        
    def get_symbol_id(self, symbol_text):
        if not symbol_text:
            return None
            
        # Берем первый символ (если распознано несколько)
        first_symbol = symbol_text[0] if symbol_text else None
        
        # Поиск ID в таблице символов
        symbol_id = self.symbol_to_id.get(first_symbol)
        
        if symbol_id is not None:
            self.logger.debug(f"Symbol '{first_symbol}' mapped to ID {symbol_id}")
        else:
            self.logger.warning(f"Symbol '{first_symbol}' not found in competition table")
            
        return symbol_id

