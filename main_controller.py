import time
import logging
import signal
import sys
import json
from pathlib import Path
from datetime import datetime
import threading
import queue

from vision_module.camera_handler import CameraHandler
from vision_module.symbol_detector import SymbolDetector
from vision_module.ocr_processor import ArmenianOCRProcessor
from flight_controller_module.mavlink_handler import MAVLinkHandler
from flight_controller_module.gps_data import GPSDataProcessor
from data_module.excel_logger import ExcelLogger
from utils.config import Config

class DroneVisionController:
    def __init__(self):
        """Инициализация системы"""
        self.setup_logging()
        self.setup_storage()
        self.setup_status_indicators()
        self.running = True
        self.processing_queue = queue.Queue(maxsize=20)
        
        # Обработка сигналов завершения
        signal.signal(signal.SIGTERM, self.graceful_shutdown)
        signal.signal(signal.SIGINT, self.graceful_shutdown)
        
        # Загрузка конфигурации
        self.config = Config()
        
        # Инициализация компонентов
        self.camera = None
        self.detector = None
        self.ocr = None
        self.mavlink = None
        self.gps_processor = None
        self.logger = None
        
    def setup_logging(self):
        """Настройка системы логирования"""
        log_dir = Path("/media/khadas/CV_DATA/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'system.log'),
                logging.StreamHandler()
            ]
        )
        self.system_logger = logging.getLogger('DroneCV')
        
    def setup_storage(self):
        """Настройка хранилища данных"""
        self.sd_path = Path("/media/khadas/CV_DATA")
        self.sd_path.mkdir(exist_ok=True)
        
        # Создание структуры папок
        (self.sd_path / "detections").mkdir(exist_ok=True)
        (self.sd_path / "images").mkdir(exist_ok=True)
        (self.sd_path / "logs").mkdir(exist_ok=True)
        (self.sd_path / "config").mkdir(exist_ok=True)
        
    def setup_status_indicators(self):
        """Настройка светодиодной индикации"""
        self.status = "initializing"
        
    def set_status_led(self, status):
        """Управление статусными светодиодами"""
        self.status = status
        self.system_logger.info(f"Status changed to: {status}")
        
    def initialize_components(self):
        """Инициализация всех компонентов системы"""
        try:
            self.system_logger.info("Initializing components...")
            
            # Инициализация GPS процессора
            self.gps_processor = GPSDataProcessor()
            self.system_logger.info("GPS processor initialized")
            
            # Инициализация камеры
            self.camera = CameraHandler(self.config.camera_config)
            self.system_logger.info("Camera initialized")
            
            # Инициализация детектора символов
            self.detector = SymbolDetector(self.config.detection_config)
            self.system_logger.info("Symbol detector initialized")
            
            # Инициализация OCR
            self.ocr = ArmenianOCRProcessor(self.config.ocr_config)
            self.system_logger.info("OCR processor initialized")
            
            # Инициализация логгера данных
            self.logger = ExcelLogger(self.sd_path / "detections")
            self.system_logger.info("Data logger initialized")
            
            return True
            
        except Exception as e:
            self.system_logger.error(f"Component initialization failed: {e}")
            self.set_status_led("error")
            return False
            
    def wait_for_flight_controller(self):
        """Ожидание подключения к полетному контроллеру"""
        self.set_status_led("waiting_fc")
        self.system_logger.info("Waiting for flight controller connection...")
        
        retry_count = 0
        max_retries = 60  # 5 минут ожидания
        
        while self.running and retry_count < max_retries:
            try:
                self.mavlink = MAVLinkHandler(self.config.mavlink_config)
                if self.mavlink.connect():
                    self.system_logger.info("Flight controller connected")
                    self.set_status_led("ready")
                    return True
                    
            except Exception as e:
                self.system_logger.warning(f"FC connection attempt {retry_count + 1} failed: {e}")
                
            time.sleep(5)
            retry_count += 1
            
        self.system_logger.error("Failed to connect to flight controller")
        self.set_status_led("error")
        return False
        
    def process_frame_worker(self):
        """Рабочий поток обработки кадров"""
        while self.running:
            try:
                frame_data = self.processing_queue.get(timeout=1)
                if frame_data is None:
                    break
                    
                frame, timestamp, gps_data = frame_data
                self.process_single_frame(frame, timestamp, gps_data)
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.system_logger.error(f"Frame processing error: {e}")
                
    def process_single_frame(self, frame, timestamp, gps_data):
        """Обработка одного кадра"""
        try:
            # Валидация GPS данных
            if not self.gps_processor.validate_gps_data(gps_data):
                self.system_logger.warning("Invalid GPS data, skipping frame")
                return
                
            # Детекция символов на кадре
            detections = self.detector.detect_symbols(frame)
            
            if not detections:
                return
                
            self.system_logger.info(f"Found {len(detections)} potential symbols")
            
            # Обработка каждой детекции
            for i, detection in enumerate(detections):
                try:
                    # Извлечение области с символом
                    symbol_region = self.detector.extract_symbol_region(frame, detection)
                    
                    # OCR распознавание
                    ocr_result = self.ocr.recognize_armenian_text(symbol_region)
                    
                    if ocr_result['confidence'] > self.config.min_confidence:
                        # Сохранение изображения
                        img_filename = f"detection_{timestamp}_{i}.jpg"
                        img_path = self.save_detection_image(symbol_region, img_filename)
                        
                        # Логирование результата
                        self.logger.log_detection(
                            symbol=ocr_result['text'],
                            confidence=ocr_result['confidence'],
                            gps_lat=gps_data['lat'],
                            gps_lon=gps_data['lon'],
                            gps_alt=gps_data['alt'],
                            timestamp=timestamp,
                            image_path=img_path,
                            detection_bbox=detection['bbox'],
                            symbol_id=ocr_result.get('symbol_id')
                        )
                        
                        self.system_logger.info(
                            f"Detected symbol: '{ocr_result['text']}' (ID: {ocr_result.get('symbol_id')}) "
                            f"(conf: {ocr_result['confidence']:.2f}) "
                            f"at {gps_data['lat']:.6f}, {gps_data['lon']:.6f}"
                        )
                        
                except Exception as e:
                    self.system_logger.error(f"Error processing detection {i}: {e}")
                    
        except Exception as e:
            self.system_logger.error(f"Frame processing error: {e}")
            
    def save_detection_image(self, image, filename):
        """Сохранение изображения детекции"""
        try:
            img_dir = self.sd_path / "images" / datetime.now().strftime("%Y%m%d")
            img_dir.mkdir(exist_ok=True)
            
            img_path = img_dir / filename
            import cv2
            cv2.imwrite(str(img_path), image)
            
            return str(img_path.relative_to(self.sd_path))
            
        except Exception as e:
            self.system_logger.error(f"Error saving image {filename}: {e}")
            return ""
            
    def main_loop(self):
        """Основной цикл работы системы"""
        self.system_logger.info("=== Drone CV System Started - SKAT Competition 2025 ===")
        
        # Инициализация компонентов
        if not self.initialize_components():
            return
            
        # Ожидание подключения к полетному контроллеру
        if not self.wait_for_flight_controller():
            return
            
        # Запуск рабочего потока обработки
        processing_thread = threading.Thread(target=self.process_frame_worker)
        processing_thread.start()
        
        # Основной цикл захвата кадров
        frame_count = 0
        last_status_time = time.time()
        
        try:
            while self.running:
                current_time = time.time()
                
                # Проверка подключения к ПК
                if not self.mavlink.is_connected():
                    self.system_logger.warning("Lost connection to flight controller")
                    if not self.wait_for_flight_controller():
                        break
                        
                # Получение GPS данных
                gps_data = self.mavlink.get_current_gps()
                
                # Проверка готовности к работе
                if not gps_data or gps_data.get('fix_quality', 0) < 3:
                    time.sleep(0.5)
                    continue
                    
                # Проверка режима полета (только в полете)
                if not self.mavlink.is_armed():
                    time.sleep(1)
                    continue
                    
                self.set_status_led("active")
                
                # Захват кадра
                frame = self.camera.capture_frame()
                if frame is None:
                    continue
                    
                # Добавление в очередь обработки
                timestamp = current_time
                try:
                    self.processing_queue.put_nowait((frame, timestamp, gps_data))
                except queue.Full:
                    self.system_logger.warning("Processing queue full, dropping frame")
                    
                frame_count += 1
                
                # Периодический вывод статистики
                if current_time - last_status_time > 30:
                    queue_size = self.processing_queue.qsize()
                    self.system_logger.info(
                        f"Status: frames={frame_count}, queue={queue_size}, "
                        f"GPS=({gps_data['lat']:.6f}, {gps_data['lon']:.6f})"
                    )
                    last_status_time = current_time
                    
                # Контроль частоты кадров
                time.sleep(1.0 / self.config.target_fps)
                
        except KeyboardInterrupt:
            self.system_logger.info("Keyboard interrupt received")
        except Exception as e:
            self.system_logger.error(f"Main loop error: {e}")
        finally:
            # Завершение рабочего потока
            self.processing_queue.put(None)
            processing_thread.join(timeout=10)
            
        self.graceful_shutdown()
        
    def graceful_shutdown(self, signum=None, frame=None):
        """Корректное завершение работы"""
        if not self.running:
            return
            
        self.system_logger.info("=== Initiating graceful shutdown ===")
        self.running = False
        self.set_status_led("shutdown")
        
        # Закрытие компонентов
        try:
            if self.camera:
                self.camera.close()
            if self.mavlink:
                self.mavlink.close()
            if self.logger:
                self.logger.close()
        except Exception as e:
            self.system_logger.error(f"Error during shutdown: {e}")
            
        self.system_logger.info("=== Shutdown complete ===")
        sys.exit(0)

if __name__ == "__main__":
    controller = DroneVisionController()
    controller.main_loop()
