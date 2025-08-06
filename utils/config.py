import json
import logging
from pathlib import Path

class Config:
    def __init__(self, config_path="/home/khadas/drone-cv/config/settings.json"):
        self.config_path = Path(config_path)
        self.logger = logging.getLogger('Config')
        self.load_config()
        
    def load_config(self):
        """Загрузка конфигурации"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                self.config = self.get_default_config()
                self.save_config()
                
            self.logger.info("Configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            self.config = self.get_default_config()
            
    def get_default_config(self):
        """Конфигурация по умолчанию для конкурса СКАТ"""
        return {
            "camera_config": {
                "device_id": 0,
                "width": 1920,
                "height": 1080,
                "fps": 30,
                "auto_focus": True,
                "focus": 50,
                "apply_preprocessing": True,
                "auto_exposure": True,
                "enhance_sharpness": True
            },
            "detection_config": {
                "min_symbol_area": 2000,    # Для символов 3x3 метра
                "max_symbol_area": 100000   # Верхний предел
            },
            "ocr_config": {
                "model_name": "hye",
                "fallback_model": "hye",
                "engine_mode": 1,           # LSTM режим
                "psm_mode": 8              # Одно слово
            },
            "mavlink_config": {
                "connection_string": "/dev/ttyUSB0",
                "baud_rate": 57600
            },
            "system_config": {
                "target_fps": 3,            
                "min_confidence": 0.8,      
                "max_processing_queue": 20,
                "competition_mode": True,   
                "symbol_size_meters": 3.0,  # Размер символов
                "max_altitude_m": 150,      # Ограничение конкурса
                "optimal_altitude_m": 85    # Оптимальная высота
            }
        }
        
    def save_config(self):
        """Сохранение конфигурации"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            
    @property
    def camera_config(self):
        return self.config["camera_config"]
        
    @property
    def detection_config(self):
        return self.config["detection_config"]
        
    @property
    def ocr_config(self):
        return self.config["ocr_config"]
        
    @property
    def mavlink_config(self):
        return self.config["mavlink_config"]
        
    @property
    def target_fps(self):
        return self.config["system_config"]["target_fps"]
        
    @property
    def min_confidence(self):
        return self.config["system_config"]["min_confidence"]
