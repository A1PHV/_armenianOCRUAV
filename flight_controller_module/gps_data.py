import time
import math
import logging

class GPSDataProcessor:
    
    def __init__(self):
        self.logger = logging.getLogger('GPSData')
        
    def validate_gps_data(self, gps_data):
        """Валидация GPS данных"""
        if not gps_data:
            return False
            
        required_fields = ['lat', 'lon', 'alt', 'fix_quality']
        for field in required_fields:
            if field not in gps_data:
                return False
                
        # Проверка качества GPS фикса (3+ для конкурса)
        if gps_data.get('fix_quality', 0) < 3:
            return False
            
        # Проверка разумных координат
        lat, lon = gps_data['lat'], gps_data['lon']
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            self.logger.warning(f"GPS coordinates out of range: {lat}, {lon}")
            return False
            
        return True
        
    def format_for_competition(self, lat, lon):
        lat_e7 = int(round(lat * 1e7))
        lon_e7 = int(round(lon * 1e7))
        
        return lat_e7, lon_e7
        
    def calculate_distance_meters(self, lat1, lon1, lat2, lon2):
        """Расчет расстояния между GPS точками в метрах"""
        # Формула гаверсинусов
        R = 6371000  # Радиус Земли в метрах
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lon/2) * math.sin(delta_lon/2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance
        
    def is_within_accuracy(self, detected_lat, detected_lon, actual_lat, actual_lon, 
                          max_error_meters=10):
        distance = self.calculate_distance_meters(
            detected_lat, detected_lon, actual_lat, actual_lon
        )
        
        return distance <= max_error_meters

