import time
import logging
import threading

# Импорт MAVLink с fallback
try:
    from pymavlink import mavutil
except ImportError:
    mavutil = None
    logging.warning("pymavlink not available, flight controller connection will not work")

class MAVLinkHandler:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('MAVLink')
        self.connection = None
        self.connected = False
        self.last_gps_data = None
        self.data_lock = threading.Lock()
        
    def connect(self):
        """Подключение к полетному контроллеру"""
        if mavutil is None:
            self.logger.error("pymavlink not available")
            return False
            
        try:
            connection_string = self.config.get('connection_string', '/dev/ttyUSB0')
            baud_rate = self.config.get('baud_rate', 57600)
            
            self.logger.info(f"Connecting to flight controller: {connection_string}")
            
            self.connection = mavutil.mavlink_connection(
                connection_string, 
                baud=baud_rate
            )
            
            # Ожидание heartbeat
            self.logger.info("Waiting for heartbeat...")
            self.connection.wait_heartbeat()
            
            self.connected = True
            self.logger.info("Flight controller connected successfully")
            
            # Запуск потока чтения данных
            self.start_data_thread()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to flight controller: {e}")
            self.connected = False
            return False
            
    def start_data_thread(self):
        """Запуск потока чтения телеметрии"""
        self.data_thread = threading.Thread(target=self.data_reader_loop)
        self.data_thread.daemon = True
        self.data_thread.start()
        
    def data_reader_loop(self):
        """Цикл чтения данных от полетного контроллера"""
        while self.connected:
            try:
                msg = self.connection.recv_match(blocking=False, timeout=1)
                if msg:
                    self.process_message(msg)
                time.sleep(0.01)  # 100 Hz
                
            except Exception as e:
                self.logger.error(f"Error reading telemetry: {e}")
                self.connected = False
                break
                
    def process_message(self, msg):
        """Обработка сообщений от полетного контроллера"""
        try:
            if msg.get_type() == 'GLOBAL_POSITION_INT':
                with self.data_lock:
                    self.last_gps_data = {
                        'lat': msg.lat / 1e7,
                        'lon': msg.lon / 1e7,
                        'alt': msg.alt / 1000.0,
                        'relative_alt': msg.relative_alt / 1000.0,
                        'vx': msg.vx / 100.0,
                        'vy': msg.vy / 100.0,
                        'vz': msg.vz / 100.0,
                        'hdg': msg.hdg / 100.0,
                        'timestamp': time.time(),
                        'fix_quality': 3  
                    }
                    
            elif msg.get_type() == 'GPS_RAW_INT':
                # Дополнительная информация о GPS
                if hasattr(msg, 'fix_type'):
                    with self.data_lock:
                        if self.last_gps_data:
                            self.last_gps_data['fix_quality'] = msg.fix_type
                            self.last_gps_data['satellites_visible'] = getattr(msg, 'satellites_visible', 0)
                            
        except Exception as e:
            self.logger.error(f"Error processing message {msg.get_type()}: {e}")
            
    def get_current_gps(self):
        """Получение текущих GPS координат"""
        with self.data_lock:
            if self.last_gps_data and \
               time.time() - self.last_gps_data['timestamp'] < 5.0:
                return self.last_gps_data.copy()
            return None
            
    def is_armed(self):
        """Проверка состояния ARMED дрона"""
        if not self.connected or mavutil is None:
            return False
            
        try:
            msg = self.connection.recv_match(type='HEARTBEAT', blocking=False)
            if msg:
                return msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            return False
        except:
            return False
            
    def is_connected(self):
        """Проверка подключения"""
        return self.connected
        
    def close(self):
        """Закрытие соединения"""
        self.connected = False
        if hasattr(self, 'data_thread'):
            self.data_thread.join(timeout=2)
        if self.connection:
            self.connection.close()
        self.logger.info("MAVLink connection closed")
