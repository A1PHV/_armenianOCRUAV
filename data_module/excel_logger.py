import pandas as pd
import time
import logging
import csv
from pathlib import Path
from datetime import datetime
import threading
from flight_controller_module.gps_data import GPSDataProcessor

class ExcelLogger:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('ExcelLogger')
        self.data_buffer = []
        self.csv_buffer = []
        self.buffer_lock = threading.Lock()
        
        # GPS процессор для форматирования
        self.gps_processor = GPSDataProcessor()
        
        # Создание файла для текущего полета
        self.current_flight_file = self.create_flight_file()
        
        self.competition_csv_file = self.create_competition_csv()
        
    def create_flight_file(self):
        """Создание файла для текущего полета"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flight_{timestamp}.xlsx"
        filepath = self.output_dir / filename
        
        # Создание начального Excel файла с заголовками
        columns = [
            'Timestamp', 'Symbol', 'Symbol_ID', 'Confidence', 'GPS_Lat', 'GPS_Lon', 
            'GPS_Alt', 'Image_Path', 'Detection_X', 'Detection_Y', 
            'Detection_W', 'Detection_H'
        ]
        
        df = pd.DataFrame(columns=columns)
        df.to_excel(filepath, index=False)
        
        self.logger.info(f"Created flight data file: {filename}")
        return filepath
        
    def create_competition_csv(self):
        csv_filename = "objects-coordinates.csv"
        csv_filepath = self.output_dir / csv_filename
        
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            pass
            
        self.logger.info(f"Created competition CSV file: {csv_filename}")
        return csv_filepath
        
    def log_detection(self, symbol, confidence, gps_lat, gps_lon, gps_alt, 
                     timestamp, image_path, detection_bbox, symbol_id=None):
        """Логирование обнаружения символа"""
        try:
            x, y, w, h = detection_bbox
            
            detection_data = {
                'Timestamp': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'Symbol': symbol,
                'Symbol_ID': symbol_id,
                'Confidence': round(confidence, 3),
                'GPS_Lat': round(gps_lat, 8),
                'GPS_Lon': round(gps_lon, 8),
                'GPS_Alt': round(gps_alt, 2),
                'Image_Path': image_path,
                'Detection_X': x,
                'Detection_Y': y,
                'Detection_W': w,
                'Detection_H': h
            }
            
            with self.buffer_lock:
                self.data_buffer.append(detection_data)
                
                if symbol_id is not None:
                    lat_e7, lon_e7 = self.gps_processor.format_for_competition(gps_lat, gps_lon)
                    
                    competition_entry = {
                        'symbol_id': symbol_id,
                        'lat_e7': lat_e7,
                        'lon_e7': lon_e7
                    }
                    self.csv_buffer.append(competition_entry)
                
            # Периодическое сохранение
            if len(self.data_buffer) >= 5:  # Каждые 5 детекций
                self.flush_to_excel()
                self.flush_to_competition_csv()
                
            self.logger.info(f"Logged detection: {symbol} (ID: {symbol_id}) at ({gps_lat:.6f}, {gps_lon:.6f})")
            
        except Exception as e:
            self.logger.error(f"Error logging detection: {e}")
            
    def flush_to_excel(self):
        """Сохранение буфера в Excel файл"""
        try:
            with self.buffer_lock:
                if not self.data_buffer:
                    return
                    
                # Читаем существующий файл
                try:
                    existing_df = pd.read_excel(self.current_flight_file)
                except:
                    existing_df = pd.DataFrame()
                    
                # Добавляем новые данные
                new_df = pd.DataFrame(self.data_buffer)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                
                # Сохраняем обратно
                combined_df.to_excel(self.current_flight_file, index=False)
                
                self.logger.info(f"Flushed {len(self.data_buffer)} detections to Excel")
                self.data_buffer.clear()
                
        except Exception as e:
            self.logger.error(f"Error flushing to Excel: {e}")
            
    def flush_to_competition_csv(self):
        try:
            with self.buffer_lock:
                if not self.csv_buffer:
                    return
                    
                with open(self.competition_csv_file, 'a', newline='', encoding='utf-8') as csvfile:
                    for entry in self.csv_buffer:
                        csvfile.write(f"{entry['symbol_id']},{entry['lat_e7']},{entry['lon_e7']}\n")
                        
                self.logger.info(f"Flushed {len(self.csv_buffer)} entries to competition CSV")
                self.csv_buffer.clear()
                
        except Exception as e:
            self.logger.error(f"Error flushing to competition CSV: {e}")
            
    def create_summary_report(self):
        """Создание сводного отчета"""
        try:
            all_files = list(self.output_dir.glob("flight_*.xlsx"))
            if not all_files:
                return
                
            all_data = []
            for file in all_files:
                try:
                    df = pd.read_excel(file)
                    all_data.append(df)
                except Exception as e:
                    self.logger.error(f"Error reading {file}: {e}")
                    
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                
                # Статистика
                summary = {
                    'Total_Detections': len(combined_df),
                    'Unique_Symbols': combined_df['Symbol'].nunique(),
                    'Unique_Symbol_IDs': combined_df['Symbol_ID'].nunique(),
                    'Average_Confidence': combined_df['Confidence'].mean(),
                    'Flights_Count': len(all_files),
                    'Competition_Score_Estimate': len(combined_df[combined_df['Symbol_ID'].notna()]) * 20  
                }
                
                summary_df = pd.DataFrame([summary])
                summary_file = self.output_dir / "summary_report.xlsx"
                
                with pd.ExcelWriter(summary_file) as writer:
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    combined_df.to_excel(writer, sheet_name='All_Detections', index=False)
                    
                self.logger.info(f"Created summary report: {summary_file}")
                
        except Exception as e:
            self.logger.error(f"Error creating summary report: {e}")
            
    def close(self):
        """Закрытие логгера"""
        self.flush_to_excel()
        self.flush_to_competition_csv()
        self.create_summary_report()
        self.logger.info("Excel logger closed")
