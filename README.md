# **Система компьютерного зрения для БВС - Конкурс СКАТ 2025**

Автономная система распознавания армянских символов размером 3×3 метра с записью GPS координат для Студенческого конкурса авиационного творчества.

## Оборудование

- **Плата**: Khadas VIM3 PRO
- **Камера**: Khadas K-CM-002  
- **OCR**: Tesseract с моделью hye (tessdata_best)
- **Связь**: MAVLink с полетным контроллером
- **Хранение**: SD карта FAT32

## Структура проекта
drone-cv/
├── main_controller.py          # Главный контроллер
├── vision_module/              # Модули компьютерного зрения
│   ├── camera_handler.py
│   ├── symbol_detector.py
│   └── ocr_processor.py
├── flight_controller_module/   # Связь с полетным контроллером
│   ├── mavlink_handler.py
│   └── gps_data.py
├── data_module/               # Логирование данных
│   └── excel_logger.py
├── utils/                     # Утилиты
│   └── config.py
└── config/                    # Конфигурация
    └── settings.json
