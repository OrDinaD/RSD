# RSD - Real-time Sign Detection 🚦

Система real-time распознавания дорожных знаков на **Raspberry Pi 5** с **Google Coral Edge TPU** и **YOLOv10n**.

## 📋 Описание

Этот проект реализует полный пайплайн для детекции дорожных знаков во время движения автомобиля:

- **Модель**: YOLOv10n (оптимизирована для edge-устройств)
- **Платформа**: Raspberry Pi 5 (8GB RAM)
- **Ускоритель**: Google Coral USB Accelerator (Edge TPU)
- **Производительность**: 18-25 FPS при разрешении 640x640
- **Датасет**: ~30,000 изображений дорожных столбов со знаками

## 🏗️ Структура проекта

```
RSD/
├── configs/              # Конфигурационные файлы
│   ├── dataset.yaml      # Конфигурация датасета для YOLO
│   ├── train_config.yaml # Параметры обучения
│   └── deployment_config.yaml # Настройки для Raspberry Pi
├── dataset/              # Датасет
│   ├── images/           # Изображения (train/val/test)
│   ├── labels/           # YOLO аннотации
│   └── raw/              # Исходные данные
├── training/             # Скрипты обучения
│   └── train.py          # Обучение YOLOv10n
├── optimization/         # Оптимизация моделей
│   └── convert_to_tflite.py # Конвертация в TFLite + Edge TPU
├── deployment/           # Deployment на Raspberry Pi
│   ├── inference.py      # Real-time детекция
│   ├── logs/             # Логи детекций
│   └── videos/           # Записанные видео
├── utils/                # Утилиты
│   ├── prepare_dataset.py # Подготовка датасета
│   └── augment_data.py   # Аугментация данных
├── models/               # Обученные модели
├── docs/                 # Документация
├── requirements.txt      # Python зависимости (GPU станция)
└── requirements_rpi.txt  # Python зависимости (Raspberry Pi)
```

## 🚀 Быстрый старт

### Этап 1: Подготовка датасета (локальная машина)

```bash
# 1. Клонировать репозиторий
git clone https://github.com/OrDinaD/RSD.git
cd RSD

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Скачать датасет из Google Drive
# Подробная инструкция: docs/DATASET_DOWNLOAD.md
python3 utils/organize_downloaded.py \
    --input ~/Downloads/named \
    --output dataset/raw

# 4. Подготовить датасет (разделить на train/val/test)
python utils/prepare_dataset.py \
    --input dataset/raw \
    --output dataset \
    --split 0.8 0.15 0.05 \
    --validate

# 5. (Опционально) Применить аугментацию
python utils/augment_data.py \
    --input dataset/images/train \
    --output dataset/augmented \
    --factor 2 \
    --weather \
    --lighting
```

### Этап 2: Обучение модели (GPU станция / Google Colab)

```bash
# Обучить YOLOv10n на вашем датасете
python training/train.py \
    --config configs/train_config.yaml \
    --model yolov10n.pt \
    --data configs/dataset.yaml \
    --epochs 200 \
    --batch 16 \
    --device 0

# Или с кастомными параметрами
python training/train.py \
    --data configs/dataset.yaml \
    --epochs 300 \
    --batch 32 \
    --imgsz 640 \
    --device 0
```

**Примечание**: Обучение займет 20-40 часов в зависимости от GPU.

### Этап 3: Оптимизация для Edge TPU (локальная машина)

```bash
# Конвертировать обученную модель в TFLite + Edge TPU
python optimization/convert_to_tflite.py \
    --model training/runs/traffic_signs_yolov10n/weights/best.pt \
    --output models/ \
    --quantize \
    --compile-edgetpu

# Результат: models/best_int8_edgetpu.tflite
```

### Этап 4: Deployment на Raspberry Pi 5

```bash
# На Raspberry Pi:

# 1. Установить Edge TPU Runtime
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std

# 2. Установить Python зависимости
pip install -r requirements_rpi.txt

# 3. Скопировать модель на RPi
scp models/best_int8_edgetpu.tflite pi@raspberry:~/RSD/models/

# 4. Запустить детекцию
python deployment/inference.py \
    --model models/best_int8_edgetpu.tflite \
    --config configs/deployment_config.yaml \
    --camera 0 \
    --display

# 5. (Опционально) Сохранять видео
python deployment/inference.py \
    --model models/best_int8_edgetpu.tflite \
    --save-video
```

## 📊 Ожидаемая производительность

| Конфигурация | FPS | Latency | mAP@0.5 | Потребление |
|--------------|-----|---------|---------|-------------|
| **RPi 5 + Edge TPU** | 18-25 | 40-55ms | 40-45% | 7-9W |
| RPi 5 CPU | 6-8 | 125-165ms | 42-45% | 5-7W |
| RPi 4 + Edge TPU | 12-18 | 55-85ms | 38-42% | 6-8W |

## 🛠️ Требования

### Для обучения (GPU станция / Cloud)
- NVIDIA GPU с минимум 8GB VRAM
- CUDA 11.8+
- Python 3.8+
- 50GB свободного места

### Для deployment (Raspberry Pi)
- **Raspberry Pi 5** (8GB RAM рекомендуется)
- **Google Coral USB Accelerator**
- Raspberry Pi Camera Module или USB webcam (минимум 720p)
- microSD карта 64GB+ (Class 10) или USB SSD
- Активное охлаждение (вентилятор + радиатор)
- Блок питания 5V 5A

## ⚙️ Конфигурация

### Настройка классов дорожных знаков

Отредактируйте `configs/dataset.yaml` и укажите ваши классы:

```yaml
names:
  0: stop_sign
  1: speed_limit_30
  2: speed_limit_50
  # ... добавьте ваши классы
nc: 20  # количество классов
```

### Настройка параметров детекции

Отредактируйте `configs/deployment_config.yaml`:

```yaml
model:
  confidence_threshold: 0.6  # Минимальная уверенность
  iou_threshold: 0.45        # NMS threshold

camera:
  resolution: [1920, 1080]   # Разрешение камеры
  fps: 30                    # FPS камеры
```

## 📈 Мониторинг и логи

```bash
# Просмотр логов детекций
tail -f deployment/logs/detections.log

# Статистика производительности
python utils/analyze_performance.py --logs deployment/logs/
```

## 🔧 Устранение неполадок

### GPU не обнаружен при обучении
```bash
# Проверить CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Edge TPU не работает на Raspberry Pi
```bash
# Проверить подключение Coral
lsusb | grep "Global Unichip"

# Переустановить libedgetpu
sudo apt-get install --reinstall libedgetpu1-std
```

### Низкий FPS на Raspberry Pi
- Убедитесь в активном охлаждении (температура < 75°C)
- Используйте качественный блок питания 5V 5A
- Закройте другие приложения
- Уменьшите разрешение камеры до 720p

## 📚 Дополнительные ресурсы

- 📥 **[Инструкция по скачиванию датасета](docs/DATASET_DOWNLOAD.md)** - как получить данные из Google Drive
- 🚀 **[Быстрый старт](docs/QUICKSTART.md)** - краткое руководство
- 🔧 **[Настройка Raspberry Pi](docs/RASPBERRY_PI_SETUP.md)** - полное руководство по deployment
- [Ultralytics YOLOv10 Docs](https://docs.ultralytics.com/models/yolov10/)
- [Google Coral Docs](https://coral.ai/docs/)
- [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)

