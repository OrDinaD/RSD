# Краткое руководство по использованию RSD

## 🎯 Для быстрого старта

### 1️⃣ Подготовка данных (на вашей машине)

```bash
# Разместите 30k изображений в dataset/raw/
# Формат: изображение.jpg + изображение.txt (YOLO аннотации)

python utils/prepare_dataset.py --input dataset/raw --output dataset --validate
```

### 2️⃣ Обучение модели (на GPU - Colab/AWS/локально)

```bash
# Простой вариант
python training/train.py --data configs/dataset.yaml --epochs 200 --batch 16

# Полный контроль
python training/train.py --config configs/train_config.yaml
```

**Ожидайте**: 20-40 часов на GPU (зависит от железа)

### 3️⃣ Конвертация для Edge TPU (на вашей машине)

```bash
python optimization/convert_to_tflite.py \
    --model training/runs/traffic_signs_yolov10n/weights/best.pt \
    --output models/ \
    --compile-edgetpu
```

**Результат**: `models/best_int8_edgetpu.tflite`

### 4️⃣ Deployment на Raspberry Pi

```bash
# На RPi (после setup из docs/RASPBERRY_PI_SETUP.md):
python deployment/inference.py \
    --model models/best_int8_edgetpu.tflite \
    --config configs/deployment_config.yaml
```

**Результат**: Real-time детекция 18-25 FPS

---

## 📁 Основные файлы для редактирования

1. **`configs/dataset.yaml`** - ваши классы дорожных знаков
2. **`configs/train_config.yaml`** - параметры обучения
3. **`configs/deployment_config.yaml`** - настройки камеры и детекции

---

## ⚡ Частые команды

```bash
# Валидация датасета
python utils/prepare_dataset.py --input dataset/raw --validate

# Аугментация (увеличить датасет x2)
python utils/augment_data.py --input dataset/images/train --factor 2 --weather

# Продолжить обучение с checkpoint
python training/train.py --resume training/runs/exp1/weights/last.pt

# Тест модели на видео
python deployment/inference.py --model models/best.tflite --source test_video.mp4

# Сохранить видео с детекциями
python deployment/inference.py --model models/best.tflite --save-video
```

---

## 🔧 Решение проблем

| Проблема | Решение |
|----------|---------|
| GPU не найден при обучении | `nvidia-smi`, проверить CUDA |
| Edge TPU не работает | `lsusb \| grep Unichip`, переподключить USB |
| Низкий FPS (<15) | Проверить температуру, включить охлаждение |
| Модель не конвертируется | Установить `onnx`, `tf2onnx`, `edgetpu_compiler` |

---

## 📊 Ожидаемые метрики

**После обучения на 30k изображений:**
- mAP@0.5: 40-45%
- mAP@0.5:0.95: 35-40%
- Precision: 45-50%
- Recall: 40-45%

**На Raspberry Pi 5 + Coral:**
- FPS: 18-25
- Latency: 40-55ms
- CPU: 30-40%
- Temp: 65-75°C

---

## 🚀 Следующие шаги

1. ✅ Подготовить и валидировать датасет
2. ✅ Настроить классы в `configs/dataset.yaml`
3. ✅ Обучить модель на GPU
4. ✅ Конвертировать в TFLite + Edge TPU
5. ✅ Настроить Raspberry Pi (см. `docs/RASPBERRY_PI_SETUP.md`)
6. ✅ Запустить real-time детекцию
7. 🔄 Тестировать и улучшать

---

**Важно**: Всегда тестируйте в безопасных условиях перед использованием на дороге!
