#!/usr/bin/env python3
"""
Inference скрипт для Raspberry Pi с Google Coral Edge TPU.
Real-time детекция дорожных знаков с камеры.

Требования:
- Raspberry Pi 5 (рекомендуется)
- Google Coral USB Accelerator
- Pi Camera или USB webcam
- TFLite модель скомпилированная для Edge TPU

Использование:
    python deployment/inference.py --model models/traffic_signs_edgetpu.tflite --config configs/deployment_config.yaml
"""

import argparse
import time
import yaml
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2

# PyCoral для Edge TPU
try:
    from pycoral.adapters import common
    from pycoral.utils.edgetpu import make_interpreter
    CORAL_AVAILABLE = True
except ImportError:
    print("⚠️  PyCoral не установлен. Используется CPU режим.")
    CORAL_AVAILABLE = False
    import tflite_runtime.interpreter as tflite


def parse_args():
    parser = argparse.ArgumentParser(description='Inference на Raspberry Pi с Edge TPU')
    parser.add_argument('--model', type=str, required=True,
                        help='Путь к TFLite модели (Edge TPU или обычная)')
    parser.add_argument('--config', type=str, default='configs/deployment_config.yaml',
                        help='Путь к конфигурации deployment')
    parser.add_argument('--camera', type=int, default=0,
                        help='ID камеры (0 для Pi Camera, 1+ для USB)')
    parser.add_argument('--source', type=str, default=None,
                        help='Видео файл вместо камеры (опционально)')
    parser.add_argument('--save-video', action='store_true',
                        help='Сохранять видео с детекциями')
    parser.add_argument('--display', action='store_true', default=True,
                        help='Отображать видео (требует дисплей)')
    parser.add_argument('--fps-target', type=int, default=20,
                        help='Целевой FPS')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Загрузить конфигурацию из YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_labels(config: dict) -> List[str]:
    """Загрузить названия классов из dataset config."""
    dataset_config_path = 'configs/dataset.yaml'
    
    if Path(dataset_config_path).exists():
        with open(dataset_config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
            return [dataset_config['names'][i] for i in range(dataset_config['nc'])]
    else:
        # Использовать дефолтные названия
        return [f"class_{i}" for i in range(20)]


class TrafficSignDetector:
    """Детектор дорожных знаков на Edge TPU."""
    
    def __init__(self, model_path: str, use_edgetpu: bool = True):
        self.model_path = model_path
        self.use_edgetpu = use_edgetpu and CORAL_AVAILABLE
        
        print(f"🔧 Инициализация детектора...")
        print(f"   Модель: {model_path}")
        print(f"   Edge TPU: {'✓' if self.use_edgetpu else '✗ (CPU режим)'}")
        
        # Загрузить модель
        if self.use_edgetpu:
            self.interpreter = make_interpreter(model_path)
        else:
            self.interpreter = tflite.Interpreter(model_path=model_path)
        
        self.interpreter.allocate_tensors()
        
        # Получить input/output детали
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()
        
        # Размеры входа
        self.input_shape = self.input_details['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        
        # Проверить квантизацию
        self.is_quantized = self.input_details['dtype'] == np.uint8
        
        print(f"   Размер входа: {self.input_width}x{self.input_height}")
        print(f"   Квантизация: {'INT8' if self.is_quantized else 'FP32'}")
        print(f"✅ Детектор готов!")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Предобработка изображения."""
        # Resize с сохранением aspect ratio (letterbox)
        img_resized = cv2.resize(image, (self.input_width, self.input_height))
        
        if self.is_quantized:
            # Для INT8 модели
            img_processed = img_resized.astype(np.uint8)
        else:
            # Для FP32 модели
            img_processed = img_resized.astype(np.float32) / 255.0
        
        # Добавить batch dimension
        img_processed = np.expand_dims(img_processed, axis=0)
        
        return img_processed
    
    def detect(self, image: np.ndarray, confidence_threshold: float = 0.6) -> List[dict]:
        """Выполнить детекцию."""
        # Предобработка
        input_data = self.preprocess(image)
        
        # Inference
        self.interpreter.set_tensor(self.input_details['index'], input_data)
        self.interpreter.invoke()
        
        # Получить результаты
        # Формат зависит от модели - адаптировать под ваш YOLO output
        detections = []
        
        # Для YOLO обычно есть boxes, scores, classes
        # Адаптируйте индексы в зависимости от вашей модели
        try:
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
            
            # Фильтровать по confidence
            for i in range(len(scores)):
                if scores[i] >= confidence_threshold:
                    detections.append({
                        'bbox': boxes[i].tolist(),
                        'score': float(scores[i]),
                        'class_id': int(classes[i])
                    })
        except Exception as e:
            print(f"⚠️  Ошибка парсинга output: {e}")
        
        return detections


class FPSCounter:
    """Счетчик FPS."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps = []
    
    def update(self):
        """Обновить timestamp."""
        current_time = time.time()
        self.timestamps.append(current_time)
        
        # Оставить только последние N timestamp'ов
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
    
    def get_fps(self) -> float:
        """Получить текущий FPS."""
        if len(self.timestamps) < 2:
            return 0.0
        
        time_diff = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / time_diff if time_diff > 0 else 0.0


def draw_detections(image: np.ndarray, detections: List[dict], 
                    labels: List[str], fps: float) -> np.ndarray:
    """Нарисовать детекции на изображении."""
    img_height, img_width = image.shape[:2]
    
    # Нарисовать FPS
    cv2.putText(
        image,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )
    
    # Нарисовать детекции
    for det in detections:
        # Конвертировать bbox в координаты изображения
        # Формат зависит от вашего YOLO output - адаптировать
        bbox = det['bbox']
        
        # Предположим формат [ymin, xmin, ymax, xmax] нормализованный
        y1 = int(bbox[0] * img_height)
        x1 = int(bbox[1] * img_width)
        y2 = int(bbox[2] * img_height)
        x2 = int(bbox[3] * img_width)
        
        # Нарисовать bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Нарисовать label
        class_id = det['class_id']
        score = det['score']
        label = labels[class_id] if class_id < len(labels) else f"class_{class_id}"
        text = f"{label}: {score:.2f}"
        
        cv2.putText(
            image,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
    
    return image


def main():
    args = parse_args()
    
    # Проверить модель
    if not Path(args.model).exists():
        print(f"❌ Модель не найдена: {args.model}")
        return
    
    # Загрузить конфигурацию
    config = load_config(args.config) if Path(args.config).exists() else {}
    labels = load_labels(config)
    
    print("\n" + "="*60)
    print("🚀 ЗАПУСК ДЕТЕКЦИИ ДОРОЖНЫХ ЗНАКОВ")
    print("="*60)
    
    # Инициализировать детектор
    use_edgetpu = config.get('edge_tpu', {}).get('enabled', True)
    detector = TrafficSignDetector(args.model, use_edgetpu=use_edgetpu)
    
    # Настроить камеру/источник
    if args.source:
        print(f"📹 Источник: {args.source}")
        cap = cv2.VideoCapture(args.source)
    else:
        print(f"📷 Камера ID: {args.camera}")
        cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("❌ Не удалось открыть камеру/видео")
        return
    
    # Настроить параметры камеры
    camera_config = config.get('camera', {})
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.get('resolution', [1920, 1080])[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.get('resolution', [1920, 1080])[1])
    cap.set(cv2.CAP_PROP_FPS, camera_config.get('fps', 30))
    
    # Настроить сохранение видео
    video_writer = None
    if args.save_video:
        output_path = Path('deployment/videos') / f"detection_{int(time.time())}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video_writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (frame_width, frame_height)
        )
        print(f"💾 Сохранение видео: {output_path}")
    
    # FPS counter
    fps_counter = FPSCounter()
    
    # Параметры детекции
    confidence_threshold = config.get('model', {}).get('confidence_threshold', 0.6)
    
    print("\n▶️  Начало детекции (нажмите 'q' для выхода)...")
    print("-" * 60)
    
    frame_count = 0
    detection_count = 0
    
    try:
        while True:
            # Захватить кадр
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Не удалось получить кадр")
                break
            
            frame_count += 1
            
            # Детекция
            start_time = time.time()
            detections = detector.detect(frame, confidence_threshold)
            inference_time = (time.time() - start_time) * 1000  # в миллисекундах
            
            detection_count += len(detections)
            
            # Обновить FPS
            fps_counter.update()
            current_fps = fps_counter.get_fps()
            
            # Визуализация
            if args.display or args.save_video:
                frame_with_detections = draw_detections(
                    frame.copy(),
                    detections,
                    labels,
                    current_fps
                )
                
                # Добавить inference time
                cv2.putText(
                    frame_with_detections,
                    f"Inference: {inference_time:.1f}ms",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )
                
                if args.display:
                    cv2.imshow('Traffic Sign Detection', frame_with_detections)
                
                if video_writer:
                    video_writer.write(frame_with_detections)
            
            # Вывести статистику каждые 30 кадров
            if frame_count % 30 == 0:
                print(f"Кадры: {frame_count} | FPS: {current_fps:.1f} | "
                      f"Inference: {inference_time:.1f}ms | Детекций: {detection_count}")
            
            # Выход по 'q'
            if args.display and cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except KeyboardInterrupt:
        print("\n⚠️  Прервано пользователем")
    
    finally:
        # Освободить ресурсы
        cap.release()
        if video_writer:
            video_writer.release()
        if args.display:
            cv2.destroyAllWindows()
        
        # Финальная статистика
        print("\n" + "="*60)
        print("📊 СТАТИСТИКА")
        print("="*60)
        print(f"Всего кадров: {frame_count}")
        print(f"Средний FPS: {fps_counter.get_fps():.1f}")
        print(f"Всего детекций: {detection_count}")
        print(f"Детекций на кадр: {detection_count/frame_count if frame_count > 0 else 0:.2f}")
        
        print("\n✨ Завершено!")


if __name__ == '__main__':
    main()
