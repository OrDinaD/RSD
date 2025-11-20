#!/usr/bin/env python3
"""
Скрипт обучения YOLOv10n модели для распознавания дорожных знаков.
Использовать на GPU-станции (не на Raspberry Pi).

Использование:
    python training/train.py --config configs/train_config.yaml
    python training/train.py --data configs/dataset.yaml --epochs 200 --batch 16
"""

import argparse
import os
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Обучение YOLOv10 для дорожных знаков')
    parser.add_argument('--config', type=str, default=None,
                        help='Путь к конфигурационному файлу')
    parser.add_argument('--model', type=str, default='yolov10n.pt',
                        help='Модель для обучения (по умолчанию: yolov10n.pt)')
    parser.add_argument('--data', type=str, default='configs/dataset.yaml',
                        help='Путь к конфигурации датасета')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Количество эпох обучения')
    parser.add_argument('--batch', type=int, default=16,
                        help='Размер батча')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Размер изображения')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device (0, 1, 2...) или cpu')
    parser.add_argument('--project', type=str, default='training/runs',
                        help='Директория для сохранения результатов')
    parser.add_argument('--name', type=str, default=None,
                        help='Имя эксперимента')
    parser.add_argument('--resume', type=str, default=None,
                        help='Путь к checkpoint для продолжения обучения')
    parser.add_argument('--pretrained', action='store_true',
                        help='Использовать предобученные веса')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Загрузить конфигурацию из YAML файла."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_training(args) -> dict:
    """Подготовить параметры обучения."""
    
    # Если указан конфигурационный файл, загрузить его
    if args.config and Path(args.config).exists():
        print(f"📋 Загрузка конфигурации из {args.config}")
        config = load_config(args.config)
        
        # Параметры из командной строки имеют приоритет
        training_args = {
            'data': args.data or config.get('data', 'configs/dataset.yaml'),
            'epochs': args.epochs or config.get('epochs', 200),
            'batch': args.batch or config.get('batch', 16),
            'imgsz': args.imgsz or config.get('imgsz', 640),
            'device': args.device or config.get('device', '0'),
            'project': args.project or config.get('project', 'training/runs'),
            'name': args.name or config.get('name', None),
        }
        
        # Добавить остальные параметры из конфига
        for key in ['optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay',
                    'patience', 'save', 'save_period', 'plots', 'verbose']:
            if key in config:
                training_args[key] = config[key]
        
    else:
        # Использовать параметры из командной строки
        training_args = {
            'data': args.data,
            'epochs': args.epochs,
            'batch': args.batch,
            'imgsz': args.imgsz,
            'device': args.device,
            'project': args.project,
            'name': args.name,
        }
    
    # Автоматическое имя эксперимента если не указано
    if not training_args.get('name'):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        training_args['name'] = f'traffic_signs_{timestamp}'
    
    return training_args


def check_gpu():
    """Проверить доступность GPU."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU доступен: {gpu_name}")
        print(f"   Количество GPU: {gpu_count}")
        print(f"   CUDA версия: {torch.version.cuda}")
        return True
    else:
        print("⚠️  GPU не обнаружен. Обучение будет на CPU (очень медленно!)")
        return False


def train_model(model_name: str, training_args: dict, resume: str = None, 
                pretrained: bool = True):
    """Обучить YOLO модель."""
    
    print("\n" + "="*60)
    print("🚀 НАЧАЛО ОБУЧЕНИЯ YOLOv10 ДЛЯ РАСПОЗНАВАНИЯ ДОРОЖНЫХ ЗНАКОВ")
    print("="*60)
    
    # Вывести параметры обучения
    print("\n📊 Параметры обучения:")
    for key, value in training_args.items():
        print(f"   {key}: {value}")
    
    print(f"\n🔧 Модель: {model_name}")
    print(f"🔧 Предобученные веса: {'Да' if pretrained else 'Нет'}")
    
    # Проверить GPU
    check_gpu()
    
    # Загрузить модель
    print(f"\n📦 Загрузка модели...")
    
    if resume:
        print(f"   Продолжение обучения с checkpoint: {resume}")
        model = YOLO(resume)
    else:
        if pretrained:
            print(f"   Загрузка предобученной модели: {model_name}")
            model = YOLO(model_name)
        else:
            print(f"   Обучение с нуля")
            # Для обучения с нуля используем .yaml конфигурацию
            config_name = model_name.replace('.pt', '.yaml')
            model = YOLO(config_name)
    
    # Начать обучение
    print("\n🎯 Запуск обучения...")
    print("-" * 60)
    
    try:
        results = model.train(**training_args)
        
        print("\n" + "="*60)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("="*60)
        
        # Информация о результатах
        save_dir = Path(training_args['project']) / training_args['name']
        print(f"\n📁 Результаты сохранены в: {save_dir}")
        print(f"   - Лучшая модель: {save_dir / 'weights' / 'best.pt'}")
        print(f"   - Последняя модель: {save_dir / 'weights' / 'last.pt'}")
        print(f"   - Метрики: {save_dir / 'results.csv'}")
        print(f"   - Графики: {save_dir / 'results.png'}")
        
        # Вывести финальные метрики
        if hasattr(results, 'results_dict'):
            print("\n📈 Финальные метрики:")
            metrics = results.results_dict
            if 'metrics/mAP50(B)' in metrics:
                print(f"   mAP@0.5: {metrics['metrics/mAP50(B)']:.4f}")
            if 'metrics/mAP50-95(B)' in metrics:
                print(f"   mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.4f}")
        
        return results
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ ОШИБКА ПРИ ОБУЧЕНИИ: {str(e)}")
        print("="*60)
        raise


def validate_model(model_path: str, data_config: str):
    """Валидировать обученную модель."""
    print(f"\n🔍 Валидация модели: {model_path}")
    
    model = YOLO(model_path)
    results = model.val(data=data_config)
    
    print("\n📊 Результаты валидации:")
    print(f"   mAP@0.5: {results.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"   Precision: {results.box.mp:.4f}")
    print(f"   Recall: {results.box.mr:.4f}")
    
    return results


def main():
    args = parse_args()
    
    # Проверить существование конфигурации датасета
    if not Path(args.data).exists():
        print(f"❌ Конфигурация датасета не найдена: {args.data}")
        print("   Создайте файл configs/dataset.yaml или укажите правильный путь")
        return
    
    # Подготовить параметры обучения
    training_args = setup_training(args)
    
    # Обучить модель
    try:
        results = train_model(
            model_name=args.model,
            training_args=training_args,
            resume=args.resume,
            pretrained=args.pretrained
        )
        
        # Валидация лучшей модели
        best_model = Path(training_args['project']) / training_args['name'] / 'weights' / 'best.pt'
        if best_model.exists():
            validate_model(str(best_model), training_args['data'])
        
        print("\n✨ Процесс завершен! Модель готова к оптимизации для Edge TPU.")
        print(f"💡 Следующий шаг: python optimization/convert_to_tflite.py --model {best_model}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Обучение прервано пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {str(e)}")
        raise


if __name__ == '__main__':
    main()
