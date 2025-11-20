#!/usr/bin/env python3
"""
Скрипт конвертации обученной YOLOv10 модели в TensorFlow Lite формат
с квантизацией INT8 для Google Coral Edge TPU.

Pipeline: PyTorch (.pt) → ONNX (.onnx) → TensorFlow (.pb) → TFLite (.tflite) → Edge TPU (.tflite)

Использование:
    python optimization/convert_to_tflite.py --model models/best.pt --output models/
"""

import argparse
import os
import subprocess
from pathlib import Path
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import onnx
from onnx_tf.backend import prepare


def parse_args():
    parser = argparse.ArgumentParser(description='Конвертация YOLO в TFLite для Edge TPU')
    parser.add_argument('--model', type=str, required=True,
                        help='Путь к обученной PyTorch модели (.pt)')
    parser.add_argument('--output', type=str, default='models/',
                        help='Директория для сохранения конвертированных моделей')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Размер входного изображения')
    parser.add_argument('--quantize', action='store_true', default=True,
                        help='Применить INT8 квантизацию')
    parser.add_argument('--compile-edgetpu', action='store_true',
                        help='Скомпилировать для Edge TPU (требует edgetpu_compiler)')
    parser.add_argument('--representative-dataset', type=str, default=None,
                        help='Путь к representative dataset для квантизации')
    return parser.parse_args()


def export_to_onnx(model_path: str, output_dir: Path, imgsz: int) -> Path:
    """Шаг 1: Экспорт PyTorch модели в ONNX."""
    print("\n" + "="*60)
    print("ШАГ 1: PyTorch → ONNX")
    print("="*60)
    
    model = YOLO(model_path)
    onnx_path = output_dir / f"{Path(model_path).stem}.onnx"
    
    print(f"📦 Загрузка модели: {model_path}")
    print(f"🔄 Экспорт в ONNX...")
    
    try:
        # Экспорт с использованием Ultralytics
        export_path = model.export(
            format='onnx',
            imgsz=imgsz,
            simplify=True,  # Упрощение графа
            dynamic=False,  # Статический размер для Edge TPU
            opset=12  # ONNX opset версия
        )
        
        # Переместить в output директорию если нужно
        if Path(export_path) != onnx_path:
            Path(export_path).rename(onnx_path)
        
        print(f"✅ ONNX модель сохранена: {onnx_path}")
        
        # Проверить ONNX модель
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX модель валидна")
        
        return onnx_path
        
    except Exception as e:
        print(f"❌ Ошибка при экспорте в ONNX: {str(e)}")
        raise


def onnx_to_tensorflow(onnx_path: Path, output_dir: Path) -> Path:
    """Шаг 2: Конвертация ONNX в TensorFlow SavedModel."""
    print("\n" + "="*60)
    print("ШАГ 2: ONNX → TensorFlow SavedModel")
    print("="*60)
    
    tf_model_dir = output_dir / f"{onnx_path.stem}_saved_model"
    
    try:
        print(f"🔄 Загрузка ONNX модели...")
        onnx_model = onnx.load(str(onnx_path))
        
        print(f"🔄 Конвертация в TensorFlow...")
        tf_rep = prepare(onnx_model)
        
        print(f"💾 Сохранение SavedModel...")
        tf_rep.export_graph(str(tf_model_dir))
        
        print(f"✅ TensorFlow SavedModel сохранена: {tf_model_dir}")
        return tf_model_dir
        
    except Exception as e:
        print(f"❌ Ошибка при конвертации в TensorFlow: {str(e)}")
        print("💡 Попробуйте альтернативный метод через tf2onnx")
        raise


def create_representative_dataset(dataset_path: str = None, imgsz: int = 640):
    """Создать representative dataset для квантизации."""
    
    def representative_data_gen():
        """Генератор данных для калибровки квантизации."""
        # Если указан датасет, использовать его
        if dataset_path and Path(dataset_path).exists():
            import cv2
            image_files = list(Path(dataset_path).glob('*.jpg'))[:100]
            
            for img_path in image_files:
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (imgsz, imgsz))
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=0)
                yield [img]
        else:
            # Использовать случайные данные
            for _ in range(100):
                img = np.random.rand(1, imgsz, imgsz, 3).astype(np.float32)
                yield [img]
    
    return representative_data_gen


def tensorflow_to_tflite(tf_model_dir: Path, output_dir: Path, quantize: bool,
                          representative_dataset_path: str = None,
                          imgsz: int = 640) -> Path:
    """Шаг 3: Конвертация TensorFlow SavedModel в TFLite с квантизацией."""
    print("\n" + "="*60)
    print("ШАГ 3: TensorFlow SavedModel → TFLite (INT8 Quantization)")
    print("="*60)
    
    tflite_path = output_dir / f"{tf_model_dir.stem}_int8.tflite"
    
    try:
        # Загрузить SavedModel
        print(f"📦 Загрузка SavedModel...")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_dir))
        
        if quantize:
            print(f"🔧 Настройка INT8 квантизации...")
            
            # Настройки для Edge TPU
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            
            # Representative dataset для калибровки
            print(f"📊 Создание representative dataset...")
            representative_data = create_representative_dataset(
                representative_dataset_path, imgsz
            )
            converter.representative_dataset = representative_data
            
            print(f"⚙️  Применение квантизации...")
        
        # Конвертировать
        print(f"🔄 Конвертация в TFLite...")
        tflite_model = converter.convert()
        
        # Сохранить
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Информация о размере
        original_size = sum(
            f.stat().st_size for f in tf_model_dir.rglob('*') if f.is_file()
        ) / (1024 * 1024)
        tflite_size = tflite_path.stat().st_size / (1024 * 1024)
        
        print(f"✅ TFLite модель сохранена: {tflite_path}")
        print(f"📊 Размер SavedModel: {original_size:.2f} MB")
        print(f"📊 Размер TFLite: {tflite_size:.2f} MB")
        print(f"📊 Сжатие: {(1 - tflite_size/original_size)*100:.1f}%")
        
        return tflite_path
        
    except Exception as e:
        print(f"❌ Ошибка при конвертации в TFLite: {str(e)}")
        raise


def compile_for_edgetpu(tflite_path: Path, output_dir: Path) -> Path:
    """Шаг 4: Компиляция TFLite модели для Edge TPU."""
    print("\n" + "="*60)
    print("ШАГ 4: TFLite → Edge TPU")
    print("="*60)
    
    edgetpu_path = output_dir / f"{tflite_path.stem}_edgetpu.tflite"
    
    try:
        # Проверить наличие edgetpu_compiler
        result = subprocess.run(
            ['edgetpu_compiler', '--version'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("⚠️  edgetpu_compiler не найден!")
            print("   Установите: https://coral.ai/docs/edgetpu/compiler/")
            return None
        
        print(f"🔧 Найден edgetpu_compiler")
        print(f"🔄 Компиляция для Edge TPU...")
        
        # Компилировать модель
        cmd = [
            'edgetpu_compiler',
            str(tflite_path),
            '-o', str(output_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Edge TPU модель скомпилирована: {edgetpu_path}")
            
            # Вывести информацию о компиляции
            if result.stdout:
                print("\n📋 Информация о компиляции:")
                print(result.stdout)
            
            return edgetpu_path
        else:
            print(f"❌ Ошибка компиляции:")
            print(result.stderr)
            return None
            
    except FileNotFoundError:
        print("⚠️  edgetpu_compiler не установлен")
        print("   Установите: https://coral.ai/docs/edgetpu/compiler/")
        return None
    except Exception as e:
        print(f"❌ Ошибка при компиляции для Edge TPU: {str(e)}")
        return None


def main():
    args = parse_args()
    
    model_path = Path(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        print(f"❌ Модель не найдена: {model_path}")
        return
    
    print("\n" + "="*60)
    print("🚀 КОНВЕРТАЦИЯ YOLO МОДЕЛИ ДЛЯ EDGE TPU")
    print("="*60)
    print(f"📦 Входная модель: {model_path}")
    print(f"📂 Выходная директория: {output_dir}")
    print(f"🔧 Квантизация INT8: {'Да' if args.quantize else 'Нет'}")
    print(f"🔧 Компиляция Edge TPU: {'Да' if args.compile_edgetpu else 'Нет'}")
    
    try:
        # Шаг 1: PyTorch → ONNX
        onnx_path = export_to_onnx(str(model_path), output_dir, args.imgsz)
        
        # Шаг 2: ONNX → TensorFlow
        tf_model_dir = onnx_to_tensorflow(onnx_path, output_dir)
        
        # Шаг 3: TensorFlow → TFLite (с квантизацией)
        tflite_path = tensorflow_to_tflite(
            tf_model_dir,
            output_dir,
            args.quantize,
            args.representative_dataset,
            args.imgsz
        )
        
        # Шаг 4: TFLite → Edge TPU (опционально)
        if args.compile_edgetpu:
            edgetpu_path = compile_for_edgetpu(tflite_path, output_dir)
            
            if edgetpu_path:
                print("\n" + "="*60)
                print("✅ ВСЕ ЭТАПЫ ЗАВЕРШЕНЫ УСПЕШНО!")
                print("="*60)
                print(f"\n📦 Итоговая модель для Raspberry Pi:")
                print(f"   {edgetpu_path}")
                print(f"\n💡 Следующий шаг:")
                print(f"   1. Скопируйте {edgetpu_path.name} на Raspberry Pi")
                print(f"   2. Запустите: python deployment/inference.py --model {edgetpu_path.name}")
        else:
            print("\n" + "="*60)
            print("✅ КОНВЕРТАЦИЯ ЗАВЕРШЕНА!")
            print("="*60)
            print(f"\n📦 TFLite модель (без Edge TPU):")
            print(f"   {tflite_path}")
            print(f"\n💡 Для компиляции под Edge TPU:")
            print(f"   python optimization/convert_to_tflite.py --model {model_path} --compile-edgetpu")
            
    except Exception as e:
        print(f"\n❌ Процесс конвертации прерван: {str(e)}")
        raise


if __name__ == '__main__':
    main()
