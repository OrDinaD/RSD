#!/usr/bin/env python3
"""
Скрипт подготовки датасета для YOLO формата.
Конвертирует аннотации из различных форматов в YOLO формат и разделяет на train/val/test.

Использование:
    python utils/prepare_dataset.py --input dataset/raw --output dataset --split 0.8 0.15 0.05
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Tuple, List
import random
import yaml
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Подготовка датасета для YOLO')
    parser.add_argument('--input', type=str, required=True, 
                        help='Путь к raw датасету с изображениями и аннотациями')
    parser.add_argument('--output', type=str, default='dataset',
                        help='Путь к выходной директории (по умолчанию: dataset)')
    parser.add_argument('--split', type=float, nargs=3, default=[0.8, 0.15, 0.05],
                        help='Пропорции train/val/test (по умолчанию: 0.8 0.15 0.05)')
    parser.add_argument('--format', type=str, default='yolo', choices=['yolo', 'coco', 'voc'],
                        help='Формат входных аннотаций (по умолчанию: yolo)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed для воспроизводимости')
    parser.add_argument('--validate', action='store_true',
                        help='Валидировать аннотации после конвертации')
    return parser.parse_args()


def get_image_files(input_dir: Path) -> List[Path]:
    """Получить список всех изображений."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = []
    
    for ext in image_extensions:
        images.extend(input_dir.glob(f'*{ext}'))
        images.extend(input_dir.glob(f'*{ext.upper()}'))
    
    return sorted(images)


def split_dataset(files: List[Path], split_ratio: Tuple[float, float, float], 
                  seed: int = 42) -> Tuple[List[Path], List[Path], List[Path]]:
    """Разделить датасет на train/val/test."""
    random.seed(seed)
    random.shuffle(files)
    
    total = len(files)
    train_ratio, val_ratio, test_ratio = split_ratio
    
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]
    
    return train_files, val_files, test_files


def copy_files_with_labels(files: List[Path], input_dir: Path, output_dir: Path, 
                           subset: str, format_type: str = 'yolo'):
    """Копировать изображения и соответствующие аннотации."""
    img_output = output_dir / 'images' / subset
    label_output = output_dir / 'labels' / subset
    
    img_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    missing_labels = []
    
    for img_file in tqdm(files, desc=f'Копирование {subset}'):
        # Копировать изображение
        shutil.copy2(img_file, img_output / img_file.name)
        
        # Найти соответствующую аннотацию
        label_file = input_dir / f"{img_file.stem}.txt"
        
        if label_file.exists():
            if format_type == 'yolo':
                # YOLO формат уже готов, просто копируем
                shutil.copy2(label_file, label_output / f"{img_file.stem}.txt")
            else:
                # TODO: Добавить конвертацию из других форматов (COCO, VOC)
                print(f"⚠️  Конвертация из формата {format_type} пока не реализована")
                shutil.copy2(label_file, label_output / f"{img_file.stem}.txt")
            
            copied_count += 1
        else:
            missing_labels.append(img_file.name)
    
    if missing_labels:
        print(f"⚠️  Найдено {len(missing_labels)} изображений без аннотаций в {subset}")
        # Сохранить список в файл
        with open(output_dir / f'missing_labels_{subset}.txt', 'w') as f:
            f.write('\n'.join(missing_labels))
    
    return copied_count, len(missing_labels)


def validate_annotations(dataset_dir: Path) -> dict:
    """Валидировать YOLO аннотации."""
    stats = {
        'total_files': 0,
        'valid_files': 0,
        'invalid_files': 0,
        'errors': []
    }
    
    for subset in ['train', 'val', 'test']:
        label_dir = dataset_dir / 'labels' / subset
        if not label_dir.exists():
            continue
        
        label_files = list(label_dir.glob('*.txt'))
        stats['total_files'] += len(label_files)
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                valid = True
                for line_num, line in enumerate(lines, 1):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        stats['errors'].append(
                            f"{label_file.name}:{line_num} - Неверное количество значений"
                        )
                        valid = False
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])
                        
                        # Проверка диапазонов (0-1 для YOLO формата)
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                0 < width <= 1 and 0 < height <= 1):
                            stats['errors'].append(
                                f"{label_file.name}:{line_num} - Координаты вне диапазона [0, 1]"
                            )
                            valid = False
                    except ValueError:
                        stats['errors'].append(
                            f"{label_file.name}:{line_num} - Невалидные числовые значения"
                        )
                        valid = False
                
                if valid:
                    stats['valid_files'] += 1
                else:
                    stats['invalid_files'] += 1
                    
            except Exception as e:
                stats['errors'].append(f"{label_file.name} - Ошибка чтения: {str(e)}")
                stats['invalid_files'] += 1
    
    return stats


def main():
    args = parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"❌ Входная директория не существует: {input_dir}")
        return
    
    print("🚀 Начинаем подготовку датасета...")
    print(f"📁 Входная директория: {input_dir}")
    print(f"📂 Выходная директория: {output_dir}")
    print(f"📊 Разделение: train={args.split[0]:.1%}, val={args.split[1]:.1%}, test={args.split[2]:.1%}")
    
    # Получить все изображения
    print("\n🔍 Поиск изображений...")
    image_files = get_image_files(input_dir)
    print(f"✅ Найдено изображений: {len(image_files)}")
    
    if len(image_files) == 0:
        print("❌ Изображения не найдены!")
        return
    
    # Разделить на train/val/test
    print("\n✂️  Разделение датасета...")
    train_files, val_files, test_files = split_dataset(image_files, args.split, args.seed)
    print(f"   Train: {len(train_files)} изображений")
    print(f"   Val:   {len(val_files)} изображений")
    print(f"   Test:  {len(test_files)} изображений")
    
    # Копировать файлы
    print("\n📋 Копирование файлов...")
    
    train_copied, train_missing = copy_files_with_labels(
        train_files, input_dir, output_dir, 'train', args.format
    )
    val_copied, val_missing = copy_files_with_labels(
        val_files, input_dir, output_dir, 'val', args.format
    )
    test_copied, test_missing = copy_files_with_labels(
        test_files, input_dir, output_dir, 'test', args.format
    )
    
    print(f"\n✅ Датасет подготовлен!")
    print(f"   Train: {train_copied} файлов (пропущено: {train_missing})")
    print(f"   Val:   {val_copied} файлов (пропущено: {val_missing})")
    print(f"   Test:  {test_copied} файлов (пропущено: {test_missing})")
    
    # Валидация
    if args.validate:
        print("\n🔍 Валидация аннотаций...")
        stats = validate_annotations(output_dir)
        print(f"   Всего файлов: {stats['total_files']}")
        print(f"   Валидных: {stats['valid_files']}")
        print(f"   Невалидных: {stats['invalid_files']}")
        
        if stats['errors']:
            print(f"\n⚠️  Найдено {len(stats['errors'])} ошибок:")
            for error in stats['errors'][:10]:  # Показать первые 10
                print(f"   - {error}")
            if len(stats['errors']) > 10:
                print(f"   ... и еще {len(stats['errors']) - 10} ошибок")
    
    print(f"\n✨ Готово! Датасет сохранен в {output_dir}")


if __name__ == '__main__':
    main()
