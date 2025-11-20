#!/usr/bin/env python3
"""
Скрипт аугментации данных для улучшения разнообразия датасета.
Применяет различные трансформации специально подобранные для дорожных знаков.

Использование:
    python utils/augment_data.py --input dataset/images/train --output dataset/augmented --factor 2
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from typing import Tuple, List


def parse_args():
    parser = argparse.ArgumentParser(description='Аугментация датасета')
    parser.add_argument('--input', type=str, required=True,
                        help='Путь к изображениям для аугментации')
    parser.add_argument('--labels', type=str, default=None,
                        help='Путь к аннотациям (по умолчанию: автоопределение)')
    parser.add_argument('--output', type=str, required=True,
                        help='Путь к выходной директории')
    parser.add_argument('--factor', type=int, default=2,
                        help='Фактор увеличения датасета (по умолчанию: 2)')
    parser.add_argument('--weather', action='store_true',
                        help='Применить симуляцию погодных условий')
    parser.add_argument('--lighting', action='store_true',
                        help='Применить изменения освещения')
    return parser.parse_args()


def get_augmentation_pipeline(weather: bool = False, lighting: bool = False) -> A.Compose:
    """Создать pipeline аугментаций для дорожных знаков."""
    
    transforms = []
    
    # Базовые геометрические трансформации
    transforms.extend([
        A.RandomRotate90(p=0.3),
        A.Rotate(limit=15, p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        A.HorizontalFlip(p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
    ])
    
    # Цветовые трансформации
    transforms.extend([
        A.OneOf([
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        ], p=0.7),
    ])
    
    # Симуляция погодных условий
    if weather:
        transforms.extend([
            A.OneOf([
                A.RandomRain(
                    slant_lower=-10,
                    slant_upper=10,
                    drop_length=20,
                    drop_width=1,
                    drop_color=(200, 200, 200),
                    blur_value=3,
                    brightness_coefficient=0.9,
                    rain_type='drizzle',
                    p=1.0
                ),
                A.RandomFog(
                    fog_coef_lower=0.2,
                    fog_coef_upper=0.5,
                    alpha_coef=0.1,
                    p=1.0
                ),
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=2,
                    shadow_dimension=5,
                    p=1.0
                ),
            ], p=0.4),
        ])
    
    # Изменения освещения (день/ночь)
    if lighting:
        transforms.extend([
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.3, -0.1),
                    contrast_limit=(-0.2, 0.2),
                    p=1.0
                ),  # Темное время суток
                A.RandomBrightnessContrast(
                    brightness_limit=(0.1, 0.3),
                    contrast_limit=(-0.1, 0.1),
                    p=1.0
                ),  # Яркий день
                A.Posterize(num_bits=4, p=1.0),  # Передержка
            ], p=0.5),
        ])
    
    # Шум и размытие (симуляция движения)
    transforms.extend([
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.3),
        
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
    ])
    
    # Финальные корректировки
    transforms.extend([
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
    ])
    
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3,
            min_area=100
        )
    )


def load_yolo_annotation(label_path: Path) -> Tuple[List[List[float]], List[int]]:
    """Загрузить YOLO аннотации."""
    bboxes = []
    class_labels = []
    
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    bbox = list(map(float, parts[1:]))
                    bboxes.append(bbox)
                    class_labels.append(class_id)
    
    return bboxes, class_labels


def save_yolo_annotation(label_path: Path, bboxes: List[List[float]], class_labels: List[int]):
    """Сохранить YOLO аннотации."""
    with open(label_path, 'w') as f:
        for bbox, class_id in zip(bboxes, class_labels):
            line = f"{class_id} {' '.join(map(str, bbox))}\n"
            f.write(line)


def augment_dataset(input_dir: Path, labels_dir: Path, output_dir: Path,
                    factor: int, weather: bool, lighting: bool):
    """Применить аугментацию ко всему датасету."""
    
    # Создать выходные директории
    output_img_dir = output_dir / 'images'
    output_label_dir = output_dir / 'labels'
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    # Получить pipeline аугментаций
    transform = get_augmentation_pipeline(weather, lighting)
    
    # Получить все изображения
    image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
    
    print(f"🎨 Применяем аугментацию к {len(image_files)} изображениям...")
    print(f"🔄 Фактор увеличения: x{factor}")
    print(f"🌦️  Погодные условия: {'✓' if weather else '✗'}")
    print(f"💡 Изменение освещения: {'✓' if lighting else '✗'}")
    
    total_generated = 0
    
    for img_path in tqdm(image_files, desc='Аугментация'):
        # Загрузить изображение
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Загрузить аннотации
        label_path = labels_dir / f"{img_path.stem}.txt"
        bboxes, class_labels = load_yolo_annotation(label_path)
        
        # Сохранить оригинал
        cv2.imwrite(
            str(output_img_dir / img_path.name),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )
        if bboxes:
            save_yolo_annotation(output_label_dir / f"{img_path.stem}.txt", bboxes, class_labels)
        
        # Создать аугментированные версии
        for i in range(factor - 1):
            try:
                transformed = transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                
                aug_image = transformed['image']
                aug_bboxes = transformed['bboxes']
                aug_labels = transformed['class_labels']
                
                # Сохранить аугментированное изображение
                aug_img_name = f"{img_path.stem}_aug{i+1}{img_path.suffix}"
                cv2.imwrite(
                    str(output_img_dir / aug_img_name),
                    cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                )
                
                # Сохранить аугментированные аннотации
                if aug_bboxes:
                    save_yolo_annotation(
                        output_label_dir / f"{img_path.stem}_aug{i+1}.txt",
                        aug_bboxes,
                        aug_labels
                    )
                
                total_generated += 1
                
            except Exception as e:
                print(f"⚠️  Ошибка при аугментации {img_path.name}: {str(e)}")
                continue
    
    print(f"\n✅ Аугментация завершена!")
    print(f"   Исходных изображений: {len(image_files)}")
    print(f"   Сгенерировано новых: {total_generated}")
    print(f"   Итого: {len(image_files) + total_generated}")


def main():
    args = parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Автоопределение пути к аннотациям
    if args.labels:
        labels_dir = Path(args.labels)
    else:
        # Предполагаем структуру dataset/images/train -> dataset/labels/train
        labels_dir = input_dir.parent.parent / 'labels' / input_dir.name
    
    if not input_dir.exists():
        print(f"❌ Входная директория не существует: {input_dir}")
        return
    
    if not labels_dir.exists():
        print(f"❌ Директория с аннотациями не существует: {labels_dir}")
        return
    
    print("🚀 Начинаем аугментацию датасета...")
    
    augment_dataset(
        input_dir,
        labels_dir,
        output_dir,
        args.factor,
        args.weather,
        args.lighting
    )
    
    print(f"\n✨ Готово! Аугментированный датасет сохранен в {output_dir}")


if __name__ == '__main__':
    main()
