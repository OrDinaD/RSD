#!/usr/bin/env python3
"""
Упрощенный скрипт для организации скачанных дорожных знаков.
Работает с уже скачанными файлами из папки Google Drive.

Использование:
    1. Скачайте файлы из Google Drive вручную
    2. python utils/organize_downloaded.py --input путь/к/скачанным/файлам --output dataset/raw
"""

import os
import argparse
import re
import shutil
from pathlib import Path
from typing import Dict
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Организация скачанных дорожных знаков')
    parser.add_argument('--input', type=str, required=True,
                        help='Путь к скачанным файлам')
    parser.add_argument('--output', type=str, default='dataset/raw',
                        help='Директория для сохранения организованных файлов')
    return parser.parse_args()


# Маппинг названий знаков
SIGN_PATTERNS = {
    # Ограничения скорости
    r'(?:speed.*?limit|limit.*?speed|ограничение.*?скорост).*?20': 'speed_limit_20',
    r'(?:speed.*?limit|limit.*?speed|ограничение.*?скорост).*?30': 'speed_limit_30',
    r'(?:speed.*?limit|limit.*?speed|ограничение.*?скорост).*?50': 'speed_limit_50',
    r'(?:speed.*?limit|limit.*?speed|ограничение.*?скорост).*?60': 'speed_limit_60',
    r'(?:speed.*?limit|limit.*?speed|ограничение.*?скорост).*?70': 'speed_limit_70',
    r'(?:speed.*?limit|limit.*?speed|ограничение.*?скорост).*?80': 'speed_limit_80',
    r'(?:speed.*?limit|limit.*?speed|ограничение.*?скорост).*?100': 'speed_limit_100',
    r'(?:speed.*?limit|limit.*?speed|ограничение.*?скорост).*?120': 'speed_limit_120',
    
    # Запрещающие знаки
    r'(?:no.*?entry|въезд.*?запрещ|3\.1)': 'no_entry',
    r'(?:stop|стоп|2\.5)': 'stop_sign',
    r'(?:no.*?overtaking|обгон.*?запрещ|3\.20)': 'no_overtaking',
    r'(?:no.*?parking|стоянка.*?запрещ|3\.28)': 'no_parking',
    
    # Предупреждающие знаки
    r'(?:pedestrian|пешеход|1\.22)': 'pedestrian_crossing',
    r'(?:children|дети|1\.23)': 'children_crossing',
    r'(?:bicycle|велосипед|1\.24)': 'bicycle_crossing',
    r'(?:slippery|скольз|1\.15)': 'slippery_road',
    r'(?:work|road.*?work|ремонт|дорож.*?работ|1\.25)': 'construction',
    r'(?:traffic.*?light|светофор|1\.8)': 'traffic_light',
    r'(?:curve|dangerous.*?curve|опасн.*?поворот|1\.11)': 'dangerous_curve',
    
    # Знаки приоритета
    r'(?:priority.*?road|главн.*?дорог|2\.1)': 'priority_road',
    r'(?:yield|уступ.*?дорог|2\.4)': 'yield_sign',
    
    # Предписывающие знаки
    r'(?:roundabout|круговое.*?движ|4\.3)': 'roundabout',
    r'(?:turn.*?left|поворот.*?налево|4\.1\.1)': 'turn_left',
    r'(?:turn.*?right|поворот.*?направо|4\.1\.2)': 'turn_right',
    r'(?:straight|ahead|только.*?прямо|4\.1\.3)': 'go_straight',
    
    # Информационные знаки
    r'(?:parking|парковка|стоянка|6\.4)': 'parking',
    r'(?:crosswalk|переход|5\.19)': 'pedestrian_crossing',
}


def normalize_filename(filename: str) -> str:
    """Нормализовать имя файла."""
    name = Path(filename).stem.lower()
    name = re.sub(r'[^\w\s-]', '_', name)
    name = re.sub(r'[-\s]+', '_', name)
    return name


def detect_sign_type(filename: str) -> str:
    """Определить тип знака по имени файла."""
    normalized = normalize_filename(filename)
    
    # Попробовать найти по паттернам
    for pattern, sign_type in SIGN_PATTERNS.items():
        if re.search(pattern, normalized, re.IGNORECASE):
            return sign_type
    
    # Если не нашли, попробовать извлечь из структуры имени
    # Например: "01_speed_limit_30.jpg"
    parts = normalized.split('_')
    if len(parts) >= 2:
        # Убрать числовой префикс если есть
        if parts[0].isdigit():
            return '_'.join(parts[1:])
        return '_'.join(parts)
    
    return 'unknown'


def organize_files(input_dir: Path, output_dir: Path) -> Dict[str, int]:
    """Организовать файлы по типам знаков."""
    print(f"\n🔄 Организация файлов...")
    print(f"   Входная папка: {input_dir}")
    print(f"   Выходная папка: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # Найти все изображения рекурсивно
    all_files = []
    for ext in image_extensions:
        all_files.extend(input_dir.rglob(f'*{ext}'))
        all_files.extend(input_dir.rglob(f'*{ext.upper()}'))
    
    print(f"📁 Найдено изображений: {len(all_files)}")
    
    if len(all_files) == 0:
        print(f"⚠️  Изображения не найдены в {input_dir}")
        print("   Проверьте путь к скачанным файлам")
        return stats
    
    processed = 0
    for img_path in all_files:
        try:
            # Определить тип знака
            sign_type = detect_sign_type(img_path.name)
            
            # Подсчитать
            if sign_type not in stats:
                stats[sign_type] = 0
            
            # Новое имя файла
            new_filename = f"{sign_type}_{stats[sign_type]:04d}{img_path.suffix.lower()}"
            new_path = output_dir / new_filename
            
            # Копировать файл
            shutil.copy2(img_path, new_path)
            
            stats[sign_type] += 1
            processed += 1
            
            if processed % 100 == 0:
                print(f"   Обработано: {processed}/{len(all_files)}")
            
        except Exception as e:
            print(f"⚠️  Ошибка при обработке {img_path.name}: {e}")
    
    print(f"✅ Обработано файлов: {processed}")
    return stats


def print_statistics(stats: Dict[str, int]):
    """Вывести статистику."""
    print(f"\n📊 Статистика датасета:")
    print(f"   Всего классов: {len(stats)}")
    print(f"   Всего изображений: {sum(stats.values())}")
    print("\n   Распределение по классам:")
    
    for sign_type, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        bar = '█' * min(50, count // 20)
        print(f"   {sign_type:30s} {count:5d} {bar}")


def save_class_mapping(stats: Dict[str, int], output_dir: Path):
    """Сохранить маппинг классов."""
    sorted_classes = sorted([k for k in stats.keys() if k != 'unknown'])
    
    class_mapping = {i: class_name for i, class_name in enumerate(sorted_classes)}
    
    # Сохранить в JSON
    mapping_file = output_dir / 'class_mapping.json'
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(class_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Маппинг классов сохранен: {mapping_file}")
    
    # Вывести для configs/dataset.yaml
    print(f"\n📋 Скопируйте в configs/dataset.yaml:")
    print("```yaml")
    print("names:")
    for class_id, class_name in class_mapping.items():
        print(f"  {class_id}: {class_name}")
    print(f"nc: {len(class_mapping)}")
    print("```")


def main():
    args = parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"❌ Входная папка не существует: {input_dir}")
        print("\n💡 Скачайте файлы из Google Drive:")
        print("   1. Откройте: https://drive.google.com/drive/folders/1xS3Hu_s-uqtVHdy7n0Y2t_Dv2RxkdCYj")
        print("   2. Нажмите на папку 'named' правой кнопкой -> Скачать")
        print("   3. Распакуйте архив")
        print(f"   4. Запустите: python utils/organize_downloaded.py --input путь/к/распакованной/папке")
        return
    
    print("="*60)
    print("🚀 ОРГАНИЗАЦИЯ ДАТАСЕТА ДОРОЖНЫХ ЗНАКОВ")
    print("="*60)
    
    # Организовать файлы
    stats = organize_files(input_dir, output_dir)
    
    if not stats:
        return
    
    # Статистика
    print_statistics(stats)
    
    # Сохранить маппинг
    save_class_mapping(stats, output_dir)
    
    print("\n" + "="*60)
    print("✅ ОРГАНИЗАЦИЯ ЗАВЕРШЕНА!")
    print("="*60)
    print(f"\n📁 Файлы сохранены в: {output_dir}")
    print(f"\n💡 Следующие шаги:")
    print("   1. Обновите configs/dataset.yaml (скопируйте классы выше)")
    print("   2. Создайте YOLO аннотации для изображений")
    print("   3. Запустите: python utils/prepare_dataset.py --input dataset/raw")


if __name__ == '__main__':
    main()
