#!/usr/bin/env python3
"""
Скрипт для скачивания и подготовки датасета дорожных знаков из Google Drive.
Скачивает изображения из папки 'named', переименовывает и организует их для обучения YOLO.

Использование:
    pip install gdown
    python utils/download_dataset.py --output dataset/raw
"""

import os
import argparse
import re
from pathlib import Path
import subprocess
import shutil
from typing import Dict, List
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Скачивание датасета из Google Drive')
    parser.add_argument('--output', type=str, default='dataset/raw',
                        help='Директория для сохранения скачанных файлов')
    parser.add_argument('--folder-id', type=str, 
                        default='1xS3Hu_s-uqtVHdy7n0Y2t_Dv2RxkdCYj',
                        help='ID папки Google Drive')
    parser.add_argument('--rename', action='store_true', default=True,
                        help='Переименовать файлы в стандартный формат')
    parser.add_argument('--organize', action='store_true', default=True,
                        help='Организовать файлы по классам')
    return parser.parse_args()


# Маппинг названий знаков из датасета в стандартные имена
SIGN_NAME_MAPPING = {
    # Знаки ограничения скорости
    'speed_limit_20': 'speed_limit_20',
    'speed_limit_30': 'speed_limit_30',
    'speed_limit_50': 'speed_limit_50',
    'speed_limit_60': 'speed_limit_60',
    'speed_limit_70': 'speed_limit_70',
    'speed_limit_80': 'speed_limit_80',
    'speed_limit_100': 'speed_limit_100',
    'speed_limit_120': 'speed_limit_120',
    
    # Запрещающие знаки
    'no_entry': 'no_entry',
    'no_vehicles': 'no_vehicles',
    'no_overtaking': 'no_overtaking',
    'no_overtaking_trucks': 'no_overtaking_trucks',
    'no_stopping': 'no_stopping',
    'no_parking': 'no_parking',
    
    # Обязательные знаки
    'mandatory_roundabout': 'roundabout',
    'mandatory_turn_left': 'turn_left',
    'mandatory_turn_right': 'turn_right',
    'mandatory_straight': 'go_straight',
    
    # Предупреждающие знаки
    'warning_pedestrian_crossing': 'pedestrian_crossing',
    'warning_children': 'children_crossing',
    'warning_bicycle': 'bicycle_crossing',
    'warning_slippery_road': 'slippery_road',
    'warning_road_work': 'construction',
    'warning_traffic_signals': 'traffic_light',
    'warning_dangerous_curve': 'dangerous_curve',
    'warning_bumpy_road': 'bumpy_road',
    
    # Знаки приоритета
    'priority_road': 'priority_road',
    'yield': 'yield_sign',
    'stop': 'stop_sign',
    
    # Информационные знаки
    'parking': 'parking',
    'crosswalk': 'pedestrian_crossing',
}


def install_gdown():
    """Установить gdown если не установлен."""
    try:
        import gdown
        print("✅ gdown уже установлен")
        return True
    except ImportError:
        print("📦 Установка gdown...")
        try:
            subprocess.check_call(['pip', 'install', 'gdown'])
            print("✅ gdown установлен")
            return True
        except Exception as e:
            print(f"❌ Не удалось установить gdown: {e}")
            print("💡 Установите вручную: pip install gdown")
            return False


def download_from_gdrive(folder_id: str, output_dir: Path) -> bool:
    """Скачать папку из Google Drive."""
    print(f"\n🔽 Скачивание файлов из Google Drive...")
    print(f"   Folder ID: {folder_id}")
    print(f"   Output: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import gdown
        
        # URL папки Google Drive
        folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
        
        print(f"📁 Скачивание папки: {folder_url}")
        print("⏳ Это может занять некоторое время...")
        
        # Скачать всю папку
        gdown.download_folder(
            url=folder_url,
            output=str(output_dir),
            quiet=False,
            use_cookies=False
        )
        
        print("✅ Скачивание завершено!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при скачивании: {e}")
        print("\n💡 Альтернативный способ:")
        print("   1. Откройте ссылку в браузере:")
        print(f"      https://drive.google.com/drive/folders/{folder_id}")
        print("   2. Выберите все файлы (Ctrl+A)")
        print("   3. Скачайте их (правый клик -> Скачать)")
        print(f"   4. Распакуйте в: {output_dir}")
        return False


def normalize_filename(filename: str) -> str:
    """Нормализовать имя файла."""
    # Убрать расширение
    name = Path(filename).stem
    
    # Преобразовать в lowercase
    name = name.lower()
    
    # Заменить пробелы и спецсимволы на подчеркивания
    name = re.sub(r'[^\w\s-]', '_', name)
    name = re.sub(r'[-\s]+', '_', name)
    
    return name


def extract_sign_type(filename: str) -> str:
    """Извлечь тип знака из имени файла."""
    normalized = normalize_filename(filename)
    
    # Попробовать найти в маппинге
    for pattern, sign_type in SIGN_NAME_MAPPING.items():
        if pattern.lower() in normalized:
            return sign_type
    
    # Попробовать извлечь из числового кода
    # Например: "01_speed_limit_30.jpg" -> "speed_limit_30"
    match = re.search(r'(\d+).*?(\w+)', normalized)
    if match:
        # Попробовать найти описательную часть
        parts = normalized.split('_')
        if len(parts) > 1:
            return '_'.join(parts[1:])  # Убрать числовой префикс
    
    # Если не удалось определить, вернуть как есть
    return normalized


def rename_and_organize(input_dir: Path, sign_mapping: Dict[str, str]) -> Dict[str, int]:
    """Переименовать и организовать файлы."""
    print(f"\n🔄 Переименование и организация файлов...")
    
    stats = {}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Найти все изображения
    all_files = []
    for ext in image_extensions:
        all_files.extend(input_dir.glob(f'**/*{ext}'))
        all_files.extend(input_dir.glob(f'**/*{ext.upper()}'))
    
    print(f"📁 Найдено изображений: {len(all_files)}")
    
    for img_path in all_files:
        try:
            # Определить тип знака
            sign_type = extract_sign_type(img_path.name)
            
            # Применить маппинг если есть
            if sign_type in sign_mapping:
                sign_type = sign_mapping[sign_type]
            
            # Подсчитать статистику
            if sign_type not in stats:
                stats[sign_type] = 0
            
            # Новое имя файла
            new_filename = f"{sign_type}_{stats[sign_type]:04d}{img_path.suffix}"
            new_path = input_dir / new_filename
            
            # Переименовать (если не совпадает)
            if img_path != new_path:
                # Если файл с таким именем уже существует, добавить суффикс
                counter = 0
                while new_path.exists():
                    counter += 1
                    new_filename = f"{sign_type}_{stats[sign_type]:04d}_{counter}{img_path.suffix}"
                    new_path = input_dir / new_filename
                
                img_path.rename(new_path)
                print(f"   {img_path.name} -> {new_filename}")
            
            stats[sign_type] += 1
            
        except Exception as e:
            print(f"⚠️  Ошибка при обработке {img_path.name}: {e}")
    
    return stats


def create_class_mapping(stats: Dict[str, int], output_dir: Path):
    """Создать маппинг классов для dataset.yaml."""
    print(f"\n📝 Создание маппинга классов...")
    
    # Сортировать классы
    sorted_classes = sorted(stats.keys())
    
    # Создать маппинг
    class_mapping = {i: class_name for i, class_name in enumerate(sorted_classes)}
    
    # Сохранить в JSON
    mapping_file = output_dir / 'class_mapping.json'
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(class_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Маппинг сохранен: {mapping_file}")
    
    # Вывести информацию для configs/dataset.yaml
    print(f"\n📋 Добавьте в configs/dataset.yaml:")
    print("---")
    print("names:")
    for class_id, class_name in class_mapping.items():
        print(f"  {class_id}: {class_name}")
    print(f"nc: {len(class_mapping)}")
    print("---")
    
    return class_mapping


def print_statistics(stats: Dict[str, int]):
    """Вывести статистику по классам."""
    print(f"\n📊 Статистика датасета:")
    print(f"   Всего классов: {len(stats)}")
    print(f"   Всего изображений: {sum(stats.values())}")
    print("\n   Распределение по классам:")
    
    for sign_type, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        bar = '█' * (count // 10)
        print(f"   {sign_type:30s} {count:5d} {bar}")


def create_dummy_annotations(input_dir: Path):
    """Создать пустые аннотации для всех изображений (для теста)."""
    print(f"\n⚠️  Создание пустых аннотаций...")
    print("   ВАЖНО: Вам нужно будет создать реальные аннотации!")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for ext in image_extensions:
        for img_path in input_dir.glob(f'*{ext}'):
            txt_path = img_path.with_suffix('.txt')
            if not txt_path.exists():
                # Создать пустой файл аннотации
                txt_path.touch()
    
    print("✅ Пустые аннотации созданы")
    print("💡 Используйте LabelImg или другой инструмент для аннотирования:")
    print("   https://github.com/HumanSignal/labelImg")


def main():
    args = parse_args()
    
    output_dir = Path(args.output)
    
    print("="*60)
    print("🚀 СКАЧИВАНИЕ И ПОДГОТОВКА ДАТАСЕТА ДОРОЖНЫХ ЗНАКОВ")
    print("="*60)
    
    # Установить gdown если нужно
    if not install_gdown():
        return
    
    # Скачать файлы из Google Drive
    success = download_from_gdrive(args.folder_id, output_dir)
    
    if not success:
        print("\n⚠️  Скачивание не удалось автоматически")
        response = input("Вы скачали файлы вручную? (y/n): ")
        if response.lower() != 'y':
            print("❌ Прерывание")
            return
    
    # Проверить что файлы есть
    image_files = list(output_dir.glob('**/*.jpg')) + list(output_dir.glob('**/*.png'))
    
    if len(image_files) == 0:
        print(f"❌ Изображения не найдены в {output_dir}")
        print("   Проверьте что файлы скачаны правильно")
        return
    
    print(f"\n✅ Найдено изображений: {len(image_files)}")
    
    # Переименовать и организовать
    if args.rename:
        stats = rename_and_organize(output_dir, SIGN_NAME_MAPPING)
        print_statistics(stats)
        create_class_mapping(stats, output_dir)
    
    # Создать пустые аннотации (если их нет)
    txt_files = list(output_dir.glob('*.txt'))
    if len(txt_files) == 0:
        create_dummy_annotations(output_dir)
    
    print("\n" + "="*60)
    print("✅ ПОДГОТОВКА ЗАВЕРШЕНА!")
    print("="*60)
    print(f"\n📁 Файлы сохранены в: {output_dir}")
    print(f"\n💡 Следующие шаги:")
    print("   1. Проверьте что все изображения скачаны")
    print("   2. Создайте YOLO аннотации (если их нет)")
    print("   3. Обновите configs/dataset.yaml с классами")
    print("   4. Запустите: python utils/prepare_dataset.py")


if __name__ == '__main__':
    main()
