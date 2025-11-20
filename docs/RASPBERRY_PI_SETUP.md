# Установка и настройка Raspberry Pi 5 для RSD

Подробное руководство по настройке Raspberry Pi 5 с Google Coral Edge TPU для системы распознавания дорожных знаков.

## 📋 Необходимое оборудование

### Основное
- ✅ Raspberry Pi 5 (8GB RAM)
- ✅ Google Coral USB Accelerator
- ✅ Raspberry Pi Camera Module 3 или USB webcam (минимум 720p, рекомендуется 1080p)
- ✅ microSD карта 64GB+ (Class 10, UHS-I) или USB SSD 128GB+
- ✅ Официальный блок питания 5V 5A (27W)

### Дополнительное (рекомендуется)
- ✅ Активное охлаждение (вентилятор + радиатор) - **критично для стабильной работы**
- ✅ Корпус с креплениями для автомобиля
- ✅ USB SSD для операционной системы (лучше производительность чем microSD)
- ✅ USB Hub с питанием (если подключаете несколько устройств)

### Бюджет
- Raspberry Pi 5 8GB: ~$80
- Google Coral USB: ~$60
- Pi Camera Module 3: ~$25-35
- Питание + охлаждение + корпус: ~$30-50
- **Итого**: ~$195-225

## 🔧 Настройка Raspberry Pi 5

### Шаг 1: Установка Raspberry Pi OS

```bash
# Скачать Raspberry Pi Imager: https://www.raspberrypi.com/software/

# Выбрать:
# - OS: Raspberry Pi OS (64-bit) - рекомендуется Lite версия для лучшей производительности
# - Storage: ваша microSD или USB SSD
# - Settings: настроить hostname, SSH, WiFi

# После записи - вставить в Raspberry Pi и загрузиться
```

### Шаг 2: Первоначальная настройка

```bash
# Подключиться по SSH
ssh pi@raspberrypi.local
# Пароль: тот что установили в Imager

# Обновить систему
sudo apt update && sudo apt full-upgrade -y

# Установить необходимые пакеты
sudo apt install -y \
    git \
    python3-pip \
    python3-venv \
    python3-opencv \
    libatlas-base-dev \
    libopenblas-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    cmake

# Перезагрузка
sudo reboot
```

### Шаг 3: Установка Google Coral Edge TPU Runtime

```bash
# Добавить Coral репозиторий
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
    sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

# Добавить GPG ключ
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Обновить и установить
sudo apt update
sudo apt install -y libedgetpu1-std

# Проверить установку
# Подключить Coral USB Accelerator
lsusb | grep "Global Unichip"
# Должно вывести: Bus XXX Device XXX: ID 1a6e:089a Global Unichip Corp.
```

**Важно**: Используйте `libedgetpu1-std` (стандартная частота), не `libedgetpu1-max` (максимальная частота, сильнее греется).

### Шаг 4: Настройка камеры

#### Для Pi Camera Module

```bash
# Включить камеру (только для старых версий Raspberry Pi OS)
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable

# Для новых версий OS камера работает автоматически

# Установить picamera2
sudo apt install -y python3-picamera2

# Проверить камеру
libcamera-hello
```

#### Для USB webcam

```bash
# Проверить подключение
v4l2-ctl --list-devices

# Установить v4l-utils для настройки
sudo apt install -y v4l-utils
```

### Шаг 5: Установка Python зависимостей

```bash
# Клонировать репозиторий RSD
cd ~
git clone https://github.com/OrDinaD/RSD.git
cd RSD

# Создать virtual environment (рекомендуется)
python3 -m venv venv
source venv/bin/activate

# Установить зависимости
pip install --upgrade pip
pip install -r requirements_rpi.txt

# Установить PyCoral
pip install pycoral

# Проверить установку
python3 -c "from pycoral.utils import edgetpu; print('PyCoral OK')"
python3 -c "import cv2; print('OpenCV', cv2.__version__)"
```

### Шаг 6: Настройка производительности

```bash
# Настроить активное охлаждение
# Отредактировать /boot/config.txt
sudo nano /boot/config.txt

# Добавить в конец файла:
# Настройки вентилятора
dtoverlay=gpio-fan,gpiopin=14,temp=60000

# Over-clocking (опционально, требует хорошее охлаждение)
# arm_freq=2600
# over_voltage=6

# Сохранить (Ctrl+O, Enter, Ctrl+X)

# Увеличить swap (для компиляции)
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Изменить CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Настроить GPU memory
sudo raspi-config
# Performance Options > GPU Memory > 256

# Перезагрузка
sudo reboot
```

### Шаг 7: Оптимизация для real-time детекции

```bash
# Отключить ненужные сервисы
sudo systemctl disable bluetooth
sudo systemctl disable avahi-daemon

# Настроить CPU governor на performance
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Сделать постоянным
sudo nano /etc/rc.local
# Добавить перед "exit 0":
# echo "performance" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Увеличить USB current limit (для Coral)
sudo nano /boot/config.txt
# Добавить:
# max_usb_current=1
```

## 🎯 Тестирование установки

### Проверка Edge TPU

```bash
cd ~/RSD

# Тест PyCoral (использует пример модель)
python3 - <<EOF
from pycoral.utils import edgetpu
import time

print("Инициализация Edge TPU...")
interpreter = edgetpu.make_interpreter(
    'https://github.com/google-coral/test_data/raw/master/mobilenet_v2_1.0_224_quant_edgetpu.tflite'
)
interpreter.allocate_tensors()
print("✅ Edge TPU работает!")

# Бенчмарк
print("\nБенчмарк inference speed...")
for i in range(10):
    start = time.time()
    interpreter.invoke()
    print(f"  Run {i+1}: {(time.time() - start)*1000:.2f}ms")
EOF
```

### Проверка камеры

```bash
# Для Pi Camera
python3 - <<EOF
from picamera2 import Picamera2
import time

picam2 = Picamera2()
picam2.start()
time.sleep(2)
frame = picam2.capture_array()
print(f"✅ Камера работает! Разрешение: {frame.shape}")
picam2.stop()
EOF

# Для USB webcam
python3 - <<EOF
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print(f"✅ Камера работает! Разрешение: {frame.shape}")
else:
    print("❌ Ошибка камеры")
cap.release()
EOF
```

### Полный тест системы

```bash
# Скопировать тестовую модель (замените на вашу реальную модель)
# scp models/best_int8_edgetpu.tflite pi@raspberrypi:~/RSD/models/

# Запустить детекцию
python3 deployment/inference.py \
    --model models/best_int8_edgetpu.tflite \
    --config configs/deployment_config.yaml \
    --camera 0 \
    --display

# Должны увидеть live видео с FPS counter
# Ожидаемый FPS: 18-25 с Edge TPU
```

## 🔍 Мониторинг производительности

```bash
# Мониторинг температуры
watch -n 1 vcgencmd measure_temp

# Мониторинг CPU/RAM
htop

# Мониторинг в реальном времени
python3 - <<EOF
import time
import psutil

while True:
    cpu = psutil.cpu_percent(interval=1)
    temp = psutil.sensors_temperatures()['cpu_thermal'][0].current
    ram = psutil.virtual_memory().percent
    
    print(f"CPU: {cpu:.1f}% | Temp: {temp:.1f}°C | RAM: {ram:.1f}%")
    time.sleep(1)
EOF
```

## 🚨 Устранение проблем

### Edge TPU не обнаружен

```bash
# Проверить подключение
lsusb | grep "Global Unichip"

# Переподключить USB
# Или попробовать другой порт (используйте USB 3.0 порты - синие)

# Переустановить libedgetpu
sudo apt remove libedgetpu1-std
sudo apt install libedgetpu1-std

# Проверить права доступа
sudo usermod -aG plugdev $USER
sudo reboot
```

### Перегрев (>80°C)

```bash
# Проверить вентилятор
# Должен крутиться при температуре выше 60°C

# Добавить агрессивное охлаждение
sudo nano /boot/config.txt
# Изменить:
dtoverlay=gpio-fan,gpiopin=14,temp=50000  # Старт при 50°C

# Установить радиаторы на CPU, GPU, RAM
```

### Низкий FPS

```bash
# Проверить что используется Edge TPU
# В логах должно быть: "Edge TPU: ✓"

# Проверить температуру - throttling начинается при 80°C
vcgencmd measure_temp
vcgencmd get_throttled
# 0x0 = OK, другое значение = throttling

# Закрыть другие процессы
sudo systemctl stop lightdm  # Отключить GUI если не нужен

# Уменьшить разрешение камеры в configs/deployment_config.yaml
# resolution: [1280, 720]  # вместо 1920x1080
```

### Питание недостаточно

```bash
# Проверить voltage
vcgencmd get_throttled
# Если 0x50000 или 0x50005 = low voltage warning

# Решение:
# - Использовать официальный блок питания 5V 5A
# - Подключить Coral через powered USB hub
# - Проверить качество USB кабеля
```

## 📊 Benchmark результаты

После установки запустите benchmark:

```bash
python3 utils/benchmark.py \
    --model models/best_int8_edgetpu.tflite \
    --runs 100

# Ожидаемые результаты на RPi 5 + Coral:
# Average FPS: 20-25
# Average Latency: 40-50ms
# CPU Usage: 30-40%
# Temperature: 65-75°C (с активным охлаждением)
```

## 🎬 Автозапуск при загрузке

```bash
# Создать systemd service
sudo nano /etc/systemd/system/rsd-detection.service

# Добавить:
[Unit]
Description=RSD Traffic Sign Detection
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/RSD
ExecStart=/home/pi/RSD/venv/bin/python3 deployment/inference.py --model models/best_int8_edgetpu.tflite --config configs/deployment_config.yaml
Restart=always

[Install]
WantedBy=multi-user.target

# Сохранить и включить
sudo systemctl enable rsd-detection.service
sudo systemctl start rsd-detection.service

# Проверить статус
sudo systemctl status rsd-detection.service
```

## ✅ Чеклист готовности

- [ ] Raspberry Pi 5 настроен и обновлен
- [ ] Edge TPU Runtime установлен и Coral обнаружен
- [ ] Камера работает и выдает изображение
- [ ] Python зависимости установлены
- [ ] Активное охлаждение работает
- [ ] Тест детекции показывает 18-25 FPS
- [ ] Температура стабильна (<75°C)
- [ ] Система готова к deployment!

---

**Примечание**: Этот гайд протестирован на Raspberry Pi 5 с Raspberry Pi OS (64-bit) и Google Coral USB Accelerator.
