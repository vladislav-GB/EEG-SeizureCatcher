"""
test_setup.py - Полная проверка системы перед запуском

Проверяет:
1. Пути к данным
2. Доступные файлы EDF и CSV
3. GPU/CPU конфигурацию
4. Установленные библиотеки
5. Конфигурацию модели
6. Создание тестового тензора
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("🔧 ПРОВЕРКА СИСТЕМЫ ДЛЯ EEGNeX")
print("="*70)

# ============================================================================
# 1. ПРОВЕРКА PYTHON И БИБЛИОТЕК
# ============================================================================
print("\n📦 1. ПРОВЕРКА БИБЛИОТЕК")

libraries = {
    'torch': 'PyTorch',
    'numpy': 'NumPy',
    'mne': 'MNE',
    'braindecode': 'BrainDecode',
    'sklearn': 'Scikit-learn',
    'pandas': 'Pandas',
    'matplotlib': 'Matplotlib',
    'tqdm': 'TQDM',
    'scipy': 'SciPy'
}

missing = []
versions = {}

for lib, name in libraries.items():
    try:
        if lib == 'sklearn':
            import sklearn
            versions[lib] = sklearn.__version__
        else:
            module = __import__(lib)
            versions[lib] = module.__version__
        print(f"  ✅ {name:15} v{versions[lib]}")
    except ImportError:
        print(f"  ❌ {name:15} НЕ УСТАНОВЛЕНА")
        missing.append(name)

if missing:
    print(f"\n  ⚠️ Установите недостающие библиотеки:")
    print(f"     pip install {' '.join([m.lower() for m in missing])}")
    sys.exit(1)

# ============================================================================
# 2. ПРОВЕРКА CUDA И GPU
# ============================================================================
print("\n🎮 2. ПРОВЕРКА GPU")

import torch

print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  GPU name: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Проверка памяти
    free_mem = torch.cuda.memory_reserved(0) / 1e9
    print(f"  Free memory: {free_mem:.1f} GB")
    
    if torch.cuda.get_device_properties(0).total_memory / 1e9 < 4:
        print(f"  ⚠️ ВНИМАНИЕ: Мало VRAM (3GB). Рекомендуется batch_size=8")
    else:
        print(f"  ✅ VRAM достаточно для batch_size=16")
else:
    print(f"  ⚠️ CUDA не найдена - будет использоваться CPU (медленнее в 5-10 раз)")

# ============================================================================
# 3. ПРОВЕРКА ПУТЕЙ К ДАННЫМ
# ============================================================================
print("\n📁 3. ПРОВЕРКА ПУТЕЙ")

from pathlib import Path
from config import Config

paths = {
    'R_DATA_PATH': Config.R_DATA_PATH,
    'EDF_PATH': Config.EDF_PATH,
    'OUTPUT_PATH': Config.OUTPUT_PATH,
    'VISUALIZATION_PATH': Config.VISUALIZATION_PATH,
    'MODEL_PATH': Config.MODEL_PATH
}

for name, path in paths.items():
    p = Path(path)
    if p.exists():
        print(f"  ✅ {name:20} существует")
    else:
        print(f"  ❌ {name:20} НЕ СУЩЕСТВУЕТ")
        try:
            p.mkdir(parents=True, exist_ok=True)
            print(f"     → Папка создана")
        except:
            print(f"     → Невозможно создать (проверьте права)")

# ============================================================================
# 4. ПРОВЕРКА CSV ФАЙЛОВ (R-результаты)
# ============================================================================
print("\n📊 4. ПРОВЕРКА CSV ФАЙЛОВ")

csv_files = ['train_sample.csv', 'val_sample.csv', 'test_sample.csv']
csv_path = Path(Config.R_DATA_PATH)

for csv_file in csv_files:
    full_path = csv_path / csv_file
    if full_path.exists():
        import pandas as pd
        try:
            df = pd.read_csv(full_path, sep=';', encoding='windows-1251', nrows=5)
            print(f"  ✅ {csv_file:20} ({len(df)} строк в образце)")
        except Exception as e:
            print(f"  ❌ {csv_file:20} ошибка чтения: {str(e)[:50]}")
    else:
        print(f"  ❌ {csv_file:20} НЕ НАЙДЕН")

# ============================================================================
# 5. ПРОВЕРКА EDF ФАЙЛОВ
# ============================================================================
print("\n🧠 5. ПРОВЕРКА EDF ФАЙЛОВ")

edf_path = Path(Config.EDF_PATH)
if edf_path.exists():
    edf_files = list(edf_path.glob("eeg*.edf"))
    print(f"  Найдено EDF файлов: {len(edf_files)}")
    
    if len(edf_files) > 0:
        # Проверяем первый файл
        test_file = edf_files[0]
        print(f"  Тестовый файл: {test_file.name}")
        
        try:
            import mne
            raw = mne.io.read_raw_edf(test_file, preload=False, verbose=False)
            print(f"  ✅ Файл читается корректно")
            print(f"     Частота дискретизации: {raw.info['sfreq']} Гц")
            print(f"     Длительность: {raw.n_times / raw.info['sfreq']:.1f} сек")
            print(f"     Каналов: {len(raw.ch_names)}")
            
            # Проверяем наличие нужных каналов
            ch_names_upper = [ch.upper() for ch in raw.ch_names]
            found_channels = []
            for target_ch in Config.CHANNELS:
                for ch in ch_names_upper:
                    if target_ch.upper() in ch:
                        found_channels.append(target_ch)
                        break
            
            print(f"     Найдено каналов из конфига: {len(found_channels)}/{len(Config.CHANNELS)}")
            if len(found_channels) < len(Config.CHANNELS):
                missing_ch = set(Config.CHANNELS) - set(found_channels)
                print(f"     ⚠️ Отсутствуют: {list(missing_ch)[:5]}...")
            
        except Exception as e:
            print(f"  ❌ Ошибка чтения EDF: {str(e)[:100]}")
    else:
        print(f"  ❌ Нет EDF файлов в {edf_path}")
else:
    print(f"  ❌ Папка EDF не существует")

# ============================================================================
# 6. ПРОВЕРКА КОНФИГУРАЦИИ
# ============================================================================
print("\n⚙️ 6. ПРОВЕРКА КОНФИГУРАЦИИ")

print(f"  SEED: {Config.SEED}")
print(f"  TARGET_SR: {Config.TARGET_SR} Гц")
print(f"  WINDOW_SIZE: {Config.WINDOW_SIZE} точек ({Config.WINDOW_SEC} сек)")
print(f"  N_CHANNELS: {Config.N_CHANNELS}")
print(f"  BATCH_SIZE: {Config.BATCH_SIZE}")
print(f"  LEARNING_RATE: {Config.LEARNING_RATE}")
print(f"  EPOCHS: {Config.EPOCHS}")
print(f"  PATIENCE: {Config.PATIENCE}")
print(f"  GRADIENT_CLIP_NORM: {Config.GRADIENT_CLIP_NORM}")

# Проверка весов паттернов
pattern_weights = Config.PATTERN_WEIGHTS
print(f"  PATTERN_WEIGHTS: {len(pattern_weights)} паттернов")
for p, w in pattern_weights.items():
    if w != 0.5:
        print(f"    {p}: {w}")

# ============================================================================
# 7. ТЕСТ ТЕНЗОРА (проверка модели)
# ============================================================================
print("\n🧪 7. ТЕСТ МОДЕЛИ (быстрый forward pass)")

try:
    from braindecode.models import EEGNeX
    
    # Создаем тестовую модель
    model = EEGNeX(
        n_chans=Config.N_CHANNELS,
        n_outputs=Config.EEGNEX_N_OUTPUTS,
        n_times=Config.WINDOW_SIZE,
        sfreq=Config.TARGET_SR,
        input_window_seconds=Config.WINDOW_SEC,
        activation=Config.EEGNEX_ACTIVATION,
        filter_1=Config.EEGNEX_FILTER_1,
        filter_2=Config.EEGNEX_FILTER_2,
        drop_prob=Config.EEGNEX_DROP_PROB
    )
    
    # Тестовый тензор
    test_input = torch.randn(1, 1, Config.N_CHANNELS, Config.WINDOW_SIZE)
    
    # Проверка на CPU
    model.eval()
    with torch.no_grad():
        output = model(test_input)
    
    print(f"  ✅ Модель создана")
    print(f"     Параметров: {sum(p.numel() for p in model.parameters()):,}")
    print(f"     Вход: {list(test_input.shape)}")
    print(f"     Выход: {list(output.shape)}")
    
    # Тест на GPU если доступен
    if torch.cuda.is_available():
        try:
            model_gpu = model.to('cuda')
            test_input_gpu = test_input.to('cuda')
            with torch.no_grad():
                output_gpu = model_gpu(test_input_gpu)
            print(f"  ✅ GPU forward pass успешен")
            del model_gpu, test_input_gpu
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ⚠️ Не хватает VRAM для теста (batch_size нужно уменьшить)")
                print(f"     Рекомендуется BATCH_SIZE=8")
            else:
                print(f"  ❌ GPU ошибка: {str(e)[:100]}")
    
except Exception as e:
    print(f"  ❌ Ошибка создания модели: {str(e)[:100]}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 8. ПРОВЕРКА ПАМЯТИ
# ============================================================================
print("\n💾 8. ПРОВЕРКА СИСТЕМНОЙ ПАМЯТИ")

import psutil
mem = psutil.virtual_memory()
print(f"  Total RAM: {mem.total / 1e9:.1f} GB")
print(f"  Available RAM: {mem.available / 1e9:.1f} GB")
print(f"  Used RAM: {mem.used / 1e9:.1f} GB ({mem.percent}%)")

if mem.available / 1e9 < 8:
    print(f"  ⚠️ Мало свободной RAM (<8GB). Закройте браузер и другие приложения.")
else:
    print(f"  ✅ RAM достаточно")

# ============================================================================
# 9. ПРОВЕРКА ДИСКОВОГО ПРОСТРАНСТВА
# ============================================================================
print("\n💿 9. ПРОВЕРКА ДИСКОВОГО ПРОСТРАНСТВА")

for path_name, path in paths.items():
    if Path(path).exists():
        usage = psutil.disk_usage(str(Path(path).parent))
        free_gb = usage.free / 1e9
        print(f"  {path_name:20}: {free_gb:.1f} GB свободно")
        
        if free_gb < 50 and path_name in ['OUTPUT_PATH', 'MODEL_PATH']:
            print(f"     ⚠️ Мало места для сохранения моделей и данных")

# ============================================================================
# 10. ФИНАЛЬНЫЙ ВЕРДИКТ
# ============================================================================
print("\n" + "="*70)
print("📋 ФИНАЛЬНЫЙ ВЕРДИКТ")
print("="*70)

issues = []

if not torch.cuda.is_available():
    issues.append("❌ CUDA не доступна - обучение будет медленным")

if len(edf_files) == 0:
    issues.append("❌ Нет EDF файлов - проверьте путь в config.py")

csv_exists = all([(csv_path / f).exists() for f in csv_files])
if not csv_exists:
    issues.append("❌ Нет CSV файлов с разметкой - проверьте R_DATA_PATH")

if mem.available / 1e9 < 8:
    issues.append("⚠️ Мало RAM - закройте другие приложения")

if issues:
    print("\n".join(issues))
    print("\n⚠️ РЕКОМЕНДАЦИИ ПЕРЕД ЗАПУСКОМ:")
    for issue in issues:
        if "EDF" in issue:
            print("  • Укажите правильный путь к EDF файлам в config.py")
        if "CSV" in issue:
            print("  • Укажите правильный путь к CSV файлам в config.py")
        if "CUDA" in issue:
            print("  • Установите CUDA версию PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        if "RAM" in issue:
            print("  • Закройте браузер, IDE, другие приложения")
else:
    print("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!")
    print("\n🚀 МОЖНО ЗАПУСКАТЬ:")
    print("   1. python preprocessing.py  (~30-60 минут)")
    print("   2. python train.py          (~1-3 часа)")
    print("   3. python evaluate.py       (~5-10 минут)")
    print("   4. python visualize.py --patient 16")

print("="*70)

# ============================================================================
# 11. СОВЕТЫ ПО ОПТИМИЗАЦИИ ДЛЯ ВАШЕЙ СИСТЕМЫ
# ============================================================================
print("\n💡 СОВЕТЫ ДЛЯ ВАШЕЙ СИСТЕМЫ (GTX 1060 3GB + Ryzen 3700x):")
print("="*70)
print("  1. BATCH_SIZE = 8 (вместо 16) - чтобы избежать OOM")
print("  2. Закройте все приложения перед обучением")
print("  3. Мониторьте температуру: sensors (в терминале)")
print("  4. При preprocessing.py CPU будет загружен на 100% - это нормально")
print("  5. Используйте screen или tmux для долгих запусков")
print("="*70 + "\n")

# Запрос на запуск (опционально)
response = input("Запустить preprocessing.py сейчас? (y/N): ")
if response.lower() == 'y':
    print("\n🚀 Запуск preprocessing.py...")
    os.system("python preprocessing.py")
else:
    print("\n✅ Проверка завершена.")