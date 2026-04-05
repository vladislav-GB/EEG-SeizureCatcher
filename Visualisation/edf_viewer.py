"""
edf_viewer.py - Визуализация ЭЭГ с аннотациями трех экспертов

НАЗНАЧЕНИЕ:
    Отдельный инструмент для врачей и исследователей. Позволяет визуально оценить,
    как эксперты размечали приступы, и сравнить с предсказаниями модели.

ВОЗМОЖНОСТИ:
    - Отображение ЭЭГ сигнала с аннотациями экспертов
    - Цветовая кодировка паттернов (000-111)
    - Навигация по времени
    - Выборка каналов
    - Сравнение с предсказаниями модели (опционально)

ИСПОЛЬЗОВАНИЕ:
    python edf_viewer.py --patient 16
    python edf_viewer.py --patient 16 --duration 60 --channels 10
    python edf_viewer.py --patient 41 --model best_eegnex_model.pth
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# Настройка Qt для Wayland/X11
if 'linux' in sys.platform:
    os.environ['QT_QPA_PLATFORM'] = 'wayland-egl' if os.environ.get('WAYLAND_DISPLAY') else 'xcb'

import matplotlib
matplotlib.use('Qt5Agg')

import mne
import torch
from braindecode.models import EEGNeX

warnings.filterwarnings('ignore')

# Добавляем путь к проекту для импорта config
sys.path.insert(0, str(Path(__file__).parent))
from config import Config


# ============================================================================
# КОНСТАНТЫ И ЦВЕТА
# ============================================================================

# Цвета для паттернов (по аналогии с R-визуализациями)
PATTERN_COLORS = {
    '000': '#4E79A7',  # синий - фон
    '001': '#F28E2B',  # оранжевый - только C
    '010': '#E15759',  # красный - только B
    '011': '#76B7B2',  # бирюзовый - B+C
    '100': '#59A14F',  # зеленый - только A
    '101': '#EDC949',  # желтый - A+C
    '110': '#AF7AA1',  # фиолетовый - A+B
    '111': '#FF9DA7',  # розовый - все три
}

# Описания паттернов
PATTERN_DESCRIPTIONS = {
    '000': 'Нет приступа (все эксперты согласны)',
    '001': 'Только эксперт C видит приступ',
    '010': 'Только эксперт B видит приступ',
    '011': 'Эксперты B и C видят приступ',
    '100': 'Только эксперт A видит приступ',
    '101': 'Эксперты A и C видят приступ',
    '110': 'Эксперты A и B видят приступ',
    '111': 'Приступ (все эксперты согласны)',
}


# ============================================================================
# ЗАГРУЗКА АННОТАЦИЙ ЭКСПЕРТОВ
# ============================================================================

def load_expert_annotations(patient_id, csv_path):
    """
    Загружает аннотации экспертов из исходных CSV файлов.
    
    Аргументы:
        patient_id: номер пациента (1-79)
        csv_path: путь к папке с CSV файлами
    
    Возвращает:
        a_data, b_data, c_data: массивы с оценками экспертов по эпохам
    """
    # Загрузка CSV файлов
    A = pd.read_csv(Path(csv_path) / "annotations_2017_A.csv", header=None)
    B = pd.read_csv(Path(csv_path) / "annotations_2017_B.csv", header=None)
    C = pd.read_csv(Path(csv_path) / "annotations_2017_C.csv", header=None)
    
    # Пропускаем первую строку (номера сигналов)
    A = A.iloc[1:].reset_index(drop=True)
    B = B.iloc[1:].reset_index(drop=True)
    C = C.iloc[1:].reset_index(drop=True)
    
    # Преобразуем в числа
    A = A.apply(pd.to_numeric, errors='coerce')
    B = B.apply(pd.to_numeric, errors='coerce')
    C = C.apply(pd.to_numeric, errors='coerce')
    
    # Берем колонку пациента
    a_data = A.iloc[:, patient_id - 1].values
    b_data = B.iloc[:, patient_id - 1].values
    c_data = C.iloc[:, patient_id - 1].values
    
    return a_data, b_data, c_data


def create_annotations(a_data, b_data, c_data, target_sr, epoch_sec=2, window_sec=4, stride_sec=1):
    """
    Создает MNE аннотации на основе оценок экспертов.
    
    Аргументы:
        a_data, b_data, c_data: массивы оценок по эпохам
        target_sr: целевая частота дискретизации
        epoch_sec: длительность эпохи (сек)
        window_sec: длительность окна для аннотации (сек)
        stride_sec: шаг между окнами (сек)
    
    Возвращает:
        mne.Annotations: аннотации для визуализации
    """
    samples_per_epoch = int(target_sr * epoch_sec)
    n_epochs = len(a_data)
    n_samples = n_epochs * samples_per_epoch
    
    # Разворачиваем в непрерывный ряд
    expert_A = np.zeros(n_samples)
    expert_B = np.zeros(n_samples)
    expert_C = np.zeros(n_samples)
    
    for epoch in range(n_epochs):
        start = epoch * samples_per_epoch
        end = start + samples_per_epoch
        if end > n_samples:
            break
        expert_A[start:end] = a_data[epoch] if not np.isnan(a_data[epoch]) else 0
        expert_B[start:end] = b_data[epoch] if not np.isnan(b_data[epoch]) else 0
        expert_C[start:end] = c_data[epoch] if not np.isnan(c_data[epoch]) else 0
    
    # Создаем окна и аннотации
    window_size = int(target_sr * window_sec)
    stride = int(target_sr * stride_sec)
    
    onsets = []
    durations = []
    descriptions = []
    patterns_list = []
    
    for start in range(0, n_samples - window_size, stride):
        end = start + window_size
        
        A_win = expert_A[start:end]
        B_win = expert_B[start:end]
        C_win = expert_C[start:end]
        
        if np.any(np.isnan(A_win)) or np.any(np.isnan(B_win)) or np.any(np.isnan(C_win)):
            continue
        
        mid = start + window_size // 2
        pattern = f"{int(expert_A[mid])}{int(expert_B[mid])}{int(expert_C[mid])}"
        
        onset_sec = start / target_sr
        
        onsets.append(onset_sec)
        durations.append(window_sec)
        descriptions.append(pattern)
        patterns_list.append(pattern)
    
    # Статистика
    print(f"\n📊 Статистика аннотаций:")
    if patterns_list:
        unique, counts = np.unique(patterns_list, return_counts=True)
        for p, c in zip(unique, counts):
            color_code = PATTERN_COLORS.get(p, '#000000')
            desc = PATTERN_DESCRIPTIONS.get(p, p)
            print(f"   {p} ({color_code}): {c} ({100*c/len(patterns_list):.1f}%) - {desc}")
    
    return mne.Annotations(onset=onsets, duration=durations, description=descriptions)


# ============================================================================
# ЗАГРУЗКА ПРЕДСКАЗАНИЙ МОДЕЛИ
# ============================================================================

def load_model_predictions(patient_id, model_path, data_path, target_sr):
    """
    Загружает предсказания обученной модели для пациента.
    
    Аргументы:
        patient_id: номер пациента
        model_path: путь к файлу модели (.pth)
        data_path: путь к обработанным данным (.pkl)
        target_sr: частота дискретизации
    
    Возвращает:
        predictions: массив с предсказаниями для каждого окна
    """
    import pickle
    
    # Загружаем тестовые данные
    pkl_file = Path(data_path) / "helsinki_test.pkl"
    if not pkl_file.exists():
        print(f"⚠️ Файл с данными не найден: {pkl_file}")
        return None
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # Ищем окна для нужного пациента
    patient_windows = []
    patient_metadata = []
    
    for i, meta in enumerate(data['metadata']):
        if meta['patient'] == patient_id:
            patient_windows.append(data['X'][i])
            patient_metadata.append(meta)
    
    if not patient_windows:
        print(f"⚠️ Пациент {patient_id} не найден в тестовых данных")
        return None
    
    # Загружаем модель
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = EEGNeX(
        n_chans=Config.N_CHANNELS,
        n_outputs=Config.EEGNEX_N_OUTPUTS,
        n_times=Config.WINDOW_SIZE,
        sfreq=target_sr,
        input_window_seconds=Config.WINDOW_SEC,
        activation=Config.EEGNEX_ACTIVATION,
        filter_1=Config.EEGNEX_FILTER_1,
        filter_2=Config.EEGNEX_FILTER_2,
        drop_prob=Config.EEGNEX_DROP_PROB
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Делаем предсказания
    predictions = []
    
    with torch.no_grad():
        for window in patient_windows:
            X = torch.FloatTensor(window).unsqueeze(0).to(device)
            logits = model(X)
            prob = torch.sigmoid(logits).item()
            predictions.append(prob)
    
    # Создаем аннотации для предсказаний
    predictions_annotations = []
    for i, (meta, pred) in enumerate(zip(patient_metadata, predictions)):
        # Определяем цвет на основе вероятности
        if pred >= 0.7:
            color = 'red'
        elif pred >= 0.3:
            color = 'orange'
        else:
            color = 'blue'
        
        desc = f"Pred: {pred:.2f} ({color})"
        predictions_annotations.append({
            'onset': meta['window_start'] / target_sr,
            'duration': Config.WINDOW_SEC,
            'description': desc,
            'probability': pred
        })
    
    print(f"\n🤖 Загружено {len(predictions)} предсказаний для пациента {patient_id}")
    print(f"   Средняя вероятность: {np.mean(predictions):.3f}")
    
    return predictions_annotations


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ ВИЗУАЛИЗАЦИИ
# ============================================================================

def view_patient(patient_id, duration=30, n_channels=19, model_path=None):
    """
    Визуализирует ЭЭГ пациента с аннотациями экспертов и модели.
    
    Аргументы:
        patient_id: номер пациента (1-79)
        duration: длительность отображения (сек)
        n_channels: количество каналов для отображения
        model_path: путь к модели для показа предсказаний (опционально)
    """
    
    print(f"\n{'='*70}")
    print(f"🧠 ВИЗУАЛИЗАЦИЯ ПАЦИЕНТА {patient_id}")
    print(f"{'='*70}")
    
    # Пути к данным
    data_path = Path(Config.EDF_PATH)
    edf_file = data_path / f"eeg{patient_id}.edf"
    csv_path = data_path
    
    # Проверка существования файлов
    if not edf_file.exists():
        print(f"❌ Файл не найден: {edf_file}")
        print(f"   Проверьте путь в config.py: EDF_PATH = {Config.EDF_PATH}")
        return
    
    # 1. Загружаем аннотации экспертов
    print(f"\n📁 Загрузка аннотаций экспертов...")
    a_data, b_data, c_data = load_expert_annotations(patient_id, csv_path)
    print(f"   Эпох в CSV: {len(a_data)}")
    
    # 2. Загружаем EDF
    print(f"\n📁 Загрузка EDF: {edf_file.name}")
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    orig_sr = int(raw.info['sfreq'])
    print(f"   Исходная частота: {orig_sr} Гц")
    print(f"   Длительность: {raw.n_times / orig_sr:.1f} сек")
    print(f"   Каналов: {len(raw.ch_names)}")
    
    # 3. Ресемплинг
    if orig_sr != Config.TARGET_SR:
        print(f"\n🔄 Ресемплинг {orig_sr} → {Config.TARGET_SR} Гц")
        raw.resample(Config.TARGET_SR)
    
    # 4. Фильтрация
    print(f"\n🔧 Фильтрация {Config.LOWCUT}-{Config.HIGHCUT} Гц")
    raw.filter(Config.LOWCUT, Config.HIGHCUT, fir_design='firwin')
    
    # 5. Создаем аннотации экспертов
    annotations = create_annotations(
        a_data, b_data, c_data, 
        Config.TARGET_SR, 
        epoch_sec=Config.EPOCH_SEC,
        window_sec=Config.WINDOW_SEC,
        stride_sec=1
    )
    raw.set_annotations(annotations)
    print(f"\n🏷️ Добавлено {len(annotations)} аннотаций экспертов")
    
    # 6. Загружаем предсказания модели (если указана)
    model_annotations = None
    if model_path:
        print(f"\n🤖 Загрузка предсказаний модели...")
        model_annotations = load_model_predictions(
            patient_id, model_path, Config.OUTPUT_PATH, Config.TARGET_SR
        )
        
        # Добавляем предсказания как отдельные аннотации
        if model_annotations:
            # Создаем копию raw для предсказаний
            raw_model = raw.copy()
            
            model_onsets = [a['onset'] for a in model_annotations]
            model_durations = [a['duration'] for a in model_annotations]
            model_descriptions = [a['description'] for a in model_annotations]
            
            model_ann = mne.Annotations(
                onset=model_onsets,
                duration=model_durations,
                description=model_descriptions
            )
            raw_model.set_annotations(model_ann)
            
            # Визуализируем с предсказаниями
            print(f"\n📊 Запуск визуализации с ПРЕДСКАЗАНИЯМИ МОДЕЛИ...")
            raw_model.plot(
                duration=duration,
                n_channels=n_channels,
                scalings="auto",
                title=f"Пациент {patient_id} | 🔴=приступ (эксперты) | 📊=предсказания модели",
                block=True,
                bgcolor='white',
                color='black'
            )
            return
    
    # 7. Визуализация без предсказаний
    print(f"\n📊 Запуск визуализации...")
    print(f"   Цветовая кодировка паттернов:")
    for pattern, color in PATTERN_COLORS.items():
        desc = PATTERN_DESCRIPTIONS.get(pattern, pattern)
        print(f"   {pattern} ({color}): {desc[:40]}...")
    
    raw.plot(
        duration=duration,
        n_channels=n_channels,
        scalings="auto",
        title=f"Пациент {patient_id} - Аннотации экспертов (цвет = паттерн)",
        block=True,
        bgcolor='white',
        color='black'
    )
    
    print("\n✅ Визуализация завершена")


# ============================================================================
# ДОПОЛНИТЕЛЬНЫЕ УТИЛИТЫ
# ============================================================================

def list_available_patients():
    """Выводит список доступных пациентов с EDF файлами"""
    data_path = Path(Config.EDF_PATH)
    if not data_path.exists():
        print(f"❌ Папка не найдена: {data_path}")
        return []
    
    edf_files = list(data_path.glob("eeg*.edf"))
    patients = []
    
    for f in edf_files:
        try:
            patient_id = int(f.stem[3:])
            patients.append(patient_id)
        except:
            continue
    
    patients.sort()
    print(f"\n📋 Доступные пациенты ({len(patients)}):")
    print(f"   {patients[:20]}{'...' if len(patients) > 20 else ''}")
    print(f"   Всего: {len(patients)} EDF файлов")
    
    return patients


def patient_info(patient_id):
    """Выводит информацию о пациенте"""
    data_path = Path(Config.EDF_PATH)
    edf_file = data_path / f"eeg{patient_id}.edf"
    
    if not edf_file.exists():
        print(f"❌ Пациент {patient_id} не найден")
        return
    
    raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
    
    print(f"\n📊 ИНФОРМАЦИЯ О ПАЦИЕНТЕ {patient_id}")
    print(f"{'='*50}")
    print(f"   Длительность: {raw.n_times / raw.info['sfreq']:.1f} сек")
    print(f"   Частота: {raw.info['sfreq']} Гц")
    print(f"   Каналов: {len(raw.ch_names)}")
    print(f"   Каналы: {', '.join(raw.ch_names[:10])}{'...' if len(raw.ch_names) > 10 else ''}")


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Визуализация ЭЭГ с аннотациями экспертов и предсказаниями модели',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ПРИМЕРЫ:
    # Базовый просмотр
    python edf_viewer.py --patient 16
    
    # Просмотр с предсказаниями модели
    python edf_viewer.py --patient 41 --model models/best_eegnex_model.pth
    
    # Длительный просмотр (60 сек)
    python edf_viewer.py --patient 16 --duration 60
    
    # Список доступных пациентов
    python edf_viewer.py --list
    
    # Информация о пациенте
    python edf_viewer.py --patient 16 --info
        """
    )
    
    parser.add_argument('--patient', type=int, help='Номер пациента (1-79)')
    parser.add_argument('--duration', type=int, default=30, help='Длительность отображения (сек, по умолчанию 30)')
    parser.add_argument('--channels', type=int, default=19, help='Количество каналов (по умолчанию 19)')
    parser.add_argument('--model', type=str, help='Путь к модели для отображения предсказаний')
    parser.add_argument('--list', action='store_true', help='Показать список доступных пациентов')
    parser.add_argument('--info', action='store_true', help='Показать информацию о пациенте')
    
    args = parser.parse_args()
    
    # Список пациентов
    if args.list:
        list_available_patients()
        sys.exit(0)
    
    # Информация о пациенте
    if args.info and args.patient:
        patient_info(args.patient)
        sys.exit(0)
    
    # Визуализация
    if args.patient:
        view_patient(args.patient, args.duration, args.channels, args.model)
    else:
        parser.print_help()