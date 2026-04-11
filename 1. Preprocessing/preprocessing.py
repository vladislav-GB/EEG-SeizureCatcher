"""
preprocessing.py - Подготовка данных для EEGNeX

НАЗНАЧЕНИЕ:
    Преобразование сырых EDF файлов и аннотаций экспертов в формат,
    готовый для обучения нейросети EEGNeX.

АЛГОРИТМ:
    1. Загрузить выборки (train/val/test) из CSV файлов, созданных R-скриптом
    2. Для каждого пациента:
       a) Загрузить EDF файл (с безопасным чтением)
       b) Извлечь 19 каналов ЭЭГ по системе 10-20
       c) Применить фильтрацию (0.5-30 Гц + режектор 50 Гц)
       d) Ресемплинг до 256 Гц
       e) Z-нормализация
       f) Загрузить экспертные оценки (A, B, C)
       g) Нарезать окна (5 сек, шаг 1 сек, перекрытие 4 сек)
       h) Рассчитать soft target = (A+B+C)/3
       i) Рассчитать weight = 0.33 (1 эксперт), 0.67 (2 эксперта), 1.0 (3 эксперта)
       j) Усреднить target и weight по окну с весами позиций [0.1,0.2,0.4,0.2,0.1]
    3. Сохранить train/val/test .pkl файлы

ВХОДНЫЕ ФАЙЛЫ:
    - EDF: /media/avengus/Локальный диск/Dev/EEG/helsinki/eeg{id}.edf
    - CSV: /media/avengus/Локальный диск/Dev/EEG/helsinki/annotations_2017_A.csv
    - CSV: /media/avengus/Локальный диск/Dev/EEG/helsinki/annotations_2017_B.csv
    - CSV: /media/avengus/Локальный диск/Dev/EEG/helsinki/annotations_2017_C.csv

ВЫХОДНЫЕ ФАЙЛЫ:
    - helsinki_train.pkl
    - helsinki_val.pkl
    - helsinki_test.pkl

КАЖДЫЙ .PKL ФАЙЛ СОДЕРЖИТ:
    - X: [N, 19, 1280] — сигналы (N окон, 19 каналов, 5 сек × 256 Гц)
    - targets: [N] — soft labels (0, 0.33, 0.67, 1.0)
    - weights: [N] — веса окон (0.33, 0.67, 1.0)
"""

import sys
import os
import gc
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import mne
import psutil
from scipy.signal import butter, filtfilt, resample
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from Config.config import Config

warnings.filterwarnings('ignore')

# ============================================================================
# ЛОГИРОВАНИЕ
# ============================================================================

log_filename = f"preprocessing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_file = open(log_filename, 'w', encoding='utf-8')

def log(msg):
    timestamp = datetime.now().strftime('%H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line)
    log_file.write(line + '\n')
    log_file.flush()

def log_memory(stage=""):
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    log(f"💾 {stage}: {mem_mb:.0f} MB")

# ============================================================================
# БЕЗОПАСНОЕ ЧТЕНИЕ EDF
# ============================================================================

def read_edf_safe(edf_path):
    """
    Безопасное чтение EDF файла.
    При ошибке возвращает None (пациент пропускается).
    """
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        return raw
    except Exception:
        return None

# ============================================================================
# ФИЛЬТРАЦИЯ
# ============================================================================

def create_filters(sr):
    """
    Создаёт коэффициенты фильтров Баттерворта 4-го порядка.
    
    Возвращает:
        band_filter: полосовой фильтр 0.5-30 Гц
        notch_filter: режекторный фильтр 50 Гц (ширина 2 Гц)
    """
    nyquist = 0.5 * sr
    
    # Полосовой фильтр 0.5-30 Гц
    low = Config.LOWCUT / nyquist
    high = Config.HIGHCUT / nyquist
    b_band, a_band = butter(4, [low, high], btype='band')
    
    # Режекторный фильтр 50 Гц (ширина ±1 Гц)
    notch_low = (Config.NOTCH - 1) / nyquist
    notch_high = (Config.NOTCH + 1) / nyquist
    b_notch, a_notch = butter(4, [notch_low, notch_high], btype='bandstop')
    
    return (b_band, a_band), (b_notch, a_notch)


def process_signal(signal, orig_sr, target_sr, filters):
    """
    Полная обработка одноканального сигнала.
    
    Порядок операций:
        1. Полосовая фильтрация 0.5-30 Гц
        2. Режекторный фильтр 50 Гц
        3. Z-нормализация
    """
    (b_band, a_band), (b_notch, a_notch) = filters
    
    # Шаг 1-2: фильтрация
    signal = filtfilt(b_band, a_band, signal)
    signal = filtfilt(b_notch, a_notch, signal)
    
    # Шаг 3: Z-нормализация
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    
    return signal

# ============================================================================
# ВЕСА ДЛЯ СЕКУНДЫ
# ============================================================================

def get_second_target_and_weight(a, b, c):
    """
    Вычисляет target и weight для одной секунды.
    
    target = (A+B+C)/3 — вероятность приступа
    weight = 1.0 (3 эксперта), 0.67 (2 эксперта), 0.33 (1 эксперт)
    """
    n_positive = a + b + c
    target = n_positive / 3.0
    
    # Используем веса из Config
    if n_positive == 0:
        weight = Config.WEIGHT_BACKGROUND  
    elif n_positive == 3:
        weight = Config.WEIGHT_ALL_AGREE   
    elif n_positive == 2:
        weight = Config.WEIGHT_TWO_AGREE   
    else:  # n_positive == 1
        weight = Config.WEIGHT_ONE_AGREE   
    
    return target, weight

# ============================================================================
# ГЕНЕРАЦИЯ ОКОН ДЛЯ ПАЦИЕНТА
# ============================================================================

def generate_windows_for_patient(signals, annotations_A, annotations_B, annotations_C, sfreq):
    """
    Генерирует окна для одного пациента.
    
    Параметры:
        signals: [channels, time] — сигнал ЭЭГ
        annotations_A/B/C: массивы оценок экспертов (0/1) по секундам
        sfreq: частота дискретизации
    
    Возвращает:
        windows: [n_windows, channels, time]
        targets: [n_windows] — soft labels
        weights: [n_windows] — веса окон
    """
    window_sec = Config.WINDOW_SEC
    pos_weights = Config.POS_WEIGHTS
    
    n_seconds = len(annotations_A)
    n_samples = signals.shape[1]
    n_sec_actual = min(n_seconds, n_samples // sfreq)
    
    # Посекундные target и weight
    sec_targets = []
    sec_weights = []
    for i in range(n_sec_actual):
        t, w = get_second_target_and_weight(
            annotations_A[i], annotations_B[i], annotations_C[i]
        )
        sec_targets.append(t)
        sec_weights.append(w)
    
    windows, targets, weights = [], [], []
    
    for start_sec in range(0, n_sec_actual - window_sec + 1, Config.STRIDE_SEC):
        end_sec = start_sec + window_sec
        
        start_sample = start_sec * sfreq
        end_sample = end_sec * sfreq
        
        window_data = signals[:, start_sample:end_sample]
        window_targets = sec_targets[start_sec:end_sec]
        window_weights = sec_weights[start_sec:end_sec]
        
        # Взвешенное среднее (центр важнее краёв)
        weighted_target = np.average(window_targets, weights=pos_weights)
        weighted_weight = np.average(window_weights, weights=pos_weights)
        
        windows.append(window_data)
        targets.append(weighted_target)
        weights.append(weighted_weight)
    
    if len(windows) == 0:
        return np.array([]), np.array([]), np.array([])
    
    return np.array(windows), np.array(targets), np.array(weights)

# ============================================================================
# ОБРАБОТКА ПАЦИЕНТА
# ============================================================================

def process_patient(patient_id, patient_data, filters):
    """
    Обрабатывает одного пациента.
    
    Аргументы:
        patient_id: номер пациента (1-79)
        patient_data: DataFrame с колонками expert_A, expert_B, expert_C
        filters: коэффициенты фильтров
    
    Возвращает:
        dict с ключами X, targets, weights
        или None, если пациент пропущен
    """
    edf_file = Path(Config.DATA_PATH) / f"eeg{patient_id}.edf"
    if not edf_file.exists():
        log(f"⚠️ Пациент {patient_id}: EDF не найден")
        return None
    
    raw = read_edf_safe(edf_file)
    if raw is None:
        log(f"⚠️ Пациент {patient_id}: не удалось прочитать EDF")
        return None
    
    orig_sfreq = int(raw.info['sfreq'])
    
    # Ресемплинг до целевой частоты через MNE
    if orig_sfreq != Config.TARGET_SR:
        raw = raw.resample(Config.TARGET_SR, npad='auto')
        log(f"   Пациент {patient_id}: ресемплинг {orig_sfreq} → {Config.TARGET_SR} Гц")
    
    sfreq = Config.TARGET_SR
    
    # Извлечение каналов
    ch_names = [ch.upper() for ch in raw.ch_names]
    signals = []
    
    for target_ch in Config.CHANNELS:
        idx = None
        for i, name in enumerate(ch_names):
            clean_name = name.replace('-REF', '').replace('-Ref', '').replace('EEG ', '')
            if target_ch.upper() in clean_name.upper():
                idx = i
                break
        if idx is None:
            log(f"❌ Пациент {patient_id}: канал {target_ch} не найден")
            return None
        
        data, _ = raw[idx, :]
        sig = data[0].astype(np.float64)
        sig = process_signal(sig, sfreq, Config.TARGET_SR, filters)
        signals.append(sig)
    
    signals = np.stack(signals, axis=0)
    
    # Обрезаем аннотации до длины сигнала
    n_samples = signals.shape[1]
    n_sec = n_samples // sfreq
    patient_data = patient_data.iloc[:n_sec]
    
    if n_sec < Config.WINDOW_SEC:
        log(f"⚠️ Пациент {patient_id}: сигнал слишком короткий ({n_sec} сек < {Config.WINDOW_SEC} сек)")
        return None
    
    # Генерация окон
    windows, targets, weights = generate_windows_for_patient(
        signals,
        patient_data['expert_A'].values,
        patient_data['expert_B'].values,
        patient_data['expert_C'].values,
        sfreq
    )
    
    if len(windows) == 0:
        log(f"⚠️ Пациент {patient_id}: не сгенерировано окон")
        return None
    
    # Очистка памяти
    del raw, signals
    gc.collect()
    
    return {
        'X': windows.astype(np.float32),
        'targets': targets.astype(np.float32),
        'weights': weights.astype(np.float32)
    }

# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def process_split(patients, split_name):
    """
    Обрабатывает выборку (train/val/test).
    
    Аргументы:
        patients: список номеров пациентов
        split_name: 'train', 'val' или 'test'
    """
    filters = create_filters(Config.TARGET_SR)

    # Создаём выходную директорию
    Path(Config.OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    
    # Загрузка CSV с аннотациями
    A = pd.read_csv(Path(Config.DATA_PATH) / "annotations_2017_A.csv", header=None)
    B = pd.read_csv(Path(Config.DATA_PATH) / "annotations_2017_B.csv", header=None)
    C = pd.read_csv(Path(Config.DATA_PATH) / "annotations_2017_C.csv", header=None)
    
    all_X, all_targets, all_weights = [], [], []
    processed_ids = []

    for patient_id in tqdm(patients, desc=f"Processing {split_name}"):
        # Находим колонку пациента
        col_idx = None
        for i, val in enumerate(A.iloc[0].values):
            if val == patient_id:
                col_idx = i
                break
        
        if col_idx is None:
            log(f"⚠️ Пациент {patient_id} не найден в CSV")
            continue
        
        # Извлекаем данные
        a_data = A.iloc[1:, col_idx].values
        b_data = B.iloc[1:, col_idx].values
        c_data = C.iloc[1:, col_idx].values
        
        # Обрезаем по NaN
        valid_len = 0
        for i in range(len(a_data)):
            if pd.isna(a_data[i]) or pd.isna(b_data[i]) or pd.isna(c_data[i]):
                break
            valid_len += 1
        
        if valid_len == 0:
            log(f"⚠️ Пациент {patient_id} не имеет валидных аннотаций")
            continue
        
        patient_df = pd.DataFrame({
            'expert_A': a_data[:valid_len].astype(int),
            'expert_B': b_data[:valid_len].astype(int),
            'expert_C': c_data[:valid_len].astype(int)
        })
        
        result = process_patient(patient_id, patient_df, filters)
        if result:
            all_X.append(result['X'])
            all_targets.append(result['targets'])
            all_weights.append(result['weights'])
            processed_ids.append(patient_id)
            log(f"✅ Пациент {patient_id}: {len(result['X'])} окон")
        else:
            log(f"⚠️ Пациент {patient_id} пропущен (ошибка чтения EDF)")
        
        # Принудительный сбор мусора
        gc.collect()
    
    if not all_X:
        log(f"❌ Нет данных для {split_name}")
        return
    
    # Объединение
    X = np.concatenate(all_X, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    weights = np.concatenate(all_weights, axis=0)
    
    # Сохранение
    output_file = Path(Config.OUTPUT_PATH) / f"helsinki_{split_name}.pkl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump({
            'X': X,
            'targets': targets,
            'weights': weights,
            'metadata': {
                'split': split_name,
                'patients': processed_ids,
                'n_windows': len(X),
                'window_sec': Config.WINDOW_SEC,
                'stride_sec': Config.STRIDE_SEC,
                'target_sr': Config.TARGET_SR,
                'channels': Config.CHANNELS,
                'created': datetime.now().isoformat()
            }
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    log(f"\n✅ {split_name}: {len(X)} окон от {len(processed_ids)}/{len(patients)} пациентов")
    log(f"   target mean: {targets.mean():.3f}, std: {targets.std():.3f}")
    log(f"   weight mean: {weights.mean():.3f}, std: {weights.std():.3f}")
    
    if len(processed_ids) < len(patients):
        missing = set(patients) - set(processed_ids)
        log(f"   ⚠️ Пропущены: {sorted(missing)}")

# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    Config.set_seed()
    
    log("="*60)
    log("ПРЕПРОЦЕССИНГ")
    log("="*60)
    log(f"SEED: {Config.SEED}")
    log(f"Окна: {Config.WINDOW_SEC} сек, шаг {Config.STRIDE_SEC} сек")
    log(f"Веса позиций: {Config.POS_WEIGHTS}")
    log(f"Частота: {Config.TARGET_SR} Гц")
    log(f"Фильтры: {Config.LOWCUT}-{Config.HIGHCUT} Гц, режектор {Config.NOTCH} Гц")
    log("="*60)
    
    # Обработка выборок
    process_split(Config.TRAIN_PATIENTS, 'train')
    process_split(Config.VAL_PATIENTS, 'val')
    process_split(Config.TEST_PATIENTS, 'test')
    
    log("\n✅ ПРЕПРОЦЕССИНГ ЗАВЕРШЁН!")
    log_file.close()
    
    print(f"\nЛог сохранён: {log_filename}")