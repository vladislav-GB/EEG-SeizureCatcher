"""
preprocessing.py - Подготовка данных для EEGNeX v5.0

НАЗНАЧЕНИЕ:
    Преобразование сырых EDF файлов и аннотаций экспертов в формат,
    готовый для обучения нейросети EEGNeX.

НОВШЕСТВА v5.0:
    - Окна 3 секунды (вместо 5) — лучше для коротких неонатальных приступов
    - Target окна = центральная (2-я) секунда — точная локализация
    - Soft targets: 0.0 (000) / 0.6 (1 эксперт) / 0.9 (2 эксперта) / 1.0 (111)
    - Все weights = 1.0 — внешние веса отключены, балансировка в Focal Loss
    - Убраны κ-веса пациентов, позиционные веса, посекундные веса

АЛГОРИТМ:
    1. Выборки (train/val/test).
    2. Для каждого пациента:
       a) Загрузить EDF файл (с безопасным чтением)
       b) Извлечь 19 каналов ЭЭГ по системе 10-20
       c) Применить фильтрацию (0.5-30 Гц + режектор 50 Гц)
       d) Ресемплинг до 256 Гц
       e) Z-нормализация каждого канала
       f) Загрузить экспертные оценки (A, B, C)
       g) Нарезать окна (3 сек, шаг 1 сек, перекрытие 2 сек)
       h) Рассчитать target по центральной секунде:
          - 3 эксперта (111) → 1.0
          - 2 эксперта → 0.9
          - 1 эксперт → 0.6
          - 0 экспертов (000) → 0.0
       i) weight всегда = 1.0
    3. Сохранить train/val/test .pkl файлы

ВХОДНЫЕ ФАЙЛЫ:
    - EDF: /media/avengus/Локальный диск/Dev/EEG/helsinki/eeg{id}.edf
    - CSV: annotations_2017_A.csv, annotations_2017_B.csv, annotations_2017_C.csv

ВЫХОДНЫЕ ФАЙЛЫ:
    - processed/train_patients/patient_*.pkl
    - processed/val_patients/patient_*.pkl
    - processed/test_patients/patient_*.pkl

КАЖДЫЙ .PKL ФАЙЛ СОДЕРЖИТ:
    - X: [N, 19, 768] — сигналы (N окон, 19 каналов, 3 сек × 256 Гц)
    - targets: [N] — soft labels (0.0, 0.6, 0.9, 1.0)
    - weights: [N] — всегда 1.0 (игнорируется в v5.0)

ИСПОЛЬЗОВАНИЕ:
    python "1. Preprocessing/preprocessing.py"
"""

import sys
import os
import gc
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import mne
import psutil
from scipy.signal import butter, filtfilt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from Config.config import Config

warnings.filterwarnings("ignore")

# ============================================================
# ЛОГИРОВАНИЕ
# ============================================================

log_dir = Config.LOG_PREPROCESSING_PATH
log_dir.mkdir(parents=True, exist_ok=True)

log_filename = log_dir / f"preprocessing_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_file = open(log_filename, "w", encoding="utf-8")


def log(msg, indent=False):
    timestamp = datetime.now().strftime('%H:%M:%S')
    line = f"[{timestamp}]    {msg}" if indent else f"[{timestamp}] {msg}"
    print(line)
    log_file.write(line + "\n")
    log_file.flush()


def log_memory(stage=""):
    mem = psutil.Process().memory_info().rss / 1024 / 1024
    log(f"💾 {stage}: {mem:.0f} MB")


# ============================================================
# ФИЛЬТРЫ
# ============================================================

def create_filters(sr):
    nyq = 0.5 * sr

    b_band, a_band = butter(4, [Config.LOWCUT / nyq, Config.HIGHCUT / nyq], btype='band')
    b_notch, a_notch = butter(4, [(Config.NOTCH - 1) / nyq, (Config.NOTCH + 1) / nyq], btype='bandstop')

    return (b_band, a_band), (b_notch, a_notch)


def process_signal(signal, filters):
    (b_band, a_band), (b_notch, a_notch) = filters
    signal = filtfilt(b_band, a_band, signal)
    signal = filtfilt(b_notch, a_notch, signal)
    signal = (signal - signal.mean()) / (signal.std() + 1e-8)
    return signal


# ============================================================
# TARGET
# ============================================================

def get_second_target(a, b, c):
    s = a + b + c
    if s == 3:
        return 1.0
    elif s == 2:
        return 0.9
    elif s == 1:
        return 0.6
    return 0.0


# ============================================================
# ОКНА (v5.3: mean pooling)
# ============================================================

def generate_windows(signals, A, B, C, sfreq):
    sec_targets = [
        get_second_target(a, b, c)
        for a, b, c in zip(A, B, C)
    ]

    windows, targets = [], []

    for start in range(0, len(sec_targets) - Config.WINDOW_SEC + 1, Config.STRIDE_SEC):
        end = start + Config.WINDOW_SEC

        start_s = start * sfreq
        end_s = end * sfreq

        window = signals[:, start_s:end_s]
        t = np.mean(sec_targets[start:end])

        windows.append(window)
        targets.append(t)

    return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32)


# ============================================================
# ПАЦИЕНТ
# ============================================================

def process_patient(pid, df, filters):
    edf_path = Path(Config.DATA_PATH) / f"eeg{pid}.edf"

    if not edf_path.exists():
        return None

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except:
        return None

    if int(raw.info['sfreq']) != Config.TARGET_SR:
        raw.resample(Config.TARGET_SR)

    sfreq = Config.TARGET_SR

    signals = []

    for ch in Config.CHANNELS:
        idx = None
        for i, name in enumerate(raw.ch_names):
            if ch.lower() in name.lower():
                idx = i
                break

        if idx is None:
            return None

        sig = raw.get_data(picks=idx)[0]
        sig = process_signal(sig, filters)
        signals.append(sig)

    signals = np.stack(signals)

    n_sec = signals.shape[1] // sfreq
    df = df.iloc[:n_sec]

    if n_sec < Config.WINDOW_SEC:
        return None

    X, y = generate_windows(
        signals,
        df['A'].values,
        df['B'].values,
        df['C'].values,
        sfreq
    )

    return X, y


# ============================================================
# СПЛИТ
# ============================================================

def process_split(patients, name):
    log("")
    log(f"=== {name.upper()} ===")

    out_dir = Path(Config.OUTPUT_PATH) / f"{name}_patients"
    out_dir.mkdir(parents=True, exist_ok=True)

    filters = create_filters(Config.TARGET_SR)

    A = pd.read_csv(Path(Config.DATA_PATH) / "annotations_2017_A.csv", header=None)
    B = pd.read_csv(Path(Config.DATA_PATH) / "annotations_2017_B.csv", header=None)
    C = pd.read_csv(Path(Config.DATA_PATH) / "annotations_2017_C.csv", header=None)

    total_windows = 0
    total_sz = 0
    processed = 0

    for pid in tqdm(patients, desc=name):
        try:
            col = list(A.iloc[0]).index(pid)
        except:
            continue

        a = A.iloc[1:, col]
        b = B.iloc[1:, col]
        c = C.iloc[1:, col]

        valid = (~a.isna()) & (~b.isna()) & (~c.isna())

        df = pd.DataFrame({
            'A': a[valid].astype(int),
            'B': b[valid].astype(int),
            'C': c[valid].astype(int)
        })

        result = process_patient(pid, df, filters)

        if result is None:
            continue

        X, y = result

        np.savez_compressed(out_dir / f"patient_{pid}.npz", X=X, y=y)

        total_windows += len(X)
        total_sz += (y >= 0.5).sum()
        processed += 1

        log(f"✅ Patient {pid:2d}: {len(X):5d} окон, приступов: {(y>=0.5).sum():5d}", indent=True)

        gc.collect()

    log("")
    log(f"ИТОГО {name}:", indent=True)
    log(f"   Обработано: {processed}/{len(patients)} пациентов", indent=True)
    log(f"   Всего окон: {total_windows}", indent=True)
    log(f"   Окон с приступом: {total_sz} ({100*total_sz/total_windows:.1f}%)", indent=True)

    return processed


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    Config.set_seed()

    log("="*60)
    log("PREPROCESSING v5")
    log("="*60)
    log(f"Window: {Config.WINDOW_SEC}s | Stride: {Config.STRIDE_SEC}s")
    log(f"Soft targets: 0.0 / 0.6 / 0.9 / 1.0 (mean pooling)")
    log(f"Output: .npz compressed")
    log("="*60)

    log_memory("Начало")

    process_split(Config.TRAIN_PATIENTS, "train")
    process_split(Config.VAL_PATIENTS, "val")
    process_split(Config.TEST_PATIENTS, "test")

    log("")
    log("="*60)
    log("PREPROCESSING ЗАВЕРШЁН")
    log("="*60)

    log_memory("Конец")
    log_file.close()

    print(f"✅ Лог сохранён: {log_filename}")