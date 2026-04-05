"""
preprocessing.py - Подготовка данных для EEGNeX

НАЗНАЧЕНИЕ:
    Преобразование сырых EDF файлов и аннотаций экспертов в формат,
    готовый для обучения нейросети EEGNeX.

АЛГОРИТМ:
    1. Загрузить выборки (train/val/test) из CSV файлов, созданных R-скриптом

    2. Для каждого пациента:
       a) Загрузить EDF файл
       b) Извлечь 19 каналов ЭЭГ
       c) Применить фильтрацию и нормализацию
       d) Загрузить экспертные оценки для нужных эпох
       e) Нарезать окна (4 сек, шаг 1 сек)
       f) Рассчитать soft labels, hard labels, confidence
       g) Рассчитать веса: 
       pattern_weight × (0.3 + 0.7 * agreement) × (0.5 + 0.5 * confidence)

    3. Сохранить train/val/test .pkl файлы

ВХОДНЫЕ ФАЙЛЫ:
    - EDF: eeg{id}.edf
    - CSV: R_results/helsinki_EDFchek/train_sample.csv
    - CSV: R_results/helsinki_EDFchek/val_sample.csv
    - CSV: R_results/helsinki_EDFchek/test_sample.csv

ВЫХОДНЫЕ ФАЙЛЫ:
    - helsinki_train.pkl
    - helsinki_val.pkl
    - helsinki_test.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import mne
from scipy.signal import butter, filtfilt
from config import Config

# ============================================================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def create_filters(sr):
    """
    Создает коэффициенты фильтров Баттерворта 4-го порядка ("шумодав" для мозговых волн).
    
    Возвращает:
        band_filter: полосовой фильтр 0.5-30 Гц
        notch_filter: режекторный фильтр 50 Гц
    """
    nyquist = 0.5 * sr
    
    # Полосовой фильтр
    low = Config.LOWCUT / nyquist
    high = Config.HIGHCUT / nyquist
    b_band, a_band = butter(4, [low, high], btype='band')
    
    # Режекторный фильтр (50 Гц)
    notch = Config.NOTCH / nyquist
    b_notch, a_notch = butter(4, [notch-0.01, notch+0.01], btype='bandstop')
    
    return (b_band, a_band), (b_notch, a_notch)


def process_signal(signal, orig_sr, target_sr, filters):
    """
    Полная обработка одноканального сигнала.
    
    Шаги:
        1. Ресемплинг до целевой частоты (если нужно)
        2. Полосовая фильтрация
        3. Режекторная фильтрация (50 Гц)
        4. Z-нормализация (среднее=0, дисперсия=1)
    """
    # Ресемплинг
    if orig_sr != target_sr:
        duration = len(signal) / orig_sr
        new_len = int(duration * target_sr)
        signal = np.interp(
            np.linspace(0, len(signal)-1, new_len),
            np.arange(len(signal)),
            signal
        )
    
    # Фильтрация
    (b_band, a_band), (b_notch, a_notch) = filters
    signal = filtfilt(b_band, a_band, signal)
    signal = filtfilt(b_notch, a_notch, signal)
    
    # Нормализация
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    
    return signal


def compute_sample_weight(A_win, B_win, C_win, pattern_weight):
    """
    ВЫЧИСЛЕНИЕ ВЕСА ОКНА НА ОСНОВЕ ТРЕХ ФАКТОРОВ
    
    ============================================================================
    НАЗНАЧЕНИЕ
    ============================================================================
    Функция рассчитывает итоговый вес для каждого временного окна ЭЭГ сигнала.
    Вес определяет, насколько сильно это окно повлияет на обучение модели.
    
    Чем выше вес, тем больше модель "учится" на этом окне.
    Чем ниже вес, тем меньше влияние окна (спорные или ненадежные данные).
    
    
    ============================================================================
    ВХОДНЫЕ ПАРАМЕТРЫ
    ============================================================================
    
    A_win : np.ndarray
        Оценки эксперта A для всех временных точек в окне (массив 0/1)
        Длина массива = WINDOW_SIZE (1024 точки при 256 Гц и 4 сек)
    
    B_win : np.ndarray
        Оценки эксперта B для всех временных точек в окне
    
    C_win : np.ndarray
        Оценки эксперта C для всех временных точек в окне
    
    pattern_weight : float
        Базовый вес паттерна из R-анализа (0.2 ... 1.0)
        Определяется по середине окна:
        - '000' (фон) → 0.2
        - '001', '010', '100' (один эксперт) → 0.3-0.5
        - '011', '101', '110' (два эксперта) → 0.85-0.9
        - '111' (полное согласие) → 1.0
    
    
    ============================================================================
    ВЫХОДНЫЕ ПАРАМЕТРЫ
    ============================================================================
    
    sample_weight : float
        Итоговый вес окна (0.0 ... 1.0)
        Используется в функции потерь для взвешивания вклада окна
    
    agreement : float
        Локальное согласие экспертов (0.0 ... 1.0)
        Показывает, насколько схожи средние оценки экспертов в окне
    
    confidence : float
        Уверенность экспертов (0.0 ... 1.0)
        Показывает, насколько эксперты единодушны в каждой временной точке
    
    
    ============================================================================
    АЛГОРИТМ РАСЧЕТА
    ============================================================================
    
    ШАГ 1: РАСЧЕТ ЛОКАЛЬНОГО СОГЛАСИЯ (AGREEMENT)
    -----------------------------------------------
    Идея: оценить, насколько схожи эксперты в среднем по всему окну.
    
    mean_A = среднее значение оценок эксперта A в окне
    mean_B = среднее значение оценок эксперта B в окне
    mean_C = среднее значение оценок эксперта C в окне
    
    Пример:
        Если mean_A = 0.1, mean_B = 0.1, mean_C = 0.1 → эксперты согласны
        Если mean_A = 0.0, mean_B = 0.5, mean_C = 1.0 → эксперты не согласны
    
    std_means = стандартное отклонение [mean_A, mean_B, mean_C]
    
    agreement = 1 - std_means
    
    Логика:
        - std_means = 0   (полное согласие) → agreement = 1.0
        - std_means = 0.5 (среднее расхождение) → agreement = 0.5
        - std_means = 1.0 (полное расхождение) → agreement = 0.0
    
    Затем agreement ограничивается диапазоном [0, 1]
    
    
    ШАГ 2: РАСЧЕТ УВЕРЕННОСТИ (CONFIDENCE)
    ---------------------------------------
    Идея: оценить, насколько эксперты согласны в КАЖДОЙ временной точке.
    
    expert_matrix = [A_win, B_win, C_win]  # матрица 3 x WINDOW_SIZE
    disagreement = среднее по времени от std(по экспертам)
    
    Пример для одной временной точки:
        Оценки [0,0,0] → std = 0.00 → disagreement малый
        Оценки [0,0,1] → std = 0.47 → disagreement средний
        Оценки [0,1,0] → std = 0.47 → disagreement средний
        Оценки [0,1,1] → std = 0.47 → disagreement средний
        Оценки [1,0,0] → std = 0.47 → disagreement средний
        Оценки [1,1,0] → std = 0.47 → disagreement средний
        Оценки [1,0,1] → std = 0.47 → disagreement средний
        Оценки [1,1,1] → std = 0.00 → disagreement малый
    
    Максимальное возможное std для бинарных значений = 0.5
    (когда один эксперт говорит 1, другой 0, третий 0 или 1)
    
    confidence = 1 - (disagreement / 0.5)
    
    Логика:
        - disagreement = 0.0 (полное согласие) → confidence = 1.0
        - disagreement = 0.25 (среднее) → confidence = 0.5
        - disagreement = 0.5 (максимальное) → confidence = 0.0
    
    Затем confidence ограничивается диапазоном [0, 1]
    
    
    ШАГ 3: РАСЧЕТ ФИНАЛЬНОГО ВЕСА
    -------------------------------
    Итоговый вес = pattern_weight × factor_agreement × factor_confidence
    
    factor_agreement = 0.3 + 0.7 × agreement
    factor_confidence = 0.5 + 0.5 × confidence
    
    Зачем нужны коэффициенты 0.3 и 0.5? (сдвиги)
    
    БЕЗ сдвига (agreement напрямую):
        Если agreement = 0 → вес обнулится → модель не учится на спорных окнах
        Но даже спорные окна содержат полезную информацию!
    
    СО сдвигом 0.3:
        agreement = 0 → factor = 0.3 (30% веса сохраняется)
        agreement = 1 → factor = 1.0 (100% веса)
        agreement = 0.5 → factor = 0.65 (65% веса)
    
    СО сдвигом 0.5 (для confidence):
        confidence = 0 → factor = 0.5 (50% веса сохраняется)
        confidence = 1 → factor = 1.0 (100% веса)
        confidence = 0.5 → factor = 0.75 (75% веса)
    
    Это гарантирует, что модель всегда получает сигнал, даже из спорных данных,
    но с соответствующим пониженным весом.
    
    
    ============================================================================
    ПРИМЕРЫ РАСЧЕТА
    ============================================================================
    
    ПРИМЕР 1: Идеальное окно (четкий приступ)
    -----------------------------------------
    A_win = [1,1,1,1,1,1,1,1,1,1]  (все 1)
    B_win = [1,1,1,1,1,1,1,1,1,1]  (все 1)
    C_win = [1,1,1,1,1,1,1,1,1,1]  (все 1)
    pattern_weight = 1.0  (паттерн 111)
    
    Шаг 1: mean_A = 1.0, mean_B = 1.0, mean_C = 1.0
           std_means = 0.0 → agreement = 1.0
    
    Шаг 2: disagreement = 0.0 (все точки одинаковы)
           confidence = 1 - (0.0/0.5) = 1.0
    
    Шаг 3: factor_agreement = 0.3 + 0.7×1.0 = 1.0
           factor_confidence = 0.5 + 0.5×1.0 = 1.0
           sample_weight = 1.0 × 1.0 × 1.0 = 1.0
    
    Интерпретация: модель учится в полную силу
    
    
    ПРИМЕР 2: Спорное окно (только B видит приступ)
    -----------------------------------------------
    A_win = [0,0,0,0,0,0,0,0,0,0]  (все 0)
    B_win = [1,1,1,1,1,1,1,1,1,1]  (все 1)
    C_win = [0,0,0,0,0,0,0,0,0,0]  (все 0)
    pattern_weight = 0.5  (паттерн 010)
    
    Шаг 1: mean_A = 0.0, mean_B = 1.0, mean_C = 0.0
           std_means = 0.47 → agreement = 0.53
    
    Шаг 2: disagreement = 0.47 (разные оценки в каждой точке)
           confidence = 1 - (0.47/0.5) = 0.06
    
    Шаг 3: factor_agreement = 0.3 + 0.7×0.53 = 0.67
           factor_confidence = 0.5 + 0.5×0.06 = 0.53
           sample_weight = 0.5 × 0.67 × 0.53 = 0.18
    
    Интерпретация: модель учится с весом ~18% от полного
    
    
    ПРИМЕР 3: Неуверенные эксперты (плавающие оценки)
    -----------------------------------------------
    A_win = [0,0,1,1,0,0,1,1,0,0]  (50% единиц)
    B_win = [0,1,0,1,0,1,0,1,0,1]  (50% единиц)
    C_win = [0,0,0,1,1,1,0,0,0,1]  (40% единиц)
    pattern_weight = 0.85  (паттерн 011 или 101 или 110)
    
    Шаг 1: mean_A = 0.5, mean_B = 0.5, mean_C = 0.4
           std_means = 0.05 → agreement = 0.95
    
    Шаг 2: disagreement = 0.32 (средний разброс)
           confidence = 1 - (0.32/0.5) = 0.36
    
    Шаг 3: factor_agreement = 0.3 + 0.7×0.95 = 0.97
           factor_confidence = 0.5 + 0.5×0.36 = 0.68
           sample_weight = 0.85 × 0.97 × 0.68 = 0.56
    
    Интерпретация: модель учится с весом ~56% от полного
    
    
    ============================================================================
    ЗНАЧЕНИЯ ВЕСОВ И ИХ ИНТЕРПРЕТАЦИЯ
    ============================================================================
    
    Вес > 0.8  → Высокая достоверность (полное согласие, уверенные эксперты)
    Вес 0.5-0.8 → Средняя достоверность (частичное согласие)
    Вес 0.2-0.5 → Низкая достоверность (спорные случаи, один эксперт)
    Вес < 0.2  → Очень низкая достоверность (фон или сильные разногласия)
    
    
    ============================================================================
    ПОЧЕМУ ЭТА ФОРМУЛА ЛУЧШЕ
    ============================================================================
    
    1. УЧЕТ ДВУХ УРОВНЕЙ НЕОПРЕДЕЛЕННОСТИ:
       - agreement: глобальное согласие по всему окну
       - confidence: локальная уверенность в каждой точке
    
    2. НИКОГДА НЕ ОБНУЛЯЕТ ВЕСА:
       - Даже при полном разногласии вес > 0
       - Модель всегда получает сигнал для обучения
    
    3. БАЗИРУЕТСЯ НА РЕАЛЬНЫХ ДАННЫХ:
       - pattern_weight из статистики R-анализа
       - Не использует kappa пациента (была слишком грубой)
    
    4. ЛИНЕЙНОСТЬ И ПРОЗРАЧНОСТЬ:
       - Легко интерпретировать
       - Легко модифицировать при необходимости
    
    
    ============================================================================
    ИСПОЛЬЗОВАНИЕ В ОБУЧЕНИИ
    ============================================================================
    
    В функции потерь weighted_bce:
        loss = (BCE(pred, target) * sample_weight).sum() / sample_weight.sum()
    
    Это означает:
        - Окна с высоким весом сильнее влияют на градиент
        - Окна с низким весом почти не влияют на обучение
        - Модель фокусируется на надежных данных, но не игнорирует спорные
    
    ============================================================================
    """

     # 1. Локальное согласие (
    mean_A = np.mean(A_win)
    mean_B = np.mean(B_win)
    mean_C = np.mean(C_win)
    agreement = 1 - np.std([mean_A, mean_B, mean_C])
    agreement = np.clip(agreement, 0, 1)

    # 2. Уверенность экспертов (на основе поточечного std)
    expert_matrix = np.stack([A_win, B_win, C_win], axis=0)
    disagreement = np.std(expert_matrix, axis=0).mean()
    confidence = 1 - (disagreement / 0.5)
    confidence = np.clip(confidence, 0, 1)

    # 3. Финальный вес 
    sample_weight = (
        pattern_weight *
        (0.3 + 0.7 * agreement) *
        (0.5 + 0.5 * confidence)
    )
    
    return sample_weight, agreement, confidence


# ============================================================================
# 2. ОСНОВНОЙ КЛАСС ПРЕПРОЦЕССОРА
# ============================================================================

class EEGPreprocessor:
    """
    Главный класс, управляющий процессом предобработки.
    """
    
    def __init__(self):
        """Инициализация: создание фильтров, загрузка путей"""
        self.filters = create_filters(Config.TARGET_SR)
        
        # Создаем выходные папки
        Path(Config.OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
        Path(Config.VISUALIZATION_PATH).mkdir(parents=True, exist_ok=True)
        Path(Config.MODEL_PATH).mkdir(parents=True, exist_ok=True)

        # Фиксируем seed для воспроизводимости
        Config.set_seed()
        
        print(f"\n{'='*60}")
        print("ПРЕПРОЦЕССОР EEGNeX")
        print(f"{'='*60}")
        print(f"SEED: {Config.SEED}")
        print(f"Целевая частота: {Config.TARGET_SR} Гц")
        print(f"Размер окна: {Config.WINDOW_SIZE} точек ({Config.WINDOW_SEC} сек)")
        print(f"Окон на пациента: {Config.N_WINDOWS}")
        print(f"Каналов: {Config.N_CHANNELS}")
        print(f"{'='*60}\n")
    
    def load_sample(self, sample_name):
        """
        Загружает выборку (train/val/test) из CSV файла.
        
        CSV файл содержит колонки:
            - record: номер эпохи
            - signal: номер пациента
            - pattern: паттерн с префиксом 'P' (например, 'P010')
            - expert_A, expert_B, expert_C: оценки экспертов
            - fleiss_kappa: каппа для этого сигнала
            - pattern_weight: вес паттерна из R-анализа
        """
        file_path = Path(Config.R_DATA_PATH) / f"{sample_name}_sample.csv"
        df = pd.read_csv(file_path, sep=';', encoding='windows-1251')
        
        # Убираем префикс 'P' из паттернов (P010 → 010)
        df['pattern'] = df['pattern'].str[1:]
        
        print(f"Загружен {sample_name}: {len(df)} строк, {df['signal'].nunique()} пациентов")
        return df
    
    def process_patient(self, patient_id, patient_data):
        """
        Обрабатывает одного пациента.
        
        Аргументы:
            patient_id: номер пациента (1-79)
            patient_data: DataFrame с эпохами для этого пациента из сплита
        
        Возвращает:
            Словарь с данными пациента (X, y_prob, y_hard, weights, metadata)
        """

        edf_file = Path(Config.EDF_PATH) / f"eeg{patient_id}.edf"
        if not edf_file.exists():
            print(f"  ⚠️ Файл не найден: {edf_file}")
            return None
        
        # ========== 1. ЗАГРУЗКА EDF ==========
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
        orig_sr = int(raw.info['sfreq'])
        ch_names = [ch.upper() for ch in raw.ch_names]
        
        # ========== 2. ИЗВЛЕЧЕНИЕ КАНАЛОВ ==========
        signals = []
        
        for ch_idx, target_ch in enumerate(Config.CHANNELS):
            # Поиск канала в EDF (по подстроке)
            idx = None
            for i, name in enumerate(ch_names):
                if target_ch.upper() in name:
                    idx = i
                    break
            
            if idx is None:
                print(f"  ⚠️ Канал {target_ch} не найден для пациента {patient_id}")
                return None
            
            # Извлечение и обработка сигнала
            data, _ = raw[idx, :]
            sig = data[0].astype(np.float64)
            sig = process_signal(sig, orig_sr, Config.TARGET_SR, self.filters)
            signals.append(sig)
        
        signals = np.stack(signals, axis=0)  # [channels, time]
        
        # ========== 3. ПОСТРОЕНИЕ ЭКСПЕРТНЫХ РЯДОВ ==========
        n_samples = signals.shape[1]
        expert_A = np.zeros(n_samples)
        expert_B = np.zeros(n_samples)
        expert_C = np.zeros(n_samples)
        
        samples_per_epoch = Config.TARGET_SR * Config.EPOCH_SEC  # 512 точек
        
        for _, row in patient_data.iterrows():
            epoch = row['record']
            start = (epoch - 1) * samples_per_epoch
            end = start + samples_per_epoch
            
            if end > n_samples:
                continue
            
            expert_A[start:end] = row['expert_A']
            expert_B[start:end] = row['expert_B']
            expert_C[start:end] = row['expert_C']
        
        # ========== 4. НАРЕЗКА ОКОН ==========
        window_size = Config.WINDOW_SIZE
        stride = Config.STRIDE
        
        windows = []
        y_prob_list = []
        y_hard_list = []
        weights_list = []
        metadata_list = []
        
        for start in range(0, n_samples - window_size, stride):
            end = start + window_size
            
            A_win = expert_A[start:end]
            B_win = expert_B[start:end]
            C_win = expert_C[start:end]
            
            # Пропускаем окна с NaN (нет данных от экспертов)
            if np.any(np.isnan(A_win)) or np.any(np.isnan(B_win)) or np.any(np.isnan(C_win)):
                continue
            
            # Soft label (вероятность приступа)
            y_prob = (np.mean(A_win) + np.mean(B_win) + np.mean(C_win)) / 3
            y_prob_list.append(y_prob)
            
            # Hard label (бинарная метка)
            y_hard_list.append(1 if y_prob >= 0.5 else 0)
            
            # Confidence (уверенность экспертов)
            confidence = compute_confidence(A_win, B_win, C_win)
            
            # Pattern для середины окна
            mid = start + window_size // 2
            pattern_at_mid = f"{int(expert_A[mid])}{int(expert_B[mid])}{int(expert_C[mid])}"
            pattern_weight = Config.PATTERN_WEIGHTS.get(pattern_at_mid, 0.5)
            
            # Расчет веса
            sample_weight, agreement, confidence = compute_sample_weight(
                A_win, B_win, C_win, pattern_weight
            )
            weights_list.append(sample_weight)
            
            # Сохраняем метаданные
            metadata_list.append({
                'patient': int(patient_id),
                'window_start': int(start),
                'window_end': int(end),
                'pattern': pattern_at_mid,
                'agreement': float(agreement),
                'confidence': float(confidence),
                'y_prob': float(y_prob),
                'y_hard': 1 if y_prob >= 0.5 else 0,
                'pattern_weight': float(pattern_weight)
            })
            
            # Окно сигнала
            windows.append(signals[:, start:end])
        
        if not windows:
            return None
        
        return {
            'X': np.stack(windows, axis=0).astype(np.float32),
            'y_prob': np.array(y_prob_list).astype(np.float32),
            'y_hard': np.array(y_hard_list).astype(np.int64),
            'weights': np.array(weights_list).astype(np.float32),
            'metadata': metadata_list  
        }
    
    def process_split(self, sample_name):
        """
        Обрабатывает выборку (train/val/test).
        """
        # Загружаем данные сплита
        split_df = self.load_split(sample_name)
        
        all_X = []
        all_y_prob = []
        all_y_hard = []
        all_weights = []
        all_metadata = []
        
        patients = split_df['signal'].unique()
        
        for patient_id in tqdm(patients, desc=f"Processing {sample_name}"):
            patient_data = split_df[split_df['signal'] == patient_id].sort_values('record')
            result = self.process_patient(patient_id, patient_data)
            
            if result:
                all_X.append(result['X'])
                all_y_prob.append(result['y_prob'])
                all_y_hard.append(result['y_hard'])
                all_weights.append(result['weights'])
                all_metadata.extend(result['metadata'])
        
        if not all_X:
            print(f"❌ Нет данных для {sample_name}")
            return
        
        # Объединяем все окна
        X = np.concatenate(all_X, axis=0)
        y_prob = np.concatenate(all_y_prob, axis=0)
        y_hard = np.concatenate(all_y_hard, axis=0)
        weights = np.concatenate(all_weights, axis=0)
        
        # Сохраняем
        output_file = Path(Config.OUTPUT_PATH) / f"helsinki_{sample_name}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump({
                'X': X,
                'y_prob': y_prob,
                'y_hard': y_hard,
                'sample_weights': weights,
                'metadata': all_metadata,
                'config': {
                    'n_channels': Config.N_CHANNELS,
                    'window_size': Config.WINDOW_SIZE,
                    'target_sr': Config.TARGET_SR,
                    'window_sec': Config.WINDOW_SEC,
                    'seed': Config.SEED
                }
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"\n✅ {sample_name}: {len(X)} окон, баланс 0/1: {np.bincount(y_hard)}")
        print(f"   Метаданных сохранено: {len(all_metadata)}")
    
    def run(self):
        """Запуск обработки всех трех выборок."""
        for split_name in ['train', 'val', 'test']:
            self.process_split(split_name)


# ============================================================================
# 3. ТОЧКА ВХОДА
# ============================================================================

if __name__ == "__main__":
    preprocessor = EEGPreprocessor()
    preprocessor.run()
    
    print("\n" + "="*60)
    print("ПРЕПРОЦЕССИНГ ЗАВЕРШЕН УСПЕШНО!")
    print("="*60)