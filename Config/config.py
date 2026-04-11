"""
config.py - Конфигурация проекта

НАЗНАЧЕНИЕ: Централизованное хранение всех параметров для preprocessing, обучения и оценки.
"""

import torch
import random
import numpy as np
import os

class Config:
    """Главный класс конфигурации"""
    
    # ================================================================
    # ПУТИ К ФАЙЛАМ 
    # ================================================================
    
    # Путь к папке с сырыми EDF файлами
    EDF_PATH = "/media/avengus/Локальный диск/Dev/EEG/helsinki/"
    
    # Путь для сохранения обработанных .pkl файлов
    OUTPUT_PATH = "/media/avengus/Локальный диск/Dev/EEG/helsinki/processed/"

    # Путь для сохранения тепловых карт визуализации
    VISUALIZATION_PATH = "/media/avengus/Локальный диск/Dev/EEG/helsinki/visual/"
    
    # Путь для сохранения моделей
    MODEL_PATH = "/media/avengus/Локальный диск/Dev/EEG/helsinki/models/"

    # ================================================================
    # ПАРАМЕТРЫ СИГНАЛА
    # ================================================================
    
    TARGET_SR = 256          # целевая частота дискретизации (Гц)
    EPOCH_SEC = 1            # длительность эпохи (сек) — из датасета
    WINDOW_SEC = 5           # длительность окна для модели (сек)
    WINDOW_SIZE = TARGET_SR * WINDOW_SEC  # 1280 точек
    STRIDE_SEC = 1           # количество окон для покрытия 30 секунд
    STRIDE = TARGET_SR * 1   # шаг между окнами (1 сек = 256 точек)
    
    # ================================================================
    # КАНАЛЫ (система 10-20)
    # ================================================================
    
    CHANNELS = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
    ]
    N_CHANNELS = len(CHANNELS)  # = 19
    
    # ================================================================
    # ПАРАМЕТРЫ ФИЛЬТРАЦИИ
    # ================================================================
    
    LOWCUT = 0.5       # нижняя граница полосового фильтра (Гц)
    HIGHCUT = 30.0     # верхняя граница (для неонатальных данных)
    NOTCH = 50.0       # частота сетевой наводки (Гц)
    NOTCH_WIDTH = 2.0  # ширина режекторного фильтра (±1 Гц)

    # ================================================================
    # ВЕСА ПОЗИЦИЙ (центр важнее краёв)
    # ================================================================
    
    POS_WEIGHTS = [0.1, 0.2, 0.4, 0.2, 0.1]

    # Проверка, что сумма весов = 1.0 (для корректного усреднения)
    assert abs(sum(POS_WEIGHTS) - 1.0) < 1e-6, "POS_WEIGHTS must sum to 1.0"

    # ================================================================
    # ВЕСА ДЛЯ СОГЛАСОВАННОСТИ ЭКСПЕРТОВ
    # ================================================================

    WEIGHT_ALL_AGREE = 1.0    # 3 эксперта согласны (000 или 111)
    WEIGHT_TWO_AGREE = 0.67   # 2 эксперта согласны (011, 101, 110)
    WEIGHT_ONE_AGREE = 0.33   # 1 эксперт (001, 010, 100)
    WEIGHT_BACKGROUND = 1.0   # фон (000) — важный negative класс

    # ================================================================
    # РАЗБИЕНИЕ ПАЦИЕНТОВ
    # ================================================================
    
    EXCLUDED_PATIENTS = [1, 25, 62]  # битые EDF
    
    TRAIN_PATIENTS = [
        3, 10, 18, 27, 28, 29, 30, 32, 35, 37, 42, 45, 48, 49, 53,
        5, 7, 9, 14, 19, 20, 22, 38, 39, 41, 44, 47, 50, 66, 67,
        2, 4, 6, 8, 11, 12, 13, 15, 17, 21, 23, 24, 26, 31, 33, 34, 
        36, 40, 43, 46, 51, 52, 56, 61, 65
    ]
    
    VAL_PATIENTS = [55, 57, 58, 69, 73, 75, 16, 54, 63, 64]
    
    TEST_PATIENTS = [59, 60, 70, 72, 78, 79, 68, 71, 74, 76, 77]
    
    # ================================================================
    # SEED CONTROL (Для воспроизводимости)
    # ================================================================
    
    SEED = 42
    
    @staticmethod
    def set_seed():
        """Фиксация всех генераторов случайных чисел"""
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        torch.cuda.manual_seed_all(Config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(Config.SEED)
    
    # ================================================================
    # ПАРАМЕТРЫ ОБУЧЕНИЯ
    # ================================================================
    
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10 # для early stopping
    LR_PATIENCE = 5      # patience для ReduceLROnPlateau
    LR_FACTOR = 0.5      # фактор уменьшения learning rate
    GRADIENT_CLIP_NORM = 1.0 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ================================================================
    # ПАРАМЕТРЫ EEGNeX
    # ================================================================
    
    EEGNEX_N_OUTPUTS = 1 # бинарная классификация
    EEGNEX_ACTIVATION = torch.nn.ELU
    EEGNEX_FILTER_1 = 8
    EEGNEX_FILTER_2 = 32
    EEGNEX_DROP_PROB = 0.5

    #================================================================
    # ПРОВЕРКА
    # ================================================================
    
    @classmethod
    def print_summary(cls):
        """Вывод сводки конфигурации."""
        print("=" * 60)
        print("КОНФИГУРАЦИЯ EEG-SeizureCatcher")
        print("=" * 60)
        print(f"Device: {cls.DEVICE}")
        print(f"Seed: {cls.SEED}")
        print(f"Target SR: {cls.TARGET_SR} Hz")
        print(f"Window: {cls.WINDOW_SEC} sec → {cls.WINDOW_SIZE} samples")
        print(f"Stride: {cls.STRIDE_SEC} sec")
        print(f"Channels: {cls.N_CHANNELS}")
        print(f"Filter: {cls.LOWCUT}-{cls.HIGHCUT} Hz, notch {cls.NOTCH} Hz")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Learning rate: {cls.LEARNING_RATE}")
        print(f"Train/Val/Test: {len(cls.TRAIN_PATIENTS)}/{len(cls.VAL_PATIENTS)}/{len(cls.TEST_PATIENTS)}")
        print("=" * 60)