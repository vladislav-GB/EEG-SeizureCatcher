"""
config.py - Конфигурация проекта EEG-SeizureCatcher 

НАЗНАЧЕНИЕ:
    Централизованное хранение всех параметров для preprocessing, обучения и оценки.
    Все настройки в одном месте — легко менять и отслеживать эксперименты.

СТРАТЕГИЯ:
    - Окна 3 секунды (768 отсчётов)
    - Target окна = центральная секунда
    - Soft targets: 0.0 / 0.6 / 0.9 / 1.0
    - AdaptiveFocalLoss с усилением для приступов
    - Scheduler по val_pr_auc
    - Сохранение моделей с PR-AUC >= 0.30
    - Формат данных: .npz compressed
"""

import torch
import random
import numpy as np
import os
from pathlib import Path


class Config:

    """Главный класс конфигурации"""  
    # ================================================================
    # ПУТИ К ФАЙЛАМ
    # ================================================================
    
    PROJECT_ROOT = Path(__file__).parent.parent

    # Путь к папке с сырыми EDF файлами и CSV аннотациями
    DATA_PATH = "/media/avengus/Локальный диск/Dev/EEG/helsinki/"
    
    # Путь для сохранения обработанных .pkl файлов (после preprocessing)
    # OUTPUT_PATH = "/media/avengus/Локальный диск/Dev/EEG/helsinki/processed/"
    OUTPUT_PATH = PROJECT_ROOT / "processed"

    # Путь для сохранения визуализаций (Grad-CAM, топокарты)
    # VISUALIZATION_PATH = "/media/avengus/Локальный диск/Dev/EEG/helsinki/visual/"
    VISUALIZATION_PATH = PROJECT_ROOT / "picture"

    # Путь для сохранения обученных моделей
    #MODEL_PATH = "/media/avengus/Локальный диск/Dev/EEG/helsinki/models/"
    MODEL_PATH = PROJECT_ROOT / "models"
    
    # Путь к R-данным (статистика пациентов)
    R_DATA_PATH = "/media/avengus/Локальный диск/Dev/EEG/R_results/annotations_wide"
    
    # Логи 
    LOG_PREPROCESSING_PATH = PROJECT_ROOT / "log-preprocessing"
    LOG_TRAIN_PATH = PROJECT_ROOT / "log-train"
    LOG_TEST_PATH = PROJECT_ROOT / "log-test"
    
    # ================================================================
    # ПАРАМЕТРЫ СИГНАЛА
    # ================================================================
    
    TARGET_SR = 256          # Целевая частота дискретизации (Гц)
    WINDOW_SEC = 3           # Длина окна для модели (сек)
    WINDOW_SIZE = TARGET_SR * WINDOW_SEC  # 768 отсчётов
    STRIDE_SEC = 1           # Шаг между окнами (сек), перекрытие 2 сек
    STRIDE = TARGET_SR * 1   # 256 отсчётов
    
    # ================================================================
    # КАНАЛЫ (система 10-20)
    # ================================================================
    
    CHANNELS = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
    ]
    N_CHANNELS = len(CHANNELS)  # 19 каналов
    
    # ================================================================
    # ПАРАМЕТРЫ ФИЛЬТРАЦИИ
    # ================================================================
    
    LOWCUT = 0.5       # Нижняя граница полосового фильтра (Гц)
    HIGHCUT = 30.0     # Верхняя граница (Гц), неонатальные приступы <15 Гц
    NOTCH = 50.0       # Частота сетевой наводки (Гц)
    
    # ================================================================
    # РАЗБИЕНИЕ ПАЦИЕНТОВ 
    # ================================================================
    
    EXCLUDED_PATIENTS = [1, 25, 62]  # не участвующие EDF
    

    # TEST: 10 пациентов
    TEST_PATIENTS = [
        60, 59, # 000: 2 пациента 
        5, 41, 14, 34, # 111: 4 пациента
        64, 74, # 2exp: 2 пациента
        56, 54 #  1exp: 2 пациент 
    ]
    
    # VAL: 10 пациентов 
    VAL_PATIENTS = [
        72, 55, # 000: 2 пациента
        39, 71, 75, 76, # 111: 4 пациента
        68, 23, # 2exp: 2 пациента
        26, 2 # 1exp: 2 пациент
    ]
    
    # TRAIN:  55 пациентов
    TRAIN_PATIENTS = [
        3, 10, 18, 27, 28, 29, 30, 32, 35, 37, 42, 45, 48, 49, 53, 58, 70, # 000: 17 пациентов

        4, 7, 9, 11, 13, 15, 16, 17, 19, 20, 21, 22, 31, 36, 38, 40, 44, 
        47, 50, 51, 52, 63, 66, 67, 69, 73, 77, 78, 79, # 111: 29 пациентов

        8, 33, # 2exp: 2 пациента

        6, 12, 24, 43, 46, 61, 65 # 1exp: 7 пациентов
    ]
    
    # ================================================================
    # SEED CONTROL (ДЛЯ ВОСПРОИЗВОДИМОСТИ)
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
    
    BATCH_SIZE = 64           # Размер батча 
    EPOCHS = 80               # Максимальное число эпох (early stopping остановит раньше)
    
    LEARNING_RATE = 1e-4      # Скорость обучения (0.001 — стандарт для Adam)
    WEIGHT_DECAY = 1e-4       # L2-регуляризация (0.0001 — мягкая)
    
    PATIENCE = 15             # Early stopping: ждём 15 эпох без улучшения PR-AUC
    LR_PATIENCE = 7           # ReduceLROnPlateau: ждём 5 эпох без улучшения loss
    LR_FACTOR = 0.5           # Фактор уменьшения learning rate (×0.5)
    
    GRADIENT_CLIP_NORM = 1.0  # Обрезка градиентов, защита от "взрыва"
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    SAVE_PR_THRESHOLD = 0.55
    SAVE_SEP_THRESHOLD = 0.20 

    # ================================================================
    # FOCAL LOSS 
    # ================================================================

    # Gamma — степень фокусировки на сложных примерах
    FOCAL_GAMMA = 1.6

    # Alpha — веса для позитивного и негативного классов
    FOCAL_ALPHA_POS = 0.8   # Вес для target > 0.1 (приступы)
    FOCAL_ALPHA_NEG = 0.2  # Вес для target = 0.0 (чистый фон)
    
    # ================================================================
    # ПАРАМЕТРЫ АРХИТЕКТУРЫ EEGNeX
    # ================================================================
    
    EEGNEX_N_OUTPUTS = 1              # Бинарная классификация (1 выход)
    EEGNEX_ACTIVATION = torch.nn.ELU  # ELU лучше ReLU для ЭЭГ
    
    EEGNEX_FILTER_1 = 8            # F1 — число темпоральных фильтров
    EEGNEX_FILTER_2 = 32           # F2 — число фильтров после расширения (4×F1)
    
    EEGNEX_DROP_PROB = 0.5         # Dropout для борьбы с переобучением
    
    # ================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ================================================================
    
    @staticmethod
    def setup_dirs():
        """Создание всех необходимых директорий"""
        for path in [Config.OUTPUT_PATH, Config.MODEL_PATH, Config.VISUALIZATION_PATH]:
            Path(path).mkdir(parents=True, exist_ok=True)
        print("✅ Все директории созданы")
    
    @classmethod
    def print_summary(cls):
        """Вывод сводки конфигурации для логирования"""
        print("=" * 60)
        print("КОНФИГУРАЦИЯ EEG-SeizureCatcher v5")
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


# ============================================================================
# АВТОМАТИЧЕСКАЯ ПРОВЕРКА ПРИ ИМПОРТЕ
# ============================================================================
if __name__ == "__main__":
    Config.setup_dirs()
    Config.set_seed()
    Config.print_summary()