"""
train.py - Обучение модели EEGNeX

НАЗНАЧЕНИЕ:
    Обучение модели EEGNeX на предобработанных данных с использованием
    weighted loss (учет весов из R-анализа и уверенности экспертов).

АЛГОРИТМ:
    1. Загрузка train и val датасетов из .pkl файлов
    2. Создание модели EEGNeX через braindecode
    3. Обучение с weighted BCE-loss (бинарная кросс-энтропия)
    4. Сохранение лучшей модели по val loss
    5. Ранняя остановка при отсутствии улучшений
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import numpy as np
from braindecode.models import EEGNeX
from pathlib import Path
from tqdm import tqdm
from config import Config

# ============================================================================
# 1. ДАТАСЕТ
# ============================================================================

class EEGDataset(torch.utils.data.Dataset):
    """
    Загрузчик данных из .pkl файлов, созданных preprocessing.py.
    
    Каждый .pkl файл содержит:
        'X': [N, 19, 1024] - сигналы
        'y_prob': [N] - soft labels (0..1)
        'y_hard': [N] - hard labels (0/1)
        'sample_weights': [N] - веса окон
    """
    
    def __init__(self, pkl_path, use_soft_labels=True):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.X = torch.FloatTensor(data['X'])
        self.y = torch.FloatTensor(data['y_prob']) if use_soft_labels else torch.LongTensor(data['y_hard'])
        self.weights = torch.FloatTensor(data['sample_weights'])
        self.metadata = data['metadata']  
        
        # Статистика по весам для мониторинга
        print(f"📁 Загружен {Path(pkl_path).name}: {len(self.X)} окон")
        print(f"   Веса: min={self.weights.min():.3f}, max={self.weights.max():.3f}, mean={self.weights.mean():.3f}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.weights[idx], self.metadata[idx]


# ============================================================================
# 2. ФУНКЦИЯ ПОТЕРЬ
# ============================================================================

def weighted_bce(pred, target, weight):
    """
    Weighted Binary Cross Entropy loss.
    
    Почему BCE, а не MSE?
        - Работает с вероятностями (0..1)
        - Устойчивее к дисбалансу классов
        - Лучше для оптимизации AUC
    
    Аргументы:
        pred: logits от модели 
        target: soft labels (0..1)
        weight: веса для каждого примера
    """
    loss = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
    return (loss * weight).sum() / (weight.sum() + 1e-8)


# ============================================================================
# 3. ОБУЧЕНИЕ
# ============================================================================

def train():
    """Главная функция обучения"""
    
    # Фиксируем seed
    Config.set_seed()

    print(f"\n{'='*60}")
    print("ОБУЧЕНИЕ EEGNeX")
    print(f"{'='*60}")
    print(f"SEED: {Config.SEED}")
    print(f"Устройство: {Config.DEVICE}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Learning rate: {Config.LEARNING_RATE}")
    print(f"Gradient clip norm: {Config.GRADIENT_CLIP_NORM}")
    print(f"Эпох: {Config.EPOCHS}")
    print(f"{'='*60}\n")
    
    # ========== 1. ЗАГРУЗКА ДАННЫХ ==========
    train_dataset = EEGDataset(Path(Config.OUTPUT_PATH) / "helsinki_train.pkl", use_soft_labels=True)
    val_dataset = EEGDataset(Path(Config.OUTPUT_PATH) / "helsinki_val.pkl", use_soft_labels=True)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    
    print(f"\n Train: {len(train_dataset)} окон, {len(train_loader)} батчей")
    print(f" Val: {len(val_dataset)} окон, {len(val_loader)} батчей")
    
    # ========== 2. МОДЕЛЬ ==========
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
    ).to(Config.DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Модель EEGNeX: {n_params:,} параметров")
    
    # ========== 3. ОПТИМИЗАТОР ==========
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # ========== 4. ЦИКЛ ОБУЧЕНИЯ ==========
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\n" + "="*60)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("="*60)
    
    for epoch in range(Config.EPOCHS):
        # --- ОБУЧЕНИЕ ---
        model.train()
        train_loss = 0
        
        for X, y, w in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}"):
            X, y, w = X.to(Config.DEVICE), y.to(Config.DEVICE), w.to(Config.DEVICE)
            
            optimizer.zero_grad()
            pred = model(X).squeeze(-1)
            loss = weighted_bce(pred, y, w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP_NORM)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # --- ВАЛИДАЦИЯ ---
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for X, y, w in val_loader:
                X, y, w = X.to(Config.DEVICE), y.to(Config.DEVICE), w.to(Config.DEVICE)
                pred = model(X).squeeze(-1)
                loss = weighted_bce(pred, y, w)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # --- ЛОГИРОВАНИЕ ---
        print(f"Epoch {epoch+1:3d}/{Config.EPOCHS} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")
        
        # --- SCHEDULER ---
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < Config.LEARNING_RATE * 0.01:
            print(f"  LR decayed to {current_lr:.6f}")
        
        # --- СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = Path(Config.MODEL_PATH) / "best_eegnex_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'seed': Config.SEED
            }, model_path)
            patience_counter = 0
            print(f"   Сохранена лучшая модель (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
        
        # --- EARLY STOPPING ---
        if patience_counter >= Config.PATIENCE:
            print(f"\n !!! Ранняя остановка на эпохе !!! {epoch+1}")
            break
    
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"Лучший val_loss: {best_val_loss:.4f}")
    print(f"Модель сохранена: {Config.MODEL_PATH}/best_eegnex_model.pth")
    print("="*60)
    
    return model


# ============================================================================
# 4. ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    train()