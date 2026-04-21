"""
evaluate.py - Тестирование модели и создание аннотаций 

НАЗНАЧЕНИЕ:
    1. Финальная оценка обученной модели EEGNeX на тестовой выборке
    2. Создание wide-таблицы с метками модели для тестовых пациентов
    3. Анализ ошибок по пациентам и паттернам
    4. Сохранение результатов для R-анализа и LaTeX-отчёта

СТРАТЕГИЯ:
    - Метрики для всех данных и отдельно для high-agreement (target=0 или 1)
    - Восстановление посекундных предсказаний (max pooling для чувствительности)
    - Анализ ошибок: сравнение весов ошибочных и правильных окон
    - Сравнение с валидационными метриками (переобучение/обобщение)
    - Автоматическая интерпретация PR-AUC

МЕТРИКИ:

    Основные:
    - ROC-AUC: способность отличать приступ от фона
    - PR-AUC: главная метрика для дисбаланса (цель >0.35)
    
    Пороговые (0.3, 0.5, 0.7):
    - F1-score, Sensitivity, Specificity, Confusion Matrix
    
    Аналитические:
    - Распределение предсказаний vs истинных меток
    - Ошибки по пациентам
    - Сравнение весов ошибок vs правильных предсказаний

ВХОДНЫЕ ФАЙЛЫ:
    - processed/test_patients/patient_*.pkl
    - models/best_model.pth (или указанная модель)
    - data/annotations_wide.csv (для создания расширенной таблицы)

ВЫХОДНЫЕ ФАЙЛЫ:
    - evaluation_log_*.txt — полный лог с метриками
    - annotations_wide_with_model_test.csv — таблица с метками модели
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    f1_score,
    confusion_matrix
)
from torch.utils.data import DataLoader, Dataset
from braindecode.models import EEGNeX
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from Config.config import Config

# ============================================================================
# ЛОГИРОВАНИЕ
# ============================================================================

log_dir = Config.LOG_TEST_PATH
log_dir.mkdir(parents=True, exist_ok=True)

log_filename = log_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_file = open(log_filename, 'w', encoding='utf-8')

def log(msg, indent=False):
    timestamp = datetime.now().strftime('%H:%M:%S')
    line = f"[{timestamp}]    {msg}" if indent else f"[{timestamp}] {msg}"
    print(line)
    log_file.write(line + '\n')
    log_file.flush()


# ============================================================================
# ДАТАСЕТ
# ============================================================================

class TestEEGDataset(Dataset):
    def __init__(self, split_dir):
        self.split_dir = Path(split_dir)
        self.patient_files = sorted(self.split_dir.glob("*.npz"))
        
        log(f"---Загрузка {len(self.patient_files)} пациентов---")
        
        all_X, all_targets = [], []
        self.patient_ids = []
        
        for npz_file in tqdm(self.patient_files, desc="Loading"):
            data = np.load(npz_file)
            X = data['X']
            y = data['y']
            
            all_X.append(X)
            all_targets.append(y)
            
            patient_id = int(npz_file.stem.replace('patient_', ''))
            n_windows = len(y)
            self.patient_ids.extend([patient_id] * n_windows)
        
        self.X = np.concatenate(all_X, axis=0)
        self.targets = np.concatenate(all_targets, axis=0)
        
        log(f"Загружено {len(self)} окон")
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.X[idx]),
            torch.FloatTensor([self.targets[idx]]),
            self.patient_ids[idx]
        )


# ============================================================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ============================================================================

def load_models(model_paths):
    models = []
    log("")
    log("="*60)
    log(f"ЗАГРУЗКА {'АНСАМБЛЯ' if len(model_paths) > 1 else 'МОДЕЛИ'}")
    log("="*60)
    
    for i, path in enumerate(model_paths, 1):
        log(f"{i}. {Path(path).name}")
        
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
        
        checkpoint = torch.load(path, map_location=Config.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
        
        log(f"Эпоха {checkpoint.get('epoch', '?')+1}, val_pr_auc={checkpoint.get('val_pr_auc', 0):.4f}", indent=True)
    
    return models


def predict(models, X):
    if len(models) == 1:
        with torch.no_grad():
            return torch.sigmoid(models[0](X)).cpu().numpy().flatten()
    else:
        preds = []
        with torch.no_grad():
            for model in models:
                pred = torch.sigmoid(model(X)).cpu().numpy().flatten()
                preds.append(pred)
        return np.mean(preds, axis=0)


# ============================================================================
# МЕТРИКИ
# ============================================================================

def compute_all_metrics(preds, targets):
    targets_bin = (targets >= 0.5).astype(int)
    
    auc = roc_auc_score(targets_bin, preds)
    pr_auc = average_precision_score(targets_bin, preds)
    
    results = {'auc': auc, 'pr_auc': pr_auc}
    
    for threshold in [0.3, 0.5, 0.7]:
        preds_bin = (preds >= threshold).astype(int)
        cm = confusion_matrix(targets_bin, preds_bin)
        tn, fp, fn, tp = cm.ravel()
        
        results[f'th_{threshold}'] = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'f1': f1_score(targets_bin, preds_bin, zero_division=0),
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }
    
    return results


def print_metrics(results, preds, targets):
    log("")
    log("="*60)
    log("РЕЗУЛЬТАТЫ")
    log("="*60)
    log(f"ROC-AUC: {results['auc']:.4f}")
    log(f"PR-AUC:  {results['pr_auc']:.4f}")
    log("")
    log("МЕТРИКИ ПО ПОРОГАМ:")
    log(f"{'Порог':<8} {'Acc':<8} {'F1':<8} {'Recall':<8} {'Spec':<8} {'Prec':<8}")
    
    for th in [0.3, 0.5, 0.7]:
        r = results[f'th_{th}']
        log(f"{th:<8.1f} {r['accuracy']*100:<8.2f} {r['f1']*100:<8.2f} {r['sensitivity']*100:<8.2f} {r['specificity']*100:<8.2f} {r['precision']*100:<8.2f}")
    
    log("")
    log("СТАТИСТИКА ПРЕДСКАЗАНИЙ:")
    log(f"Mean pred: {preds.mean():.4f} ± {preds.std():.4f}")
    log(f" % >0.3: {(preds > 0.3).mean()*100:.1f}%")
    log(f" % >0.5: {(preds > 0.5).mean()*100:.1f}%")

# ============================================================================
# РАСШИРЕННЫЕ МЕТРИКИ 
# ============================================================================

def compute_extended_metrics(preds, targets, threshold=0.5):
    """Расширенные метрики: чувствительность, специфичность, ПЦПР, ПЦОР."""
    targets_bin = (targets >= 0.5).astype(int)
    preds_bin = (preds >= threshold).astype(int)
    
    cm = confusion_matrix(targets_bin, preds_bin)
    tn, fp, fn, tp = cm.ravel()
    
    # Основные метрики
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Se, Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Sp
    
    # Прогностические ценности
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # ПЦПР (Precision)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # ПЦОР
    
    # Распространённость (prevalence)
    prevalence = (tp + fn) / (tp + tn + fp + fn)
    
    # Скорректированные ПЦПР и ПЦОР на другие распространённости
    def adjust_ppv(se, sp, prev):
        return (se * prev) / (se * prev + (1 - sp) * (1 - prev)) if (se * prev + (1 - sp) * (1 - prev)) > 0 else 0
    
    def adjust_npv(se, sp, prev):
        return (sp * (1 - prev)) / ((1 - se) * prev + sp * (1 - prev)) if ((1 - se) * prev + sp * (1 - prev)) > 0 else 0
    
    return {
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'prevalence': prevalence,
        'ppv_at_5pct': adjust_ppv(sensitivity, specificity, 0.05),
        'npv_at_5pct': adjust_npv(sensitivity, specificity, 0.05),
        'ppv_at_0_1pct': adjust_ppv(sensitivity, specificity, 0.001),
        'npv_at_0_1pct': adjust_npv(sensitivity, specificity, 0.001),
    }


def analyze_by_patient(preds, targets, patient_ids, threshold=0.5):
    """Анализ по каждому пациенту."""
    targets_bin = (targets >= 0.5).astype(int)
    preds_bin = (preds >= threshold).astype(int)
    
    results = {}
    for pid in np.unique(patient_ids):
        mask = np.array(patient_ids) == pid
        n_total = mask.sum()
        n_seizures = targets_bin[mask].sum()
        n_pred_seizures = preds_bin[mask].sum()
        
        tp = ((preds_bin[mask] == 1) & (targets_bin[mask] == 1)).sum()
        fp = ((preds_bin[mask] == 1) & (targets_bin[mask] == 0)).sum()
        fn = ((preds_bin[mask] == 0) & (targets_bin[mask] == 1)).sum()
        tn = ((preds_bin[mask] == 0) & (targets_bin[mask] == 0)).sum()
        
        results[pid] = {
            'total': n_total,
            'seizures': n_seizures,
            'pred_seizures': n_pred_seizures,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'recall': n_pred_seizures / n_seizures if n_seizures > 0 else 1.0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'error_rate': (fp + fn) / n_total * 100 if n_total > 0 else 0,
        }
    return results

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True, help='Пути к моделям')
    args = parser.parse_args()
    
    Config.set_seed()
    
    log("="*70)
    log("ТЕСТИРОВАНИЕ EEGNeX v5")
    log("="*70)
    log(f"Моделей: {len(args.models)}")
    log(f"Устройство: {Config.DEVICE}")
    log("="*70)
    
    models = load_models(args.models)
    
    test_dir = Path(Config.OUTPUT_PATH) / "test_patients"
    test_dataset = TestEEGDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    all_preds, all_targets, all_patient_ids = [], [], []
    
    with torch.no_grad():
        for X, target, pids in tqdm(test_loader, desc="Testing"):
            X = X.to(Config.DEVICE)
            pred = predict(models, X)
            all_preds.extend(pred)
            all_targets.extend(target.numpy().flatten())
            all_patient_ids.extend(pids)

    preds = np.array(all_preds)
    targets = np.array(all_targets)
    all_patient_ids = np.array(all_patient_ids)
    
    # После вычисления preds и targets:
    results = compute_all_metrics(preds, targets)
    print_metrics(results, preds, targets)

    # ========== РАСШИРЁННЫЕ МЕТРИКИ ==========
    log("")
    log("="*60)
    log("РАСШИРЁННЫЕ МЕТРИКИ")
    log("="*60)

    for th in [0.3, 0.5, 0.7]:
        ext = compute_extended_metrics(preds, targets, th)
        log(f"--- Порог {th} ---")
        log(f"Чувствительность (Se):     {ext['sensitivity']*100:.1f}%")
        log(f"Специфичность (Sp):         {ext['specificity']*100:.1f}%")
        log(f"ПЦПР (Precision):           {ext['ppv']*100:.1f}%")
        log(f"ПЦОР (NPV):                 {ext['npv']*100:.1f}%")
        log(f"Распространённость:         {ext['prevalence']*100:.1f}%")
        log(f"ПЦПР при P=5%:              {ext['ppv_at_5pct']*100:.1f}%")
        log(f"ПЦОР при P=5%:              {ext['npv_at_5pct']*100:.1f}%")
        log(f"ПЦПР при P=0.1%:            {ext['ppv_at_0_1pct']*100:.1f}%")
        log(f"ПЦОР при P=0.1%:            {ext['npv_at_0_1pct']*100:.1f}%")
        log(f"TP={ext['tp']}, FP={ext['fp']}, FN={ext['fn']}, TN={ext['tn']}")

    # ========== АНАЛИЗ ПО ПАЦИЕНТАМ ==========
    log("")
    log("="*60)
    log("АНАЛИЗ ПО ПАЦИЕНТАМ (порог 0.5)")
    log("="*60)
    log(f"{'ID':<6} {'Всего':<8} {'Приступов':<10} {'Предск':<8} {'TP':<6} {'FP':<6} {'FN':<6} {'Recall%':<9} {'Prec%':<8} {'Ошибок%':<9}")
    log("-"*95)

    patient_results = analyze_by_patient(preds, targets, all_patient_ids, 0.5)
    for pid, res in sorted(patient_results.items(), key=lambda x: int(x[0])):
        log(f"{pid:<6} {res['total']:<8} {res['seizures']:<10} {res['pred_seizures']:<8} "
        f"{res['tp']:<6} {res['fp']:<6} {res['fn']:<6} "
        f"{res['recall']*100:<9.1f} {res['precision']*100:<8.1f} {res['error_rate']:<9.1f}")

    results = compute_all_metrics(preds, targets)
    print_metrics(results, preds, targets)
    
    log("")
    log("="*60)
    log("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    log("="*60)
    
    return results


if __name__ == "__main__":
    try:
        main()
    finally:
        log_file.close()
        print(f"✅ Лог: {log_filename}")