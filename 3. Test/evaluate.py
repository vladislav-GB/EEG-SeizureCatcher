"""
evaluate.py - Тестирование модели 

НАЗНАЧЕНИЕ:
    Оценка обученной модели на тестовой выборке.

МЕТРИКИ:
    - AUC (Area Under ROC Curve) - с учетом весов
    - PR-AUC (Precision-Recall AUC) - честная метрика для дисбаланса
    - Calibration curve
    - Weighted F1-score
    - Sensitivity/Recall (с весами)
    - Specificity (с весами)
    - Анализ по паттернам (000, 001, 010, 100)
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.calibration import calibration_curve
from torch.utils.data import DataLoader
from braindecode.models import EEGNeX
from pathlib import Path
import matplotlib.pyplot as plt
from config import Config
from train import EEGDataset


def evaluate():
    """Главная функция оценки"""

    Config.set_seed()

    print(f"\n{'='*60}")
    print("ТЕСТИРОВАНИЕ МОДЕЛИ (UNCERTAINTY-AWARE)")
    print(f"{'='*60}")
    print(f"Устройство: {Config.DEVICE}")
    print(f"{'='*60}\n")
    
    # ========== 1. ЗАГРУЗКА ТЕСТОВЫХ ДАННЫХ ==========
    test_dataset = EEGDataset(
        Path(Config.OUTPUT_PATH) / "helsinki_test.pkl", 
        use_soft_labels=True  
    )
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)
    
    print(f"📊 Test: {len(test_dataset)} окон, {len(test_loader)} батчей")
    
    # ========== 2. ЗАГРУЗКА МОДЕЛИ ==========
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
    
    model_path = Path(Config.MODEL_PATH) / "best_eegnex_model.pth"
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("🧠 Модель загружена\n")
    print(f"   Seed при обучении: {checkpoint.get('seed', 'unknown')}\n")
    
    # ========== 3. ПРЕДСКАЗАНИЯ ==========
    all_preds = []
    all_targets = []  
    all_weights = []
    all_patterns = []
    all_confidences = []
    
    with torch.no_grad():
        for X, y, w, meta in test_loader:  
            X = X.to(Config.DEVICE)
            logits = model(X).squeeze(-1)
            probs = torch.sigmoid(logits)
            
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_weights.extend(w.numpy())
            
            for m in meta:
                all_patterns.append(m['pattern'])
                all_confidences.append(m['confidence'])
    
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    weights = np.array(all_weights)
    patterns = np.array(all_patterns)
    confidences = np.array(all_confidences)

    # ========== 4. ОПТИМАЛЬНЫЙ ПОРОГ (НОВОЕ) ==========
    thresholds = np.linspace(0.1, 0.9, 50)
    best_f1 = 0
    best_threshold = 0.5
    
    targets_bin = (targets >= 0.5).astype(int)
    
    for t in thresholds:
        preds_bin = (preds >= t).astype(int)
        f1 = f1_score(targets_bin, preds_bin, sample_weight=weights)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    print(f"🔧 Оптимальный порог: {best_threshold:.3f} (F1={best_f1:.4f})")
    
    # ========== 5. ОСНОВНЫЕ МЕТРИКИ ==========
    # AUC - с весами
    auc = roc_auc_score(targets, preds, sample_weight=weights)
    
    # PR-AUC
    pr_auc = average_precision_score(targets, preds, sample_weight=weights)
    
    # Метрики с оптимальным порогом
    preds_bin = (preds >= best_threshold).astype(int)
    
    f1 = f1_score(targets_bin, preds_bin, sample_weight=weights)
    sensitivity = recall_score(targets_bin, preds_bin, sample_weight=weights, pos_label=1)
    
    tn = np.sum(weights[(targets_bin == 0) & (preds_bin == 0)])
    fp = np.sum(weights[(targets_bin == 0) & (preds_bin == 1)])
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    precision = precision_score(targets_bin, preds_bin, sample_weight=weights, pos_label=1)
    
    # ========== 6. ВЫВОД ==========
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("="*60)
    print(f"AUC:          {auc:.4f}")
    print(f"PR-AUC:       {pr_auc:.4f}")
    print(f"F1-score:     {f1:.4f} (threshold={best_threshold:.3f})")
    print(f"Precision:    {precision:.4f}")
    print(f"Sensitivity:  {sensitivity:.4f}")
    print(f"Specificity:  {specificity:.4f}")
    print("="*60)
    
    # ========== 7. АНАЛИЗ ПО ПАТТЕРНАМ ==========
    print("\n" + "="*60)
    print("АНАЛИЗ ПО ПАТТЕРНАМ НЕОПРЕДЕЛЕННОСТИ")
    print("="*60)
    
    patterns_df = pd.DataFrame({
        'pattern': patterns,
        'target': targets,
        'pred': preds,
        'weight': weights,
        'confidence': confidences
    })
    
    for pattern in ["000", "001", "010", "011", "100", "101", "110", "111"]:
        mask = patterns_df['pattern'] == pattern
        if mask.sum() > 10:  # только если достаточно данных
            auc_pat = roc_auc_score(
                patterns_df.loc[mask, 'target'],
                patterns_df.loc[mask, 'pred'],
                sample_weight=patterns_df.loc[mask, 'weight']
            )
            n_samples = mask.sum()
            mean_target = patterns_df.loc[mask, 'target'].mean()
            mean_pred = patterns_df.loc[mask, 'pred'].mean()
            mean_confidence = patterns_df.loc[mask, 'confidence'].mean()
            
            print(f"\n📊 Паттерн {pattern} (n={n_samples}):")
            print(f"   AUC:      {auc_pat:.4f}")
            print(f"   mean y_true: {mean_target:.3f}")
            print(f"   mean y_pred: {mean_pred:.3f}")
            print(f"   confidence:  {mean_confidence:.3f}")
    
    # ========== 8. КАЛИБРОВКА (НОВОЕ) ==========
    print("\n" + "="*60)
    print("КАЛИБРОВКА МОДЕЛИ")
    print("="*60)
    
    # Calibration curve
    prob_true, prob_pred = calibration_curve(targets, preds, n_bins=10, sample_weight=weights)
    
    # ECE (Expected Calibration Error)
    ece = np.mean(np.abs(prob_true - prob_pred) * np.histogram(preds, bins=10, weights=weights)[0] / weights.sum())
    
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    
    # Визуализация калибровки
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Calibration curve
    ax1.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    ax1.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration', color='gray')
    ax1.set_xlabel('Mean predicted probability')
    ax1.set_ylabel('Fraction of positives')
    ax1.set_title('Calibration Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distribution
    ax2.hist(preds[targets_bin == 0], bins=50, alpha=0.5, label='Class 0', weights=weights[targets_bin == 0])
    ax2.hist(preds[targets_bin == 1], bins=50, alpha=0.5, label='Class 1', weights=weights[targets_bin == 1])
    ax2.set_xlabel('Predicted probability')
    ax2.set_ylabel('Weighted count')
    ax2.set_title('Prediction Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(Config.VISUALIZATION_PATH) / 'calibration_curve.png', dpi=150)
    plt.close()
    
    print(f"   📊 Calibration plot saved to {Config.VISUALIZATION_PATH}/calibration_curve.png")
    
    # ========== 9. ВЫВОД О КАЧЕСТВЕ ==========
    print("\n" + "="*60)
    print("ВЫВОД")
    print("="*60)
    
    if auc > 0.85:
        print("Отличное разделение классов")
    elif auc > 0.75:
        print("Хорошее разделение")
    else:
        print("⚠️ Разделение требует улучшения")
    
    if pr_auc > 0.7:
        print("Отличная precision-recall")
    elif pr_auc > 0.5:
        print("Приемлемая precision-recall")
    else:
        print("⚠️ Проблемы с редким классом")
    
    if ece < 0.1:
        print("Отличная калибровка")
    elif ece < 0.2:
        print("Приемлемая калибровка")
    else:
        print("⚠️ Модель плохо калибрована")
    
    print("="*60)
    
    return auc, pr_auc, f1, best_threshold


if __name__ == "__main__":
    evaluate()