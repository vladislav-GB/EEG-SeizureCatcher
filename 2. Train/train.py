"""
train.py - Обучение модели EEGNeX v5.0

НАЗНАЧЕНИЕ:
    Обучение модели EEGNeX на предобработанных данных с использованием
    AdaptiveFocalLoss и сбалансированных батчей.

НОВШЕСТВА v5.0:
    - AdaptiveFocalLoss с усилением штрафа для приступов
    - Undersampling 1:2 (42 приступа : 86 фона в каждом батче)
    - Окна 3 секунды (768 отсчётов)
    - Все sample_weights игнорируются (всегда 1.0)
    - Curriculum Learning полностью отключён
    - Scheduler по val_pr_auc
    - Сохранение моделей с PR-AUC >= 0.30
    - Чекпоинты каждые 5 эпох

СТРАТЕГИЯ:
    - Soft targets: 0.0 / 0.6 / 0.9 / 1.0 (из preprocessing)
    - AdaptiveFocalLoss: gamma=1.6, alpha_pos=0.8, alpha_neg=0.2
    - Сбалансированные батчи: 1/3 приступов, 2/3 фона
    - Ранняя остановка по PR-AUC

АЛГОРИТМ:
    1. Загрузить train и val датасеты (IterableDataset с undersampling)
    2. Создать модель EEGNeX (вход [B, 19, 768])
    3. Обучать с AdaptiveFocalLoss
    4. Сохранять модели с val_pr_auc >= 0.30
    5. Остановиться при отсутствии улучшений (patience=15)

МЕТРИКИ:
    - val_pr_auc: главная метрика (Precision-Recall AUC)
    - val_auc: дополнительная (ROC-AUC)
    - recall_03, recall_05: чувствительность
    - mean_pred, pct_gt_05: калибровка
    - Гистограмма предсказаний, процентили, разбивка по target

ВХОДНЫЕ ФАЙЛЫ:
    - processed/train_patients/patient_*.pkl
    - processed/val_patients/patient_*.pkl

ВЫХОДНЫЕ ФАЙЛЫ:
    - models/v5.0_epXX_aucX.XXX_prX.XXX.pth — лучшие модели
    - models/history/v5.0_epXX.pth — чекпоинты каждые 5 эпох
    - training_v5_*.txt — лог обучения

ИСПОЛЬЗОВАНИЕ:
    python "2. Train/train.py" --version v5
"""

import sys
import os
import gc
from pathlib import Path
from datetime import datetime
import pynvml
from collections import deque

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from Config.config import Config
from braindecode.models import EEGNeX


# ============================================================
# ЛОГИРОВАНИЕ
# ============================================================

log_dir = Config.LOG_TRAIN_PATH
log_dir.mkdir(parents=True, exist_ok=True)

log_filename = log_dir / f"training_log_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_file = open(log_filename, 'w', encoding='utf-8')

def log(msg, indent=False):
    timestamp = datetime.now().strftime('%H:%M:%S')
    line = f"[{timestamp}]    {msg}" if indent else f"[{timestamp}] {msg}"
    print(line)
    log_file.write(line + '\n')
    log_file.flush()


# ============================================================
# DATASET
# ============================================================

class EEGDataset(IterableDataset):
    """
    v5.3 FINAL SAFE:
    - deque (no RAM explosion)
    - mmap (no IO freeze)
    - stable 1:2 sampling
    """

    def __init__(self, split, shuffle=True, seed=42, epoch=0):
        self.files = sorted((Path(Config.OUTPUT_PATH) / f"{split}_patients").glob("*.npz"))
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch

        self.MAX_BG = 5000
        self.MAX_SZ = 2000

    def __iter__(self):
        worker = get_worker_info()
        files = self.files.copy()

        rng = np.random.default_rng(self.seed + self.epoch)

        if self.shuffle:
            rng.shuffle(files)

        if worker:
            per = len(files) // worker.num_workers
            start = worker.id * per
            end = len(files) if worker.id == worker.num_workers - 1 else start + per
            files = files[start:end]

        bg_pool = deque(maxlen=self.MAX_BG)
        sz_pool = deque(maxlen=self.MAX_SZ)

        sz_threshold = Config.BATCH_SIZE // 3
        bg_threshold = Config.BATCH_SIZE - sz_threshold

        for f in files:
            try:
                data = np.load(f, mmap_mode='r')
                X = data['X']
                y = data['y']

                for i in range(len(y)):
                    if y[i] >= 0.5:
                        sz_pool.append((X[i], y[i]))
                    else:
                        bg_pool.append((X[i], y[i]))

                while len(sz_pool) >= sz_threshold and len(bg_pool) >= bg_threshold:
                    sz_batch = [sz_pool.popleft() for _ in range(sz_threshold)]
                    bg_batch = [bg_pool.popleft() for _ in range(bg_threshold)]

                    batch = sz_batch + bg_batch
                    rng.shuffle(batch)

                    Xb = torch.from_numpy(np.stack([x for x, _ in batch])).float()
                    yb = torch.from_numpy(np.array([y for _, y in batch])).float()

                    yield Xb, yb

                del data, X, y

            except:
                continue

        if len(sz_pool) >= 5 and len(bg_pool) >= 10:
            batch = list(sz_pool)[:5] + list(bg_pool)[:10]
            rng.shuffle(batch)

            Xb = torch.from_numpy(np.stack([x for x, _ in batch])).float()
            yb = torch.from_numpy(np.array([y for _, y in batch])).float()

            yield Xb, yb

# ============================================================
# FOCAL LOSS
# ============================================================

class AdaptiveFocalLoss(torch.nn.Module):
    def __init__(self, gamma=None, alpha_pos=None, alpha_neg=None):
        super().__init__()
        # Берём из Config
        self.gamma = gamma if gamma is not None else Config.FOCAL_GAMMA
        self.alpha_pos = alpha_pos if alpha_pos is not None else Config.FOCAL_ALPHA_POS
        self.alpha_neg = alpha_neg if alpha_neg is not None else Config.FOCAL_ALPHA_NEG

    def forward(self, preds, targets):
        if preds.dim() == 2 and preds.size(1) == 1:
            preds = preds.squeeze(1)

        p = torch.sigmoid(preds).clamp(1e-7, 1 - 1e-7)

        bce = -(targets * torch.log(p) + (1 - targets) * torch.log(1 - p))

        pt = torch.where(targets >= 0.5, p, 1 - p)
        alpha = torch.where(targets >= 0.5, self.alpha_pos, self.alpha_neg)

        loss = alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


# ============================================================
# МЕТРИКИ
# ============================================================

def compute_metrics(preds, targets):
    targets_bin = (targets >= 0.5).astype(int)

    try:
        auc = roc_auc_score(targets_bin, preds)
        pr_auc = average_precision_score(targets_bin, preds)
    except:
        auc, pr_auc = 0.5, 0.0

    preds_bin_03 = (preds >= 0.3).astype(int)
    preds_bin_05 = (preds >= 0.5).astype(int)

    recall_03 = recall_score(targets_bin, preds_bin_03, zero_division=0)
    recall_05 = recall_score(targets_bin, preds_bin_05, zero_division=0)

    return {
        'auc': auc,
        'pr_auc': pr_auc,
        'recall_03': recall_03,
        'recall_05': recall_05,
        'mean_pred': preds.mean(),
        'pct_gt_03': (preds > 0.3).mean() * 100,
        'pct_gt_05': (preds > 0.5).mean() * 100
    }

def compute_extended_metrics(preds, targets):
    """Расширенные метрики для диагностики разделения классов."""
    preds_np = preds.squeeze()
    targets_np = targets
    
    # Процентили
    percentiles = {
        'p10': np.percentile(preds_np, 10),
        'p25': np.percentile(preds_np, 25),
        'p50': np.percentile(preds_np, 50),
        'p75': np.percentile(preds_np, 75),
        'p90': np.percentile(preds_np, 90),
        'p95': np.percentile(preds_np, 95),
    }
    
    # Гистограмма (10 бинов)
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(preds_np, bins=bins)
    hist_pct = hist / len(preds_np) * 100
    
    # Разбивка по target
    target_bins = {
        'target=0.0': (targets_np == 0.0),
        'target=0.6': (targets_np == 0.6),
        'target=0.9': (targets_np == 0.9),
        'target=1.0': (targets_np == 1.0),
    }
    
    target_metrics = {}
    for name, mask in target_bins.items():
        if mask.sum() > 0:
            target_metrics[name] = {
                'mean_pred': preds_np[mask].mean(),
                'std_pred': preds_np[mask].std(),
                'median_pred': np.median(preds_np[mask]),
                'n': mask.sum()
            }
    
    # Бимодальность: разница между средними для target=0.0 и target=1.0
    if 'target=0.0' in target_metrics and 'target=1.0' in target_metrics:
        separation = target_metrics['target=1.0']['mean_pred'] - target_metrics['target=0.0']['mean_pred']
    else:
        separation = 0.0
    
    return {
        'percentiles': percentiles,
        'hist_pct': hist_pct.tolist(),
        'target_metrics': target_metrics,
        'mean_pred': preds_np.mean(),
        'std_pred': preds_np.std(),
        'separation': separation
    }


def log_extended_metrics(metrics, epoch):
    """Форматированный вывод расширенных метрик."""
    log(f"   Mean pred={metrics['mean_pred']:.4f} ± {metrics['std_pred']:.4f}", indent=True)
    
    p = metrics['percentiles']
    log(f"   Процентили: P10={p['p10']:.4f} P25={p['p25']:.4f} P50={p['p50']:.4f} P75={p['p75']:.4f} P90={p['p90']:.4f} P95={p['p95']:.4f}", indent=True)
    
    hist_str = " ".join([f"{v:.1f}%" for v in metrics['hist_pct']])
    log(f"   Гистограмма [0-1]: {hist_str}", indent=True)
    
    log(f"   По target:", indent=True)
    for name, tm in metrics['target_metrics'].items():
        log(f"      {name}: n={tm['n']:6d} mean={tm['mean_pred']:.4f} ± {tm['std_pred']:.4f} median={tm['median_pred']:.4f}", indent=True)
    
    log(f"   Разделение (target=1.0 - target=0.0): {metrics['separation']:.4f}", indent=True)

# ============================================================
# TRAIN
# ============================================================

def train(model_version="v5.1", resume_path=None):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    Config.set_seed()

    log("="*60)
    log(f"ОБУЧЕНИЕ EEGNeX {model_version}")
    log("="*60)
    log(f"Окна: {Config.WINDOW_SEC} сек ({Config.WINDOW_SIZE} отсчётов)")
    log(f"Focal Loss: gamma={Config.FOCAL_GAMMA}, alpha_pos={Config.FOCAL_ALPHA_POS}")
    log(f"BATCH_SIZE: {Config.BATCH_SIZE}")
    log("="*60)

    HISTORY_PATH = Path(Config.MODEL_PATH) / "history"
    HISTORY_PATH.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True

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

    log(f"🧠 EEGNeX: {sum(p.numel() for p in model.parameters()):,} параметров")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max',
        patience=Config.LR_PATIENCE,
        factor=Config.LR_FACTOR
    )

    criterion = AdaptiveFocalLoss(
        gamma=Config.FOCAL_GAMMA,
        alpha_pos=Config.FOCAL_ALPHA_POS,
        alpha_neg=Config.FOCAL_ALPHA_NEG
    )

    # ========== СТАРТ|РЕСТАРТ ==========
    if resume_path:
        log(f"Загружаем чекпоинт: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_pr_auc = checkpoint.get('val_pr_auc', 0.0)
        log(f"Продолжаем с эпохи {start_epoch}, best PR-AUC = {best_val_pr_auc:.4f}")
    else:
        start_epoch = 0
        best_val_pr_auc = 0.0

    patience_counter = 0

    log("="*60)
    log("НАЧАЛО ОБУЧЕНИЯ")
    log("="*60)

    for epoch in range(start_epoch, Config.EPOCHS):

        train_loader = DataLoader(
            EEGDataset('train', epoch=epoch),
            batch_size=None,       # ← КРИТИЧЕСКИ ВАЖНО! Датасет сам отдаёт батчи
            num_workers=0,         # ← Безопасный режим (без многопоточности)
            pin_memory=True
        )

        val_loader = DataLoader(
            EEGDataset('val', shuffle=False),
            batch_size=None,
            num_workers=2,
            pin_memory=True
        )

        log("")
        log("="*60)
        log(f"ЭПОХА {epoch+1}/{Config.EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        log("="*60)

        # ================= TRAIN =================
        model.train()
        train_loss_sum, train_samples = 0.0, 0
        batch_count = 0
        last_gpu_util = 0
        last_mem_used = 0
        last_mem_total = 0  
        
        pbar = tqdm(train_loader, desc=f"Train {epoch+1}", ncols=100)

        for X, y in pbar:
            X = X.to(Config.DEVICE, non_blocking=True)
            y = y.to(Config.DEVICE, non_blocking=True)

            if X.dim() == 2:
                X = X.unsqueeze(0)

            optimizer.zero_grad()

            pred = model(X)
            loss = criterion(pred, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP_NORM)
            optimizer.step()

            train_loss_sum += loss.item() * X.size(0)
            train_samples += X.size(0)

            # Обновляем GPU метрики КАЖДЫЕ 10 БАТЧЕЙ
            if batch_count % 10 == 0:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    last_gpu_util = util.gpu
                    last_mem_used = mem.used / 1024**2
                    last_mem_total = mem.total / 1024**2
                except:
                    pass
            # ВСЕГДА показываем последние известные значения
            if last_gpu_util > 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    GPU=f"{last_gpu_util}%",
                    VRAM=f"{last_mem_used:.0f}/{last_mem_total:.0f}MB"
                )
            else:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            batch_count += 1
        
        pbar.close()
        train_loss = train_loss_sum / max(train_samples, 1)
        log(f"Train loss={train_loss:.4f}", indent=True)
        log(f"GPU util: {last_gpu_util}%, VRAM: {last_mem_used:.0f}/{last_mem_total:.0f}MB", indent=True)

        # ================= VAL =================
        model.eval()
        val_loss_sum, val_samples = 0.0, 0
        all_preds, all_targets = [], []
        batch_count = 0
        last_gpu_util = 0
        last_mem_used = 0
        last_mem_total = 0 

        val_pbar = tqdm(val_loader, desc=f"Val {epoch+1}", ncols=100)

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(Config.DEVICE, non_blocking=True)
                y = y.to(Config.DEVICE, non_blocking=True)

                if X.dim() == 2:
                    X = X.unsqueeze(0)

                pred = model(X)
                loss = criterion(pred, y)

                val_loss_sum += loss.item() * X.size(0)
                val_samples += X.size(0)

                all_preds.append(torch.sigmoid(pred).detach().cpu().numpy())
                all_targets.append(y.cpu().numpy())
                
                # Обновляем GPU метрики КАЖДЫЕ 10 БАТЧЕЙ
                if batch_count % 10 == 0:
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        last_gpu_util = util.gpu
                        last_mem_used = mem.used / 1024**2
                        last_mem_total = mem.total / 1024**2
                    except:
                        pass

                # ВСЕГДА показываем последние известные значения
                if last_gpu_util > 0:
                    val_pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        GPU=f"{last_gpu_util}%",
                        VRAM=f"{last_mem_used:.0f}/{last_mem_total:.0f}MB"
                    )
                else:
                    val_pbar.set_postfix(loss=f"{loss.item():.4f}")

                batch_count += 1

        val_pbar.close()
        val_loss = val_loss_sum / max(val_samples, 1)

        log(f"Val loss={val_loss:.4f}", indent=True)
        log(f"GPU util: {last_gpu_util}%, VRAM: {last_mem_used:.0f}/{last_mem_total:.0f}MB", indent=True)

        all_preds = np.concatenate(all_preds).squeeze()
        all_targets = np.concatenate(all_targets)

        metrics = compute_metrics(all_preds, all_targets)
        
        val_auc = metrics['auc']
        val_pr_auc = metrics['pr_auc']
        recall_03 = metrics['recall_03']
        recall_05 = metrics['recall_05']

        ext_metrics = compute_extended_metrics(all_preds, all_targets)
        log_extended_metrics(ext_metrics, epoch+1)
        
        log("")
        log(f"PR-AUC={val_pr_auc:.4f} | AUC={val_auc:.4f}", indent=True)
        log(f"Recall@0.3={recall_03*100:.1f}%, Recall@0.5={recall_05*100:.1f}%", indent=True)
        log(f"Mean pred={metrics['mean_pred']:.4f}, >0.3={metrics['pct_gt_03']:.1f}%, >0.5={metrics['pct_gt_05']:.1f}%", indent=True)

        scheduler.step(val_pr_auc)

        # ================= СОХРАНЕНИЕ =================
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_auc': val_auc,
            'val_pr_auc': val_pr_auc,
            'version': model_version
        }
        
        separation = ext_metrics.get('separation', 0.0)
        prev_best = best_val_pr_auc

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            model_name = f"{model_version}_ep{epoch+1:02d}_auc{val_auc:.3f}_pr{val_pr_auc:.3f}_sep{separation:.3f}.pth"
            torch.save(checkpoint, Path(Config.MODEL_PATH) / model_name)
            log(f"💾 Сохранена (рекорд): {model_name}", indent=True)
            patience_counter = 0

        elif val_pr_auc >= Config.SAVE_PR_THRESHOLD and separation >= Config.SAVE_SEP_THRESHOLD:
            model_name = f"{model_version}_ep{epoch+1:02d}_auc{val_auc:.3f}_pr{val_pr_auc:.3f}_sep{separation:.3f}.pth"
            torch.save(checkpoint, Path(Config.MODEL_PATH) / model_name)
            log(f"💾 Сохранена (PR≥{Config.SAVE_PR_THRESHOLD}, sep≥{Config.SAVE_SEP_THRESHOLD}): {model_name}", indent=True)
            patience_counter += 1
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, HISTORY_PATH / f"{model_version}_ep{epoch+1:02d}.pth")
            log(f"📂 History: ep{epoch+1:02d}", indent=True)

        improved = "✅" if val_pr_auc > prev_best else "  "
        log(f"Epoch {epoch+1:2d}/{Config.EPOCHS} | "
            f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
            f"best: {best_val_pr_auc:.4f} {improved}")

        if patience_counter >= Config.PATIENCE:
            log(f"Ранняя остановка на эпохе {epoch+1}")
            break

        gc.collect()
        torch.cuda.empty_cache()

    log("")
    log("="*60)
    log("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    log(f"Лучший val_pr_auc: {best_val_pr_auc:.4f}")
    log("="*60)

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='v5.1')
    parser.add_argument('--resume', type=str, default=None)
    args, _ = parser.parse_known_args()

    try:
        train(model_version=args.version, resume_path=args.resume)
    finally:
        log_file.close()
        print(f"✅ Лог сохранён в файл: {log_filename}")