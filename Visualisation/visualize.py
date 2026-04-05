"""
visualize.py - Grad-CAM визуализация для EEGNeX

НАЗНАЧЕНИЕ:
    Показать, на какие участки ЭЭГ сигнала модель обращает внимание при принятии решения.

ИСПОЛЬЗОВАНИЕ:
    python visualize.py --patient 16 --window 5 --class 1
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from config import Config
from train import EEGDataset
import mne
from mne.channels import make_standard_montage
from mne.viz import plot_topomap

# ============================================================================
# 1. GRAD-CAM ДЛЯ EEG 
# ============================================================================

class EEGGradCAM:
    """
    Grad-CAM для ЭЭГ сигналов.
    """
    
    def __init__(self, model, target_layer):
        """
        Аргументы:
            model: обученная модель EEGNeX
            target_layer: последний сверточный слой (например, model.separable_conv)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Регистрируем хуки
        target_layer.register_forward_hook(self._save_activation)
        # Используем full_backward_hook
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Сохраняет активации целевого слоя"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Сохраняет градиенты целевого слоя"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, x, target_class=None):
        """
        Генерирует карту активации для входного сигнала.
        
        Аргументы:
            x: входной тензор [1, channels, time]
            target_class: класс, для которого строим карту (0 или 1)
        
        Возвращает:
            cam: карта внимания [time] (нормализована от 0 до 1)
            channel_attention: важность каналов [channels]
        """
        # Forward pass
        x = x.unsqueeze(0)  # [1, 1, channels, time] для EEGNeX
        output = self.model(x)
        
        if target_class is None:
            target_class = 1 if torch.sigmoid(output) > 0.5 else 0
        
        # Backward pass
        self.model.zero_grad()
        
        # Для бинарной классификации
        if output.shape[-1] == 1:
            output[0, 0].backward()
        else:
            one_hot = torch.zeros_like(output)
            one_hot[0][target_class] = 1
            output.backward(gradient=one_hot)
        
        # self.gradients: [batch, channels, freq, time]
        weights = torch.mean(self.gradients, dim=(2, 3))  # [batch, channels]
        
        # Взвешиваем активации
        # self.activations: [batch, channels, freq, time]
        cam = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * self.activations, dim=1)
        
        # Берем среднее по частотным бинам
        cam = torch.mean(cam, dim=1)  # [batch, time]
        
        # ReLU и нормализация
        cam = torch.clamp(cam, min=0)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        
        # Важность каналов
        channel_attention = torch.mean(weights, dim=0).cpu().numpy()
        channel_attention = (channel_attention - channel_attention.min()) / (channel_attention.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy(), channel_attention


# ============================================================================
# 2. ФУНКЦИИ ДЛЯ ПОЛУЧЕНИЯ ДАННЫХ
# ============================================================================

def get_window_from_dataset(dataset, patient_id, window_idx):
    """
    Получаем конкретное окно из датасета по patient_id и индексу окна
    
    Аргументы:
        dataset: EEGDataset
        patient_id: номер пациента
        window_idx: индекс окна для этого пациента (0-based)
    
    Возвращает:
        X: сигнал [channels, time]
        y: метка
        weight: вес
        meta: метаданные
    """
    count = 0
    for i in range(len(dataset)):
        X, y, w, meta = dataset[i]
        if meta['patient'] == patient_id:
            if count == window_idx:
                return X, y, w, meta
            count += 1
    return None, None, None, None


# ============================================================================
# 3. ВИЗУАЛИЗАЦИЯ
# ============================================================================

def plot_signal_with_cam(signal, cam, channel_attention, channel_names, sample_rate, 
                         meta, save_path, patient_id, window_idx):
    """
    Визуализирует ЭЭГ сигнал с наложением карты внимания.
    """
    time_axis = np.arange(signal.shape[1]) / sample_rate
    
    # Выбираем топ-5 каналов по важности
    top_channels = np.argsort(channel_attention)[-5:][::-1]
    
    fig, axes = plt.subplots(len(top_channels) + 2, 1, 
                              figsize=(15, (len(top_channels) + 2) * 1.5),
                              sharex=True)
    
    # Рисуем топ-5 каналов
    for idx, ch in enumerate(top_channels):
        axes[idx].plot(time_axis, signal[ch], color='gray', alpha=0.7, linewidth=0.8)
        axes[idx].set_ylabel(channel_names[ch], rotation=0, labelpad=20, fontsize=9)
        axes[idx].set_yticks([])
        axes[idx].set_ylim(-3, 3)
        
        # Подсвечиваем важные участки
        for t in range(len(cam)):
            if cam[t] > 0.7:
                axes[idx].axvspan(time_axis[t], time_axis[t] + 0.02, 
                                 alpha=0.3, color='red')
            elif cam[t] > 0.4:
                axes[idx].axvspan(time_axis[t], time_axis[t] + 0.02,
                                 alpha=0.15, color='orange')
    
    # Карта внимания
    axes[-2].fill_between(time_axis, cam, alpha=0.6, color='red')
    axes[-2].set_ylabel('Внимание', rotation=0, labelpad=20, fontsize=9)
    axes[-2].set_ylim(0, 1)
    axes[-2].set_yticks([0, 0.5, 1])
    axes[-2].grid(True, alpha=0.3)
    
    # Важность каналов (горизонтальная)
    axes[-1].barh(channel_names, channel_attention, color='steelblue')
    axes[-1].set_xlabel('Важность канала', fontsize=9)
    axes[-1].set_xlim(0, 1)
    axes[-1].axvline(x=0.5, color='orange', linestyle='--', alpha=0.5)
    axes[-1].axvline(x=0.7, color='red', linestyle='--', alpha=0.5)
    axes[-1].tick_params(axis='y', labelsize=7)
    
    # Заголовок с метаданными
    title = (f"Grad-CAM: Patient {patient_id}, Window {window_idx}\n"
             f"Pattern: {meta['pattern']}, "
             f"y_true={meta['y_prob']:.2f} ({meta['y_hard']}), "
             f"Confidence={meta['confidence']:.2f}, "
             f"Agreement={meta['agreement']:.2f}")
    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    
    # Сохраняем
    filepath = save_path / f"gradcam_p{patient_id}_w{window_idx}.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Сохранено: {filepath.name}")


def plot_topomap_at_time(signal, cam, montage, time_point, sample_rate, 
                         save_path, patient_id, window_idx, channel_names):
    """
    Визуализирует топографическую карту в момент времени.
    """
    idx = int(time_point * sample_rate)
    if idx >= signal.shape[1]:
        idx = signal.shape[1] - 1
    
    # Взвешиваем сигнал вниманием
    channel_values = signal[:, idx] * cam[idx]
    
    # Находим позиции каналов в монтаже
    ch_pos = {ch: montage.get_positions().get_positions()[ch] 
              for ch in channel_names if ch in montage.ch_names}
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Сортируем каналы
    ch_names_ordered = [ch for ch in channel_names if ch in ch_pos]
    ch_values_ordered = [channel_values[channel_names.tolist().index(ch)] 
                         for ch in ch_names_ordered]
    
    im, cn = plot_topomap(ch_values_ordered, ch_pos, axes=ax, show=False, 
                          cmap='RdBu_r', names=ch_names_ordered)
    ax.set_title(f"Внимание модели в момент t={time_point:.2f} сек\n"
                 f"Patient {patient_id}, Window {window_idx}")
    plt.colorbar(im, ax=ax, label='Важность × сигнал')
    
    # Сохраняем
    filepath = save_path / f"topomap_p{patient_id}_w{window_idx}_t{time_point:.2f}.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Сохранено: {filepath.name}")


# ============================================================================
# 4. ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def visualize_patient(patient_id, window_idx=0, target_class=None):
    """
    Визуализирует Grad-CAM для конкретного пациента и окна.
    """
    # Создаем папку для визуализаций
    vis_path = Path(Config.VISUALIZATION_PATH) / f"patient_{patient_id}"
    vis_path.mkdir(parents=True, exist_ok=True)
    
    # ========== 1. ЗАГРУЗКА МОДЕЛИ ==========
    from braindecode.models import EEGNeX
    
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
    )
    
    model_path = Path(Config.MODEL_PATH) / "best_eegnex_model.pth"
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Выбираем целевой слой
    target_layer = model.separable_conv
    
    gradcam = EEGGradCAM(model, target_layer)
    
    # ========== 2. ЗАГРУЗКА ДАННЫХ ==========
    # Загружаем тестовый датасет 
    test_dataset = EEGDataset(Path(Config.OUTPUT_PATH) / "helsinki_test.pkl")
    
    X, y, w, meta = get_window_from_dataset(test_dataset, patient_id, window_idx)
    
    if X is None:
        print(f"❌ Пациент {patient_id}, окно {window_idx} не найдены в тестовом датасете")
        print("   Пробуем в валидационном...")
        
        val_dataset = EEGDataset(Path(Config.OUTPUT_PATH) / "helsinki_val.pkl")
        X, y, w, meta = get_window_from_dataset(val_dataset, patient_id, window_idx)
        
        if X is None:
            print(f"❌ Пациент {patient_id} не найден")
            return
    
    print(f"\n📊 Данные окна:")
    print(f"   Patient: {meta['patient']}")
    print(f"   Pattern: {meta['pattern']}")
    print(f"   y_prob: {meta['y_prob']:.3f}")
    print(f"   y_hard: {meta['y_hard']}")
    print(f"   Confidence: {meta['confidence']:.3f}")
    print(f"   Agreement: {meta['agreement']:.3f}")
    print(f"   Weight: {w:.3f}")
    
    # ========== 3. ГЕНЕРАЦИЯ CAM ==========
    if target_class is None:
        target_class = meta['y_hard']
    
    print(f"\n🎯 Целевой класс для Grad-CAM: {target_class}")
    
    with torch.no_grad():
        X_tensor = X.unsqueeze(0)  # [1, channels, time]
        logits = model(X_tensor)
        prob = torch.sigmoid(logits).item()
        print(f"🤖 Модель предсказывает: {prob:.3f} (класс {1 if prob >= 0.5 else 0})")
    
    cam, channel_attention = gradcam.generate_cam(X, target_class=target_class)
    
    # ========== 4. ВИЗУАЛИЗАЦИЯ ==========
    signal = X.numpy()  # [channels, time]
    
    plot_signal_with_cam(
        signal, cam, channel_attention, np.array(Config.CHANNELS),
        Config.TARGET_SR, meta, vis_path, patient_id, window_idx
    )
    
    # ========== 5. ТОПОГРАФИЧЕСКАЯ КАРТА ==========
    montage = make_standard_montage('standard_1020')
    
    # В момент максимального внимания
    max_attention_time = np.argmax(cam) / Config.TARGET_SR
    
    plot_topomap_at_time(
        signal, cam, montage, max_attention_time, Config.TARGET_SR,
        vis_path, patient_id, window_idx, np.array(Config.CHANNELS)
    )
    
    print(f"\n✅ Визуализация сохранена в {vis_path}")


# ============================================================================
# 5. ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grad-CAM визуализация (v2.0)')
    parser.add_argument('--patient', type=int, default=16, help='Номер пациента')
    parser.add_argument('--window', type=int, default=0, help='Индекс окна')
    parser.add_argument('--class', dest='target_class', type=int, default=None,
                        help='Целевой класс для CAM (0 или 1, по умолчанию = y_true)')
    
    args = parser.parse_args()
    
    visualize_patient(args.patient, args.window, args.target_class)