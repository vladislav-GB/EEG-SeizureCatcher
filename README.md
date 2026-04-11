# EEG-SeizureCatcher

**Детекция эпилептических приступов по ЭЭГ с учётом неопределённости экспертных аннотаций**

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Оглавление

- [Описание](#-описание)
- [Методология](#-методология)
- [Архитектура](#-архитектура)
- [Установка](#-установка)
- [Запуск](#-запуск)
- [Результаты](#-результаты)
- [Ссылки](#-ссылки)

---

## 🧠 Описание

**EEG-SeizureCatcher** — это система автоматической детекции эпилептических приступов по данным электроэнцефалографии (ЭЭГ) с использованием нейросетевых методов.

### Ключевые особенности

- ✅ **Учёт неопределённости экспертов** — мягкие метки (soft labels) вместо бинарных
- ✅ **Взвешенное обучение** — разные веса для разной степени согласия экспертов
- ✅ **Контекстные окна** — анализ 5-секундных фрагментов с перекрытием
- ✅ **Интерпретируемость** — Grad-CAM визуализация важных участков сигнала
- ✅ **Воспроизводимость** — фиксированный seed и детальное логирование

---

## 📊 Методология

### 1. Данные

- **Датасет:** Helsinki Neonatal EEG Dataset (79 пациентов)
- **Эксперты:** 3 независимых эксперта (A, B, C)
- **Разметка:** посекундная (0 — нет приступа, 1 — есть приступ)

### 2. Предобработка

# Для каждой секунды
target = (A + B + C) / 3           
weight = количество_экспертов_за / 3  

# Исключение для фона (000)
if target == 0:
    weight = 1.0                    

### 3. Окна и перекрытие
Параметр	  Значение
Длина окна	  5 секунд (1280 точек)
Шаг окна	  1 секунда (256 точек)
Перекрытие	  4 секунды
Веса позиций  [0.1, 0.2, 0.4, 0.2, 0.1]

### 4. Функция потерь
loss = BCE(pred, target) × weight

## Архитектура

### Модель EEGNeX

Input: [Batch, 19 channels, 1280 time points]
    ↓
Temporal Convolution (filter_1=8)
    ↓
Depthwise Convolution
    ↓
Pointwise Convolution
    ↓
Separable Convolution (filter_2=32)
    ↓
Global Average Pooling
    ↓
Dropout (0.5)
    ↓
Output: [Batch, 1] (logit)

Параметры модели:

Параметр	Значение
Каналов	       19
Частота	      256 Гц
Параметров	 ~183K
Dropout	      0.5
Активация	  ELU

## 🚀 Установка

1. Клонирование репозитория

git clone https://github.com/yourusername/EEG-SeizureCatcher.git
cd EEG-SeizureCatcher

Создайте папку для временных файлов на диске с местом:
mkdir -p "/media/avengus/Локальный диск/Dev/tmp_pip"

Установите переменные окружения:
export TMPDIR="/media/avengus/Локальный диск/Dev/tmp_pip"
export TEMP="/media/avengus/Локальный диск/Dev/tmp_pip"
export TMP="/media/avengus/Локальный диск/Dev/tmp_pip"

Добавьте в .bashrc для постоянного использования:
echo 'export TMPDIR="/media/avengus/Локальный диск/Dev/tmp_pip"' >> ~/.bashrc
echo 'export TEMP="/media/avengus/Локальный диск/Dev/tmp_pip"' >> ~/.bashrc
echo 'export TMP="/media/avengus/Локальный диск/Dev/tmp_pip"' >> ~/.bashrc

# Примените
source ~/.bashrc

2. Создание виртуального окружения

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows

3. Установка зависимостей

pip install -r requirements.txt
pip freeze > requirements.txt
pip list

## 🎯 Запуск

1. Предобработка данных

python Preprocessing/preprocessing.py

Что происходит:

    Загрузка EDF файлов и аннотаций

    Фильтрация (0.5-30 Гц + режектор 50 Гц)

    Нарезка окон (5 сек, шаг 1 сек)

    Расчёт target и weight

    Сохранение .pkl файлов

Выходные файлы:

    helsinki_train.pkl

    helsinki_val.pkl

    helsinki_test.pkl

2. Обучение модели

python Training/train.py

Что происходит:

    Загрузка предобработанных данных

    Обучение EEGNeX с weighted BCE loss

    Early stopping по val_loss

    Сохранение лучшей модели

3. Оценка модели

python Evaluation/evaluate.py

Метрики:

    ROC-AUC

    PR-AUC (главная метрика для дисбаланса)

    F1-score

4. Визуализация

# Просмотр ЭЭГ с аннотациями
python Visualization/edf_viewer.py --patient 41

# Grad-CAM для конкретного окна
python Visualization/visualize.py --patient 16 --window 0

## 📚 Ссылки

    Датасет Helsinki

    EEGNeX

    MNE-Python

## 📝 Лицензия

MIT License

## 🌑 Автор

avengus.gb