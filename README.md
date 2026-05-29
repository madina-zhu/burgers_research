# Исследование нейросетевых методов решения уравнения Бюргерса

## Описание проекта

Данный проект посвящён **сравнительному анализу трёх нейросетевых архитектур** для решения одномерного уравнения Бюргерса:

- **PINN** (Physics-Informed Neural Network) — физически информированная нейронная сеть
- **DeepONet** (Deep Operator Network) — глубокая операторная сеть
- **FNO** (Fourier Neural Operator) — оператор Фурье

Уравнение Бюргерса объединяет нелинейную конвекцию и линейную диффузию:

$$u_t + u u_x - \nu u_{xx} = 0, \quad \nu = 0.01/\pi$$

## Структура проекта

```markdown
burgers_research-main/
│
├── models/                          # Основные модели
│   ├── burgers_pinn_fnn_tf.py       # PINN (TensorFlow + DeepXDE)
│   ├── burgers_deeponet.py          # DeepONet (TensorFlow + DeepXDE)
│   └── burgers_fno.py               # FNO (PyTorch)
│
├── experiments/                     # Эксперименты
│   ├── experiment_layers.py         # Сравнение PINN с разным кол-вом слоёв
│   └── visualize_analytic.py        # Визуализация аналитического решения
│
├── utils/                           # Вспомогательные скрипты
│   ├── plot_results.py              # Сравнение PINN и DeepONet
│   └── visualize_dataset.py         # Визуализация датасета
│
├── dataset/                         # Данные
│   └── Burgers.npz                  # Эталонное решение (DeepXDE)
│
├── results/                         # Результаты (создаётся автоматически)
│   ├── pinn_loss.png
│   ├── pinn_solution.png
│   ├── deeponet_loss.png
│   ├── deeponet_prediction.png
│   ├── fno_loss.png
│   ├── fno_solution.png
│   ├── comparison_plot.png
│   ├── experiment_layers.png
│   ├── analytic_solution_2d.png
│   ├── analytic_solution_3d.png
│   ├── analytic_vs_numerical.png
│   └── dataset_visualization.png
│
├── requirements.txt                 # Зависимости
└── README.md
```

## Установка и запуск

### 1. Клонирование репозитория

```bash
git clone https://github.com/madina-zhu/burgers_research.git
cd burgers_research
```

### 2. Создание виртуального окружения

```bash
python3.10 -m venv venv_burgers
source venv_burgers/bin/activate   # Linux/Mac
# или
venv_burgers\Scripts\activate      # Windows
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Запуск моделей

```bash
# PINN
python3 models/burgers_pinn_fnn_tf.py

# DeepONet
python3 models/burgers_deeponet.py

# FNO
python3 models/burgers_fno.py

# Сравнение PINN и DeepONet
python3 utils/plot_results.py

# Эксперимент со слоями PINN
python3 experiments/experiment_layers.py

# Визуализация датасета
python3 utils/visualize_dataset.py

# Визуализация аналитического решения
python3 experiments/visualize_analytic.py
```

## Результаты

| Модель | L2 Error | Время | Framework |
|--------|----------|-------|-----------|
| **PINN** | 0.00323 (0.32%) | 575 сек | TensorFlow |
| **DeepONet** | 0.0062 (0.62%) | 600 сек | TensorFlow |
| **FNO** | **0.0011 (0.11%)** | **120 сек** | PyTorch |

### Эксперимент со слоями PINN

| Архитектура | L2 Error |
|-------------|----------|
| 3×20 (исходная) | 9.5% |
| 3×50 | 7.7% |
| **4×50 (моя)** | **3.0%** |
| 4×100 | 26.6% |

**Вывод:** оптимальная архитектура — 4 скрытых слоя по 50 нейронов.

## Выводы

1. **FNO** — лучший по точности (0.11%) и скорости (120 сек)
2. **PINN** — надёжная физически информированная сеть (0.32%)
3. **DeepONet** — потенциал для операторного обучения (0.62%)

## Характеристики оборудования

- **Ноутбук:** MCLF-XX
- **Процессор:** Intel Core i5-12450H (12 ядер, до 4.4 ГГц)
- **ОЗУ:** 16 ГБ
- **GPU:** не использовался


## Автор

**Жувангараева Мадина Ермуратовна**  
Группа М80-105СВ-25, МАИ  
Преподаватель: Стрижак Сергей Владимирович

