"""
Визуализация датасета Burgers.npz
"""

import numpy as np
import matplotlib.pyplot as plt

# Загружаем датасет
data = np.load("dataset/Burgers.npz")

t = data["t"]      # время (100, 1)
x = data["x"]      # пространство (256, 1)
usol = data["usol"] # решение u(x,t) (256, 100)

print(f"Размерность: t={t.shape}, x={x.shape}, usol={usol.shape}")

# Преобразуем в одномерные массивы для удобства
x_flat = x.flatten()
t_flat = t.flatten()

# =====================================================
# 1. Начальное условие u(x,0) - берём первый столбец usol (t=0)
# =====================================================
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(x_flat, usol[:, 0], 'b-', linewidth=2)
plt.xlabel("x")
plt.ylabel("u(x,0)")
plt.title("Начальное условие: u(x,0) = -sin(πx)")
plt.grid(True)

# =====================================================
# 2. Решение в разные моменты времени
# =====================================================
plt.subplot(2, 2, 2)

# Индексы для разных моментов времени
time_indices = [0, 25, 50, 75, 99]  # примерно t = 0, 0.25, 0.5, 0.75, 0.99
time_labels = [0, 0.25, 0.5, 0.75, 0.99]

for idx, label in zip(time_indices, time_labels):
    plt.plot(x_flat, usol[:, idx], label=f't={label}')

plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Решение в разные моменты времени")
plt.legend()
plt.grid(True)

# =====================================================
# 3. Тепловая карта (2D)
# =====================================================
plt.subplot(2, 2, (3, 4))
Xg, Tg = np.meshgrid(x_flat, t_flat)
# Транспонируем usol для тепловой карты (t по вертикали, x по горизонтали)
plt.pcolormesh(Xg, Tg, usol.T, shading='auto', cmap='viridis')
plt.colorbar(label='u(x,t)')
plt.xlabel("x")
plt.ylabel("t")
plt.title("Тепловая карта решения u(x,t)")

plt.tight_layout()
plt.savefig("results/dataset_visualization.png", dpi=300)
plt.close()

print("Сохранено: results/dataset_visualization.png")