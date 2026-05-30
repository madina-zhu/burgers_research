"""
Эксперимент: сравнение FCNN с разным количеством слоёв
"""

import os
os.environ["DDE_BACKEND"] = "tensorflow.compat.v1"

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# Фиксируем seed
np.random.seed(42)
dde.config.set_random_seed(42)

NU = 0.01 / np.pi

# =====================================================
# ОБЛАСТЬ, УСЛОВИЯ, PDE (как в основном коде)
# =====================================================

def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - NU * dy_xx

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)

data = dde.data.TimePDE(
    geomtime, pde, [bc, ic],
    num_domain=2540, num_boundary=80, num_initial=160, num_test=1000,
)

# =====================================================
# ЭКСПЕРИМЕНТ: разные архитектуры
# =====================================================

configs = [
    {"name": "3x20", "layers": [2] + [20]*3 + [1]},
    {"name": "3x50", "layers": [2] + [50]*3 + [1]},
    {"name": "4x50", "layers": [2] + [50]*4 + [1]},  # мой улучшенный вариант
    {"name": "4x100", "layers": [2] + [100]*4 + [1]},
]

results = []

for cfg in configs:
    print(f"\n=== Training {cfg['name']} ===")
    
    net = dde.nn.FNN(cfg["layers"], activation="tanh", kernel_initializer="Glorot normal")
    model = dde.Model(data, net)
    
    model.compile("adam", lr=1e-3)
    model.train(iterations=5000, display_every=1000)  # уменьшил для скорости
    
    # Загружаем тестовые данные
    def gen_testdata():
        data_np = np.load("dataset/Burgers.npz")
        t = data_np["t"]
        x = data_np["x"]
        exact = data_np["usol"].T
        xx, tt = np.meshgrid(x, t)
        X = np.vstack((np.ravel(xx), np.ravel(tt))).T
        y = exact.flatten()[:, None]
        return X, y
    
    X_test, y_true = gen_testdata()
    y_pred = model.predict(X_test)
    
    l2_error = dde.metrics.l2_relative_error(y_true, y_pred)
    results.append({"name": cfg["name"], "layers": cfg["layers"], "l2_error": l2_error})
    
    print(f"{cfg['name']} - L2 error: {l2_error:.6f}")

# =====================================================
# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТА
# =====================================================

names = [r["name"] for r in results]
errors = [r["l2_error"] for r in results]

plt.figure(figsize=(10, 6))
bars = plt.bar(names, errors, color=['blue', 'green', 'red', 'purple'])
plt.ylabel("L2 relative error")
plt.title("Сравнение FCNN с разным количеством слоёв")
plt.yscale("log")

# Подписываем значения на столбцах
for bar, err in zip(bars, errors):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
             f'{err:.5f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("results/experiment_layers.png", dpi=300)
plt.close()

print("\n=== РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА ===")
for r in results:
    print(f"{r['name']}: {r['layers']} -> L2 error = {r['l2_error']:.6f}")
print("\nСохранено: results/experiment_layers.png")