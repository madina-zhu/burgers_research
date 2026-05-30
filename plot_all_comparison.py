"""
Общий график сравнения: True, FCNN, DeepONet, FNO
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Пути к файлам
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "results")

# =====================================================
# ЗАГРУЗКА ДАННЫХ
# =====================================================

# FCNN
fcnn_file = os.path.join(results_dir, "FCNN_predictions.dat")
fcnn_data = np.loadtxt(fcnn_file)
u_true_fcnn = fcnn_data[:, 2]
u_pred_fcnn = fcnn_data[:, 3]

# DeepONet
deep_file = os.path.join(results_dir, "deeponet_predictions.dat")
deep_data = np.loadtxt(deep_file)
u_true_deep = deep_data[:, 2]
u_pred_deep = deep_data[:, 3]

# FNO
fno_file = os.path.join(results_dir, "fno_predictions.npz")
fno_data = np.load(fno_file)
u_pred_fno = fno_data["prediction"].flatten()[:1000]
u_true_fno = fno_data["target"].flatten()[:1000]

# =====================================================
# ПОСТРОЕНИЕ ОБЩЕГО ГРАФИКА
# =====================================================

plt.figure(figsize=(12, 6))

n_plot = 1000
x_axis = np.arange(n_plot)

# True
plt.plot(x_axis, u_true_fcnn[:n_plot], 'k-', linewidth=2, label='True')

# FCNN
plt.plot(x_axis, u_pred_fcnn[:n_plot], 'b--', linewidth=1.5, label='FCNN')

# DeepONet
plt.plot(x_axis, u_pred_deep[:n_plot], 'g--', linewidth=1.5, label='DeepONet')

# FNO
plt.plot(x_axis, u_pred_fno[:n_plot], 'r--', linewidth=1.5, label='FNO')

plt.xlabel("Point index")
plt.ylabel("u(x,t)")
plt.title("Сравнение всех трёх архитектур с точным решением")
plt.legend()
plt.grid(True)
plt.tight_layout()

output_file = os.path.join(results_dir, "all_models_comparison.png")
plt.savefig(output_file, dpi=300)
plt.close()

print(f"Сохранено: {output_file}")