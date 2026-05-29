"""
Визуализация аналитического решения уравнения Бюргерса
"""

import numpy as np
import matplotlib.pyplot as plt

NU = 0.01 / np.pi

# Параметры сетки
nx = 200
nt = 100

x = np.linspace(-1, 1, nx)
t = np.linspace(0, 0.99, nt)

# Аналитическое решение
def analytic_solution(x, t, nu):
    """Аналитическое решение уравнения Бюргерса"""
    return -np.exp(-np.pi**2 * nu * t) * np.sin(np.pi * x)

# =====================================================
# 1. Решение в разные моменты времени (2D графики)
# =====================================================
plt.figure(figsize=(12, 5))

for time in [0, 0.25, 0.5, 0.75, 0.99]:
    u = analytic_solution(x, time, NU)
    plt.plot(x, u, label=f't={time}')

plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Аналитическое решение в разные моменты времени")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/analytic_solution_2d.png", dpi=300)
plt.close()

# =====================================================
# 2. 3D поверхность аналитического решения
# =====================================================
Xg, Tg = np.meshgrid(x, t)
U_analytic = analytic_solution(Xg, Tg, NU)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xg, Tg, U_analytic, cmap='viridis', linewidth=0)

ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u(x,t)")
ax.set_title("Аналитическое решение уравнения Бюргерса (3D)")

plt.tight_layout()
plt.savefig("results/analytic_solution_3d.png", dpi=300)
plt.close()

# =====================================================
# 3. Сравнение с численным решением (для верификации)
# =====================================================
try:
    data = np.load("dataset/Burgers.npz")
    t_num = data["t"].flatten()
    x_num = data["x"].flatten()
    usol_num = data["usol"]
    
    # Берём срез при t=0.5
    idx_t = np.argmin(np.abs(t_num - 0.5))
    u_numerical = usol_num[:, idx_t]
    
    # Аналитическое при t=0.5 на той же сетке x
    u_analytic_at_t05 = analytic_solution(x_num, 0.5, NU)
    
    plt.figure(figsize=(10, 5))
    plt.plot(x_num, u_numerical, 'b-', linewidth=2, label='Численное решение (DeepXDE)')
    plt.plot(x_num, u_analytic_at_t05, 'r--', linewidth=2, label='Аналитическое решение')
    plt.xlabel("x")
    plt.ylabel("u(x,0.5)")
    plt.title("Сравнение аналитического и численного решения при t=0.5")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/analytic_vs_numerical.png", dpi=300)
    plt.close()
    
    print("Сравнение с численным решением выполнено")
    
except Exception as e:
    print(f"Ошибка: {e}")

print("\nСохранено:")
print(" - results/analytic_solution_2d.png")
print(" - results/analytic_solution_3d.png")
print(" - results/analytic_vs_numerical.png")