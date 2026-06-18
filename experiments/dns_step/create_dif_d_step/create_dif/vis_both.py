"""
Сравнение гармонического (cotangent) vs quasi-harmonic на одной сетке.
2×3 панели: физический домен, универсальный домен, гистограммы искажений.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import cm

from read_msh import read_msh_file
from quasi_harmonic import QuasiHarmonicMapper, compute_distortions

# 1. Загрузка
test_dir = os.path.join(os.path.dirname(__file__), "../test_domains")
files = sorted([f for f in os.listdir(test_dir) if f.endswith(".msh")])
if not files:
    print("Нет .msh файлов!")
    sys.exit(1)

path = os.path.join(test_dir, files[0])
X, F, bnds = read_msh_file(path)
name = files[0].replace('.msh', '')
print(f"Файл: {files[0]}, узлов: {len(X)}, треугольников: {len(F)}")

# 2. Гармоническое
mapper_h = QuasiHarmonicMapper(X, F, bnds, n_iter=0)
Y_h = mapper_h.build_mapping(verbose=False)
angle_h, area_h = compute_distortions(X, Y_h, F)
stats_h = f"угол: mean={angle_h.mean():.2f}, max={angle_h.max():.2f}\nплощадь: mean={area_h.mean():.2f}, max={area_h.max():.2f}"

# 3. Quasi-harmonic (осторожно)
# Используем paper formulation C = sqrtm(JTJ), relax=0.5, всего 3 итерации
print("\nСтроим QH (mode='sqrt', n_iter=15)...")
mapper_q = QuasiHarmonicMapper(X, F, bnds, n_iter=15, mode='sqrt')
Y_q = mapper_q.build_mapping(Y_init=Y_h, verbose=True)
angle_q, area_q = compute_distortions(X, Y_q, F)
stats_q = f"угол: mean={angle_q.mean():.2f}, max={angle_q.max():.2f}\nплощадь: mean={area_q.mean():.2f}, max={area_q.max():.2f}"

# 4. Масштабируем для subplots
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 4)

# Общие настройки
norm_angle = plt.Normalize(vmin=1.0, vmax=10.0)
norm_area = plt.Normalize(vmin=0.0, vmax=5.0)

# ----------------------- ROW 1: Physical Domain -----------------------
ax = fig.add_subplot(gs[0, 0])
coll = PolyCollection([X[f] for f in F], array=angle_h, cmap='viridis',
                      norm=norm_angle, edgecolors='none', alpha=0.9)
ax.add_collection(coll); ax.set_aspect('equal')
ax.set_title(f"Harmonic — physical (angle)\n{stats_h}", fontsize=9)

ax = fig.add_subplot(gs[0, 1])
coll = PolyCollection([Y_h[f] for f in F], array=angle_h, cmap='viridis',
                      norm=norm_angle, edgecolors='none', alpha=0.9)
ax.add_collection(coll); ax.set_aspect('equal')
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
ax.set_title(f"Harmonic — universal domain", fontsize=9)

ax = fig.add_subplot(gs[0, 2])
coll = PolyCollection([X[f] for f in F], array=angle_q, cmap='viridis',
                      norm=norm_angle, edgecolors='none', alpha=0.9)
ax.add_collection(coll); ax.set_aspect('equal')
ax.set_title(f"QH — physical (angle)\n{stats_q}", fontsize=9)

ax = fig.add_subplot(gs[0, 3])
coll = PolyCollection([Y_q[f] for f in F], array=angle_q, cmap='viridis',
                      norm=norm_angle, edgecolors='none', alpha=0.9)
ax.add_collection(coll); ax.set_aspect('equal')
ax.set_title(f"QH — universal domain", fontsize=9)
plt.colorbar(coll, ax=ax, label='angle distortion', shrink=0.6)

# ----------------------- ROW 2: Area + histograms -----------------------
ax = fig.add_subplot(gs[1, 0])
coll = PolyCollection([X[f] for f in F], array=area_h, cmap='plasma',
                      norm=norm_area, edgecolors='none', alpha=0.9)
ax.add_collection(coll); ax.set_aspect('equal')
ax.set_title(f"Harmonic — area", fontsize=9)

ax = fig.add_subplot(gs[1, 1])
bins = np.linspace(1, angle_h.max(), 50)
ax.hist(angle_h, bins=bins, alpha=0.6, label='harmonic', color='tab:blue', density=True)
if angle_q.max() < 100:
    ax.hist(angle_q, bins=bins, alpha=0.6, label='QH', color='tab:orange', density=True)
ax.set_xlabel('angle distortion'); ax.set_ylabel('density')
ax.legend(); ax.set_title("Angle distortion histogram", fontsize=9)

ax = fig.add_subplot(gs[1, 2])
coll = PolyCollection([X[f] for f in F], array=area_q, cmap='plasma',
                      norm=norm_area, edgecolors='none', alpha=0.9)
ax.add_collection(coll); ax.set_aspect('equal')
ax.set_title(f"QH — area", fontsize=9)

ax = fig.add_subplot(gs[1, 3])
bins = np.linspace(0, min(area_h.max(), area_q.max(), 20), 50)
ax.hist(area_h, bins=bins, alpha=0.6, label='harmonic', color='tab:blue', density=True)
ax.hist(area_q, bins=bins, alpha=0.6, label='QH', color='tab:orange', density=True)
ax.set_xlabel('area distortion'); ax.set_ylabel('density')
ax.legend(); ax.set_title("Area distortion histogram", fontsize=9)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "harmonic_vs_QH.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nСохранено: {out_path}")
print(f"\nHARMONIC:  {stats_h}")
print(f"QH:        {stats_q}")
