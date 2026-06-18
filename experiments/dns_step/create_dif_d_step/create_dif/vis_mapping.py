"""
Визуализация гармонического отображения: физический vs логический домен
с раскраской треугольников по искажению.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import cm

from read_msh import read_msh_file
from quasi_harmonic import QuasiHarmonicMapper, compute_distortions

# Берём первую сетку
test_dir = os.path.join(os.path.dirname(__file__), "../test_domains")
files = sorted([f for f in os.listdir(test_dir) if f.endswith(".msh")])
if not files:
    print("Нет .msh файлов!")
    sys.exit(1)

path = os.path.join(test_dir, files[0])
X, F, bnds = read_msh_file(path)
print(f"Файл: {files[0]}")
print(f"Узлов: {len(X)}, треугольников: {len(F)}")

# Строим гармоническое отображение
mapper = QuasiHarmonicMapper(X, F, bnds, n_iter=0)
Y = mapper.build_mapping(verbose=False)

# Искажения
angle_dist, area_dist = compute_distortions(X, Y, F)

print(f"Angle distortion: mean={angle_dist.mean():.3f}, max={angle_dist.max():.3f}")
print(f"Area distortion:  mean={area_dist.mean():.3f}, max={area_dist.max():.3f}")

# Собираем треугольники для PolyCollection
phys_polys = [X[f] for f in F]
log_polys = [Y[f] for f in F]

# Нормализация для цветов
norm_angle = plt.Normalize(vmin=1.0, vmax=min(angle_dist.max(), 10.0))
norm_area = plt.Normalize(vmin=0.0, vmax=min(area_dist.max(), 5.0))

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# === 1. Физический домен (угловое искажение) ===
ax = axes[0, 0]
coll = PolyCollection(phys_polys, array=angle_dist, cmap='viridis', 
                      norm=norm_angle, edgecolors='none', alpha=0.9)
ax.add_collection(coll)
ax.plot(*zip(*[X[b] for b in bnds['inlet']]), 'ro', ms=2, label='inlet')
ax.plot(*zip(*[X[b] for b in bnds['outlet']]), 'bo', ms=2, label='outlet')
ax.plot(*zip(*[X[b] for b in bnds['bottom']]), 'g,', ms=1, alpha=0.3)
ax.plot(*zip(*[X[b] for b in bnds['top']]), 'm,', ms=1, alpha=0.3)
ax.set_aspect('equal')
ax.set_title(f"Physical Domain (angle distortion)\nmax={angle_dist.max():.2f}, mean={angle_dist.mean():.2f}")
ax.legend(loc='upper right', markerscale=3)
plt.colorbar(coll, ax=ax, label='angle distortion')

# === 2. Логический домен (угловое искажение) ===
ax = axes[0, 1]
coll = PolyCollection(log_polys, array=angle_dist, cmap='viridis',
                      norm=norm_angle, edgecolors='none', alpha=0.9)
ax.add_collection(coll)
ax.plot(*zip(*[Y[b] for b in bnds['inlet']]), 'ro', ms=2, label='inlet → ξ=0')
ax.plot(*zip(*[Y[b] for b in bnds['outlet']]), 'bo', ms=2, label='outlet → ξ=1')
ax.plot(*zip(*[Y[b] for b in bnds['bottom']]), 'g,', ms=1, alpha=0.3, label='bottom → η=0')
ax.plot(*zip(*[Y[b] for b in bnds['top']]), 'm,', ms=1, alpha=0.3, label='top → η=1')
ax.set_aspect('equal')
ax.set_title(f"Universal Domain (ξ, η)\nboundary-preserving harmonic map")
ax.legend(loc='upper right', markerscale=3)
plt.colorbar(coll, ax=ax, label='angle distortion')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)

# === 3. Физический домен (площадное искажение) ===
ax = axes[1, 0]
coll = PolyCollection(phys_polys, array=area_dist, cmap='plasma',
                      norm=norm_area, edgecolors='none', alpha=0.9)
ax.add_collection(coll)
ax.set_aspect('equal')
ax.set_title(f"Physical Domain (area distortion)\nmax={area_dist.max():.2f}, mean={area_dist.mean():.2f}")
plt.colorbar(coll, ax=ax, label='area distortion')

# === 4. Логический домен (площадное искажение) ===
ax = axes[1, 1]
coll = PolyCollection(log_polys, array=area_dist, cmap='plasma',
                      norm=norm_area, edgecolors='none', alpha=0.9)
ax.add_collection(coll)
ax.set_aspect('equal')
ax.set_title(f"Universal Domain (area distortion)")
plt.colorbar(coll, ax=ax, label='area distortion')

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "harmonic_mapping_check.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nСохранено: {out_path}")
