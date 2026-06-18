"""Диагностика: проверяем гармоническое отображение на первой сетке."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from read_msh import read_msh_file
from quasi_harmonic import QuasiHarmonicMapper, compute_distortions
import numpy as np

# Берём первую сетку
test_dir = os.path.join(os.path.dirname(__file__), "../test_domains")
files = sorted([f for f in os.listdir(test_dir) if f.endswith(".msh")])
if not files:
    print("Нет .msh!")
    sys.exit(1)

path = os.path.join(test_dir, files[0])
print(f"Файл: {files[0]}")
X, F, bnds = read_msh_file(path)
print(f"Узлов: {len(X)}, треугольников: {len(F)}")

# Строим только гармоническое отображение
mapper = QuasiHarmonicMapper(X, F, bnds, n_iter=0)
Y_harm = mapper.build_mapping(verbose=False)

# Проверяем: сколько треугольников с отрицательной площадью в логике?
n_flip = mapper._count_flipped(Y_harm)
print(f"Перевёрнутых треугольников: {n_flip} / {len(F)} ({100*n_flip/len(F):.1f}%)")

# Проверяем координаты
print(f"ξ range: [{Y_harm[:,0].min():.4f}, {Y_harm[:,0].max():.4f}]")
print(f"η range: [{Y_harm[:,1].min():.4f}, {Y_harm[:,1].max():.4f}]")

# Проверяем какие вершины дают переворот
# Найдём треугольники с отрицательной площадью
flipped_tris = []
for t in range(len(F)):
    a, b, c = F[t]
    log = Y_harm[[a, b, c]]
    A_signed = 0.5 * (
        log[1, 0] * (log[2, 1] - log[0, 1]) +
        log[2, 0] * (log[0, 1] - log[1, 1]) +
        log[0, 0] * (log[1, 1] - log[2, 1])
    )
    if A_signed < 0:
        flipped_tris.append(t)
        if len(flipped_tris) <= 5:
            print(f"\n  Перевёрнутый треугольник #{t}: вершины {F[t]}")
            print(f"  Физические: {X[F[t]]}")
            print(f"  Логические: {Y_harm[F[t]]}")

# Смотрим на распределение площадей логических треугольников
log_areas = []
for t in range(len(F)):
    a, b, c = F[t]
    log = Y_harm[[a, b, c]]
    A_signed = 0.5 * (
        log[1, 0] * (log[2, 1] - log[0, 1]) +
        log[2, 0] * (log[0, 1] - log[1, 1]) +
        log[0, 0] * (log[1, 1] - log[2, 1])
    )
    log_areas.append(A_signed)

log_areas = np.array(log_areas)
print(f"\nПлощади в логическом пространстве:")
print(f"  min:  {log_areas.min():.8f}")
print(f"  max:  {log_areas.max():.6f}")
print(f"  сред: {log_areas.mean():.6f}")
print(f"  < 0:  {(log_areas < 0).sum()} / {len(F)}")
print(f"  ≈ 0:  {(np.abs(log_areas) < 1e-10).sum()} / {len(F)}")

# Проверяем котангенсные веса: есть ли отрицательные?
cot_laplacian = mapper._build_cotangent_laplacian()
n_neg = (cot_laplacian.data < -1e-12).sum()
n_pos = (cot_laplacian.data > 1e-12).sum()
print(f"\nКотангенсные веса: {n_neg} отриц., {n_pos} полож., всего {len(cot_laplacian.data)}")
