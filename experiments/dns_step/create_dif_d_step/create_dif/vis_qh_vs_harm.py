#!/usr/bin/env python3
"""
vis_qh_vs_harm.py — Сравнение harmonic vs QH mapping для 5 примеров.

Для каждой .msh:
  [1] Harmonic отображение (ξ, η) — scatter узлов
  [2] QH отображение (ξ, η) — scatter узлов
  [3] 128×128 → Physical (HARMONIC) — scatter, цвет=ξ
  [4] 128×128 → Physical (QH) — scatter, цвет=ξ

Выход: vis/qh_vs_harm_5samples.png
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))
from read_msh import read_msh_file
from interpolate import GridInterpolator
from quasi_harmonic import QuasiHarmonicMapper

# Параметры
MSH_DIR = os.path.join(os.path.dirname(__file__), "../test_domains")
RESOLUTION = 128
QH_MODE = 'invsqrt'
QH_EPERP = 'phys'
QH_DAMPING = 0.3
QH_NITER = 5
N_SAMPLES = 5

# Цвета границ
BND_COLORS = {'inlet': 'red', 'outlet': 'blue',
              'bottom': 'green', 'top': 'magenta'}
BND_LABELS = {'inlet': 'inlet (ξ=0)', 'outlet': 'outlet (ξ=1)',
              'bottom': 'bottom (η=0)', 'top': 'top (η=1)'}


def interpolate_grid(mapper, resolution):
    """Барицентрическая интерполяция: регулярная сетка → физическое пространство."""
    interp = GridInterpolator(mapper)
    Mx, My = interp.interpolate(sampling_size=resolution)

    # NaN fill
    if np.isnan(Mx).any():
        from scipy.interpolate import NearestNDInterpolator
        mask = ~np.isnan(Mx)
        coords = np.argwhere(mask)
        fx = NearestNDInterpolator(coords, Mx[mask])
        fy = NearestNDInterpolator(coords, My[mask])
        for r, c in np.argwhere(np.isnan(Mx)):
            Mx[r, c] = fx(r, c)
            My[r, c] = fy(r, c)

    return Mx, My


def process_one(msh_path):
    """Возвращает (Y_h, Y_q, Mx_h, My_h, Mx_q, My_q, X, F, bnds)."""
    X, F, bnds = read_msh_file(msh_path)
    if X is None:
        return (None,) * 9

    # Harmonic
    m_h = QuasiHarmonicMapper(X, F, bnds, n_iter=0)
    Y_h = m_h.build_mapping(verbose=False)
    m_h.Y = Y_h; m_h.X = X; m_h.F = F
    Mx_h, My_h = interpolate_grid(m_h, RESOLUTION)

    # QH
    m_q = QuasiHarmonicMapper(X, F, bnds, n_iter=QH_NITER,
                              mode=QH_MODE, eperp_mode=QH_EPERP,
                              damping=QH_DAMPING)
    Y_q = m_q.build_mapping(Y_init=Y_h, verbose=False)

    if np.any(np.isnan(Y_q)):
        Mx_q, My_q = Mx_h, My_h  # fallback
    else:
        m_q.Y = Y_q; m_q.X = X; m_q.F = F
        Mx_q, My_q = interpolate_grid(m_q, RESOLUTION)

    return Y_h, Y_q, Mx_h, My_h, Mx_q, My_q, X, F, bnds


def plot_boundaries(ax, Y, bnds):
    """Нанести границы на scatter plot в (ξ,η)."""
    for name, color in BND_COLORS.items():
        ids = bnds[name]
        if ids:
            ax.scatter(Y[ids, 0], Y[ids, 1], c=color, s=6,
                       edgecolors='k', linewidths=0.2, zorder=5,
                       label=BND_LABELS[name])


def main():
    files = sorted(
        [f for f in os.listdir(MSH_DIR) if f.endswith(".msh") and "step" in f],
        key=lambda x: int(x.split('_')[1].split('.')[0]) if '_' in x else x
    )[:N_SAMPLES]

    if not files:
        print(f"No .msh files in {MSH_DIR}")
        sys.exit(1)

    print(f"Загружаю {len(files)} примеров...")
    results = []
    for fname in files:
        path = os.path.join(MSH_DIR, fname)
        print(f"  {fname}...", end=" ", flush=True)
        res = process_one(path)
        if res[0] is not None:
            results.append((fname, *res))
            print("OK")
        else:
            print("FAIL")

    if not results:
        print("Ничего не обработано.")
        sys.exit(1)

    n = len(results)

    # Фигура: n строк × 4 столбца
    fig = plt.figure(figsize=(24, 4.6 * n))
    gs = GridSpec(n, 4, figure=fig, hspace=0.4, wspace=0.35)

    for row, (fname, Y_h, Y_q, Mx_h, My_h, Mx_q, My_q, X, F, bnds) \
             in enumerate(results):
        short_name = fname.replace('.msh', '')

        # Col 0: Harmonic (ξ,η)
        ax = fig.add_subplot(gs[row, 0])
        ax.scatter(Y_h[:, 0], Y_h[:, 1], s=2, c='gray', alpha=0.4, edgecolors='none')
        plot_boundaries(ax, Y_h, bnds)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.set_title(f"Harmonic — {short_name}", fontsize=9)
        ax.grid(True, alpha=0.2)

        # Col 1: QH (ξ,η)
        ax = fig.add_subplot(gs[row, 1])
        ax.scatter(Y_q[:, 0], Y_q[:, 1], s=2, c='gray', alpha=0.4, edgecolors='none')
        plot_boundaries(ax, Y_q, bnds)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.set_title(f"QH (invsqrt+phys) — {short_name}", fontsize=9)
        ax.grid(True, alpha=0.2)

        # Col 2: Harmonic → Physical
        ax = fig.add_subplot(gs[row, 2])
        ax.scatter(Mx_h.ravel(), My_h.ravel(), c='k', s=1, alpha=0.5, edgecolors='none')
        for name, color in BND_COLORS.items():
            ids = bnds[name]
            if ids:
                ax.scatter(X[ids, 0], X[ids, 1], c=color, s=4,
                           edgecolors='k', linewidths=0.2, zorder=5)
        ax.set_aspect('equal')
        ax.set_title(f"Harmonic → Physical", fontsize=10)

        # Col 3: QH → Physical
        ax = fig.add_subplot(gs[row, 3])
        ax.scatter(Mx_q.ravel(), My_q.ravel(), c='k', s=1, alpha=0.5, edgecolors='none')
        for name, color in BND_COLORS.items():
            ids = bnds[name]
            if ids:
                ax.scatter(X[ids, 0], X[ids, 1], c=color, s=4,
                           edgecolors='k', linewidths=0.2, zorder=5)
        ax.set_aspect('equal')
        ax.set_title(f"QH → Physical", fontsize=10)

    # Подписи колонок
    fig.text(0.14, 0.96, "Harmonic (ξ, η)", ha='center', fontsize=11, fontweight='bold')
    fig.text(0.37, 0.96, "QH (ξ, η)", ha='center', fontsize=11, fontweight='bold')
    fig.text(0.61, 0.96, "Harmonic → Physical", ha='center', fontsize=11, fontweight='bold')
    fig.text(0.86, 0.96, "QH → Physical", ha='center', fontsize=11, fontweight='bold')

    out_dir = os.path.join(os.path.dirname(__file__), "vis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "qh_vs_harm_5samples.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()
