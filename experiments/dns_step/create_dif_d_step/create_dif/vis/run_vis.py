#!/usr/bin/env python3
"""
Визуализация: 6 комбинаций QH × 2 режима damping × 3 среза итераций.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from read_msh import read_msh_file
from run_compare_maps import (
    harmonic_map, quasi_harmonic_map, build_qh_weights,
    compute_distortions, distortions_table, count_flipped,
    solve_dirichlet, build_cotangent_laplacian
)
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

VIS_DIR = os.path.dirname(__file__)
MSH_PATH = os.path.join(VIS_DIR, "../../test_domains/step_1.msh")

# Загрузка
print(f"Загружаем: {MSH_PATH}")
X, F, bnds = read_msh_file(MSH_PATH)
print(f"Узлов: {X.shape[0]}, треугольников: {F.shape[0]}")

# Гармоническое отображение
print("\nГармоническое...")
Y_harm = harmonic_map(X, F, bnds)
nf_harm = count_flipped(Y_harm, F)
angle_h, area_h = compute_distortions(X, Y_harm, F)
s_harm = distortions_table(angle_h, area_h)

# Параметры
combos = [
    ('sqrt',   'log',  'sqrt + log'),
    ('sqrt',   'phys', 'sqrt + phys'),
    ('invsqrt','log',  'invsqrt + log'),
    ('invsqrt','phys', 'invsqrt + phys'),
    ('inv',    'log',  'inv + log'),
    ('inv',    'phys', 'inv + phys'),
]
dampings = [(1.0, 'nodamping'), (0.3, 'damping')]
n_iters = [0, 5, 10]
COLORS = plt.cm.viridis(np.linspace(0.2, 0.9, 6))

def per_vertex_area(Y):
    """Площадное искажение per-vertex (минимум по инцидентным треугольникам)."""
    dist = np.ones(X.shape[0])
    _, area_d = compute_distortions(X, Y, F)
    for t in range(F.shape[0]):
        for v in F[t]:
            dist[v] = min(dist[v], area_d[t])
    return dist

# Запуск QH для всех комбинаций
all_results = {}  # {(C_mode, eperp, damping): {0: Y0, 5: Y5, 10: Y10, stats: {...}}}
all_nf = {}       # flipped counts
all_times = {}

for C_mode, eperp_mode, label in combos:
    for damping, damp_name in dampings:
        key = (C_mode, eperp_mode, damping)
        print(f"\n{'='*50}")
        print(f"  {label}, damping={damping}")
        print(f"{'='*50}")

        Y = Y_harm.copy()
        results_key = {0: Y_harm.copy()}

        for it in range(1, 11):
            W = build_qh_weights(X, F, Y, C_mode=C_mode, eperp_mode=eperp_mode)
            deg = np.array(W.sum(axis=1)).flatten()
            L = (diags(deg) - W).tocsr()

            fix_xi_nodes = list(bnds['inlet']) + list(bnds['outlet'])
            fix_xi_vals  = [0.0]*len(bnds['inlet']) + [1.0]*len(bnds['outlet'])
            fix_eta_nodes = list(bnds['bottom']) + list(bnds['top'])
            fix_eta_vals  = [0.0]*len(bnds['bottom']) + [1.0]*len(bnds['top'])

            xi  = solve_dirichlet(L, X.shape[0], fix_xi_nodes, fix_xi_vals)
            eta = solve_dirichlet(L, X.shape[0], fix_eta_nodes, fix_eta_vals)

            Y_new = np.column_stack([xi, eta])
            if np.any(np.isnan(Y_new)) or np.any(np.isinf(Y_new)):
                print(f"  NaN на итерации {it} — стоп")
                break

            diff = np.max(np.abs(Y_new - Y))
            Y = damping * Y_new + (1 - damping) * Y
            print(f"  iter {it}: Δ={diff:.2e} flipped={count_flipped(Y, F)}")

            if it in n_iters:
                results_key[it] = Y.copy()

        # Сохраняем
        t0 = time.time()
        angle_q, area_q = compute_distortions(X, Y, F)
        all_results[key] = results_key
        all_nf[key] = count_flipped(Y, F)
        s_q = distortions_table(angle_q, area_q)
        all_results[key]['stats'] = s_q

# Отрисовка: 6 комбинаций × 2 damping, одна картинка на комбинацию
print("\n\nОтрисовка картинок...")

for C_mode, eperp_mode, label in combos:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(f"{label}", fontsize=14, fontweight='bold')

    for row, (damping, damp_name) in enumerate(dampings):
        key = (C_mode, eperp_mode, damping)

        # Строка: harmonic, 0 it, 5 it, 10 it
        titles = ['Harmonic', '0 it (harm)', '5 it', '10 it']
        Ys = [Y_harm, all_results[key].get(0, Y_harm),
              all_results[key].get(5, Y_harm), all_results[key].get(10, Y_harm)]

        for col in range(4):
            ax = axes[row, col]
            Y = Ys[col]

            ax.scatter(Y[:, 0], Y[:, 1], s=3, c='k', alpha=0.5, edgecolors='none')

            for bn, cl in [('inlet','r'),('outlet','b'),
                           ('bottom','g'),('top','m')]:
                ids = bnds[bn]
                if ids:
                    ax.scatter(Y[ids, 0], Y[ids, 1], c=cl, s=8,
                               edgecolors='k', linewidths=0.3, zorder=5)

            nf = count_flipped(Y, F)
            ax.set_title(f"{titles[col]} (fl={nf})", fontsize=9)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)

    # Подписи строк
    axes[0, 0].set_ylabel("no damping", fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel("damping α=0.3", fontsize=11, fontweight='bold')

    plt.tight_layout()
    fname = f"{C_mode}_{eperp_mode}.png"
    out_path = os.path.join(VIS_DIR, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {fname}")

# Сводная статистика (одна картинка)
print("\nОтрисовка статистики...")

fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('off')

# Заголовок
ax.text(0.5, 0.97, "Сравнение 6 комбинаций Quasi-Harmonic Mapping",
        ha='center', va='top', fontsize=14, fontweight='bold',
        transform=ax.transAxes)

# Столбцы
cols_def = [
    ('Метрика', 0.02, 0.80),
]
col_x = 0.02
headers = ['Метрика'] + [c[2] for c in combos] + [c[2] + ' (damping)' for c in combos]
col_widths = [0.12] + [0.115]*6 + [0.115]*6

# Таблица: каждое значение в процентах
y_start = 0.88
row_h = 0.045

# Заголовок таблицы
x_pos = col_x
for hdr, w in zip(headers, col_widths):
    ax.text(x_pos, y_start, hdr, fontsize=7, fontweight='bold',
            va='bottom', ha='left', transform=ax.transAxes)
    x_pos += w

# Данные
metrics = [
    ('angle_max',  'max angle'),
    ('angle_mean', 'mean angle'),
    ('angle_median', 'med angle'),
    ('area_max',   'max area'),
    ('area_median','med area'),
]

for row_idx, (key, display_name) in enumerate(metrics):
    y = y_start - (row_idx + 1) * row_h - 0.02
    ax.text(col_x, y, display_name, fontsize=8, fontweight='bold',
            va='top', ha='left', transform=ax.transAxes)
    x_pos = col_x + col_widths[0]

    for col_idx in range(12):
        combo_idx = col_idx % 6
        is_damping = col_idx >= 6
        damping_val = 0.3 if is_damping else 1.0
        cm, ep, _ = combos[combo_idx]
        key_t = (cm, ep, damping_val)
        result_key = all_results[key_t]
        
        if key in result_key.get('stats', {}):
            val = result_key['stats'][key]
        else:
            val = 0.0

        # Для искажений — показываем % изменения относительно harmonic
        if key in s_harm:
            h_val = s_harm[key]
            chg = (val - h_val) / max(h_val, 1e-15) * 100

        color = 'green' if chg < -5 else ('red' if chg > 5 else 'gray')
        ax.text(x_pos, y, f"{val:.2f} ({chg:+.1f}%)", fontsize=6,
                color=color, va='top', ha='left', transform=ax.transAxes)
        x_pos += col_widths[col_idx + 1]

    x_pos = col_x + col_widths[0]

# flipped
y = y_start - (len(metrics) + 1) * row_h - 0.02
ax.text(col_x, y, 'flipped', fontsize=8, fontweight='bold',
        va='top', ha='left', transform=ax.transAxes)
x_pos = col_x + col_widths[0]
for col_idx in range(12):
    combo_idx = col_idx % 6
    is_damping = col_idx >= 6
    damping_val = 0.3 if is_damping else 1.0
    cm, ep, _ = combos[combo_idx]
    nf = all_nf.get((cm, ep, damping_val), 0)
    ax.text(x_pos, y, str(nf), fontsize=7, color='red' if nf > 0 else 'green',
            va='top', ha='left', transform=ax.transAxes)
    x_pos += col_widths[col_idx + 1]

# Легенда цветов для change
legend_y = 0.1
ax.text(0.02, legend_y, "Цвет:  green = улучшение >5%  |  gray = <5%  |  red = ухудшение >5%",
        fontsize=9, transform=ax.transAxes)

# Информация о сетке
ax.text(0.02, 0.05, f"Сетка: step_1.msh, {X.shape[0]} узлов, {F.shape[0]} треугольников",
        fontsize=8, color='gray', transform=ax.transAxes)
ax.text(0.02, 0.02, f"Гармоническое: angle_mean={s_harm['angle_mean']:.2f}, area_max={s_harm['area_max']:.1f}",
        fontsize=8, color='gray', transform=ax.transAxes)

out_path = os.path.join(VIS_DIR, "statistics.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ statistics.png")

# Сводка сходимости (Δ по итерациям)
# Перезапускаем с логгированием Δ
print("\nОтрисовка сходимости...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for idx, (C_mode, eperp_mode, label) in enumerate(combos):
    ax = axes[idx]

    for damping, damp_name, style in [(1.0, 'no damping', 'o-'), (0.3, 'α=0.3', 's--')]:
        Y = Y_harm.copy()
        deltas = []
        key = (C_mode, eperp_mode, damping)

        for it in range(1, 11):
            W = build_qh_weights(X, F, Y, C_mode=C_mode, eperp_mode=eperp_mode)
            deg = np.array(W.sum(axis=1)).flatten()
            L = (diags(deg) - W).tocsr()

            fix_xi_nodes = list(bnds['inlet']) + list(bnds['outlet'])
            fix_xi_vals  = [0.0]*len(bnds['inlet']) + [1.0]*len(bnds['outlet'])
            fix_eta_nodes = list(bnds['bottom']) + list(bnds['top'])
            fix_eta_vals  = [0.0]*len(bnds['bottom']) + [1.0]*len(bnds['top'])

            xi  = solve_dirichlet(L, X.shape[0], fix_xi_nodes, fix_xi_vals)
            eta = solve_dirichlet(L, X.shape[0], fix_eta_nodes, fix_eta_vals)
            Y_new = np.column_stack([xi, eta])

            if np.any(np.isnan(Y_new)) or np.any(np.isinf(Y_new)):
                break

            diff = np.max(np.abs(Y_new - Y))
            deltas.append(diff)
            Y = damping * Y_new + (1 - damping) * Y

        ax.semilogy(range(1, len(deltas)+1), deltas, style,
                    label=damp_name, markersize=4)

    ax.set_title(label, fontsize=9)
    ax.set_xlabel('iter')
    ax.set_ylabel('Δ')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=7)

plt.suptitle("Сходимость: Δ (max change) по итерациям", fontsize=13)
plt.tight_layout()
out_path = os.path.join(VIS_DIR, "convergence.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ convergence.png")

# Барчарты: angle_mean и area_max для всех 12 вариантов
print("\nОтрисовка барчартов...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

labels = []
angle_means = []
area_maxs = []
colors_bars = []

for damping, damp_name in dampings:
    for cm, ep, label in combos:
        short = f"{cm[:4]}+{ep[:2]}" + ("\ndamp" if damping < 1 else "\nraw")
        labels.append(short)
        key = (cm, ep, damping)
        s = all_results[key].get('stats', {})
        angle_means.append(s.get('angle_mean', 0))
        area_maxs.append(s.get('area_max', 0))

bar_colors = np.repeat(COLORS, 2, axis=0)
ax = axes[0]
bars = ax.bar(range(12), angle_means, color=bar_colors)
ax.axhline(s_harm['angle_mean'], color='red', ls='--', lw=1, label=f"Harmonic: {s_harm['angle_mean']:.2f}")
ax.set_xticks(range(12))
ax.set_xticklabels(labels, fontsize=6, rotation=20)
ax.set_ylabel('angle mean')
ax.set_title('Угловое искажение (cond(J))')
ax.legend(fontsize=7)
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
bars = ax.bar(range(12), area_maxs, color=bar_colors)
ax.axhline(s_harm['area_max'], color='red', ls='--', lw=1, label=f"Harmonic: {s_harm['area_max']:.1f}")
ax.set_xticks(range(12))
ax.set_xticklabels(labels, fontsize=6, rotation=20)
ax.set_ylabel('area max')
ax.set_title('Площадное искажение (area ratio)')
ax.legend(fontsize=7)
ax.grid(axis='y', alpha=0.3)

plt.suptitle("Сравнение 6 комбинаций × 2 режима damping", fontsize=13)
plt.tight_layout()
out_path = os.path.join(VIS_DIR, "bar_charts.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ bar_charts.png")

print(f"\nВсего сохранено: 6 (комбо) + 4 (статы, сходимость, барчарты) = 10 файлов в {VIS_DIR}/")
