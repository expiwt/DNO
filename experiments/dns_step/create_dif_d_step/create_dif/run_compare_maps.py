#!/usr/bin/env python3
"""
run_compare_maps.py — Сравнение гармонического и квази-гармонического
отображения (Zayer, Rössl, Seidel 2005) с демпфированием.

Формула (12):
    w_ij = [x⊥_{j+1,j} · C · x⊥_{i,j+1}] / (4·A_j)
         + [x⊥_{j-1,i} · C · x⊥_{j,j-1}] / (4·A_{j-1})

    x⊥_{a,b} = CCW(p_b - p_a)  — ребро в параметрическом (ξ,η) пространстве
    C = inv(JᵀJ) или sqrtm(JᵀJ)  — тензор искажения
    A_j — площадь треугольника в параметрическом пространстве

На step-геометрии работает только C = inv(JᵀJ) с демпфированием α = 0.3.
"""
import sys, os, time
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import matplotlib.pyplot as pl

from scipy.sparse.linalg import bicgstab

sys.path.insert(0, os.path.dirname(__file__))
from read_msh import read_msh_file



# 1.  Solver
def solve_dirichlet(L, n_nodes, fixed_nodes, fixed_values):
    """Точное наложение BC: зануляем строку, diag=1, b=val."""
    A = L.tolil()
    b = np.zeros(n_nodes)
    for i, val in zip(fixed_nodes, fixed_values):
        A.data[i] = []
        A.rows[i] = []
        A[i, i] = 1.0
        b[i] = val
    A = A.tocsr()
    try:
        return spsolve(A, b)
    except Exception:
        u, info = bicgstab(A, b, atol=1e-12, maxiter=10000)
        return u if info >= 0 else np.zeros(n_nodes)


# 2.  Котангенсный Laplacian (гармоническая инициализация)
def build_cotangent_laplacian(X, F):
    n = X.shape[0]
    v0, v1, v2 = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]

    def cotan(u, v):
        dot = np.sum(u * v, axis=1)
        cross = np.maximum(np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]), 1e-12)
        return dot / cross

    cots = [cotan(v1 - v0, v2 - v0),
            cotan(v2 - v1, v0 - v1),
            cotan(v0 - v2, v1 - v2)]

    rows, cols, data = [], [], []
    for k, cot in enumerate(cots):
        i, j = F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        rows.extend(i); cols.extend(j); data.extend(cot * 0.5)
        rows.extend(j); cols.extend(i); data.extend(cot * 0.5)

    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


# 3.  Quasi-harmonic weights — формула (12)
def build_qh_weights(X, F, Y, C_mode='inv', eperp_mode='log'):
    """
    Строит весовую матрицу W по формуле (12) из статьи.

    Parameters
    C_mode : 'inv' | 'sqrt' | 'invsqrt'
        Тип тензора C.
    eperp_mode : 'log' | 'phys'
        'log'  — перпендикуляры x⊥ из параметрического пространства (ξ,η),
                 площадь — логическая. Стандартная формула (12).
        'phys' — перпендикуляры из физического пространства (x,y),
                 площадь — физическая. Альтернативный вариант.
    """
    n_nodes = X.shape[0]
    n_tri = F.shape[0]
    rows, cols, data = [], [], []

    # Предвычисляем физические площади
    phys_areas = np.zeros(n_tri)
    for t in range(n_tri):
        v = F[t]
        phys_areas[t] = 0.5 * abs(
            X[v[1], 0] * (X[v[2], 1] - X[v[0], 1]) +
            X[v[2], 0] * (X[v[0], 1] - X[v[1], 1]) +
            X[v[0], 0] * (X[v[1], 1] - X[v[2], 1])
        )
    phys_areas = np.maximum(phys_areas, 1e-15)

    n_tot = len(F)
    n_flipped = 0

    for t in range(n_tri):
        a, b, c = F[t]
        phys = X[[a, b, c]]   # (3,2) физические
        log = Y[[a, b, c]]    # (3,2) логические


        # Площадь в параметрическом пространстве
        A_log = 0.5 * (
            log[1, 0] * (log[2, 1] - log[0, 1]) +
            log[2, 0] * (log[0, 1] - log[1, 1]) +
            log[0, 0] * (log[1, 1] - log[2, 1])
        )
        if A_log < 0:
            n_flipped += 1

        if abs(A_log) < 1e-15:
            continue

        sign_log = 1.0 if A_log > 0 else -1.0


        A_log_abs = abs(A_log)

        # Якобиан отображения (ξ,η) → (x,y)
        dX = np.column_stack([phys[1] - phys[0], phys[2] - phys[0]])
        dU = np.column_stack([log[1] - log[0],   log[2] - log[0]])
        try:
            J = dX @ np.linalg.inv(dU)
        except np.linalg.LinAlgError:
            J = np.eye(2)

        # Первая фундаментальная форма I = JᵀJ
        #регуляризация Тихонова
        I_reg = J.T @ J + 1e-10 * np.eye(2)

        # Тензор C
        try:
            if C_mode == 'sqrt':
                C = np.real(sqrtm(I_reg))
            elif C_mode == 'invsqrt':
                C = np.real(np.linalg.inv(sqrtm(I_reg)))
            elif C_mode == 'inv':
                C = np.real(np.linalg.inv(I_reg))
            else:
                C = np.eye(2)
            C = np.nan_to_num(C, nan=1.0, posinf=1e12, neginf=-1e12)
        except Exception:
            C = np.eye(2)
            
        # Повёрнутые рёбра
        def ccw(v):
            return np.array([-v[1], v[0]])

        if eperp_mode == 'phys':
            # Перпендикуляры из физического пространства
            e_ab = phys[1] - phys[0]
            e_bc = phys[2] - phys[1]
            e_ca = phys[0] - phys[2]

            x_perp_BC = ccw(e_bc)   # ⊥ ребру BC
            x_perp_CA = ccw(e_ca)   # ⊥ ребру CA
            x_perp_AB = ccw(e_ab)   # ⊥ ребру AB

            area = phys_areas[t]
            sign_orient = 1.0  # физические треугольники всегда CCW
        else:
            # Перпендикуляры из параметрического (логического) пространства
            p_ab = log[1] - log[0]
            p_bc = log[2] - log[1]
            p_ca = log[0] - log[2]

            x_perp_BC = ccw(p_bc)   # x⊥_{j,k}
            x_perp_CA = ccw(p_ca)   # x⊥_{k,i}
            x_perp_AB = ccw(p_ab)   # x⊥_{i,j}

            area = A_log_abs
            sign_orient = sign_log

        # Формула (12): для каждого ребра треугольника
        #   w_ij += (x⊥_{j,k} · C · x⊥_{k,i}) / (4·A)
        edges = [
            (a, b, x_perp_BC, x_perp_CA),
            (b, c, x_perp_CA, x_perp_AB),
            (c, a, x_perp_AB, x_perp_BC),
        ]

        for vi, vj, e1, e2 in edges:
            # w = -sign_orient · (e1·C·e2) / (4·A)  →  положительные веса
            val = - sign_orient * (e1 @ (C @ e2)) / (4.0 * area)
            if abs(val) < 1e-15:
                continue
            rows.extend([vi, vj])
            cols.extend([vj, vi])
            data.extend([val, val])
    
    print(f"flipped = {n_flipped}, n_tot = {n_tot}, ratio = {n_flipped/n_tot} ")

    if not data:
        return coo_matrix((n_nodes, n_nodes)).tocsr()

    return coo_matrix((data, (rows, cols)),
                      shape=(n_nodes, n_nodes)).tocsr()


# 4.  Mapping
def harmonic_map(X, F, bnds):
    """Гармоническое отображение: Laplace(ξ)=0, Laplace(η)=0."""
    W = build_cotangent_laplacian(X, F)
    deg = np.array(W.sum(axis=1)).flatten()
    L = (diags(deg) - W).tocsr()

    fix_xi_nodes = list(bnds['inlet']) + list(bnds['outlet'])
    fix_xi_vals  = [0.0] * len(bnds['inlet']) + [1.0] * len(bnds['outlet'])
    fix_eta_nodes = list(bnds['bottom']) + list(bnds['top'])
    fix_eta_vals  = [0.0] * len(bnds['bottom']) + [1.0] * len(bnds['top'])

    xi  = solve_dirichlet(L, X.shape[0], fix_xi_nodes, fix_xi_vals)
    eta = solve_dirichlet(L, X.shape[0], fix_eta_nodes, fix_eta_vals)

    return np.column_stack([xi, eta])


def quasi_harmonic_map(X, F, bnds, Y_init, C_mode='inv',
                       eperp_mode='log', n_iter=10, damping=0.3,
                       verbose=True):
    """
    Итерационное quasi-harmonic отображение с демпфированием.

    Параметры
    C_mode : 'inv' | 'sqrt' | 'invsqrt'
    eperp_mode : 'log' | 'phys'
    damping : коэффициент релаксации (0.3 — оптимально)
    n_iter : число итераций
    """
    Y = Y_init.copy()

    for it in range(n_iter):
        W = build_qh_weights(X, F, Y, C_mode=C_mode, eperp_mode=eperp_mode)
        deg = np.array(W.sum(axis=1)).flatten()
        L = (diags(deg) - W).tocsr()

        #Условия Дирихле
        fix_xi_nodes = list(bnds['inlet']) + list(bnds['outlet'])
        fix_xi_vals  = [0.0] * len(bnds['inlet']) + [1.0] * len(bnds['outlet'])
        fix_eta_nodes = list(bnds['bottom']) + list(bnds['top'])
        fix_eta_vals  = [0.0] * len(bnds['bottom']) + [1.0] * len(bnds['top'])

        xi  = solve_dirichlet(L, X.shape[0], fix_xi_nodes, fix_xi_vals)
        eta = solve_dirichlet(L, X.shape[0], fix_eta_nodes, fix_eta_vals)

        Y_new = np.column_stack([xi, eta])

        if np.any(np.isnan(Y_new)) or np.any(np.isinf(Y_new)):
            if verbose:
                print(f"  iter {it+1}/{n_iter}: NaN/Inf — стоп")
            break

        diff = np.max(np.abs(Y_new - Y))
        Y = damping * Y_new + (1 - damping) * Y

        if verbose:
            print(f"  iter {it+1}/{n_iter}: Δ={diff:.2e}")

    return Y


def compute_distortions(X, Y, F):
    """(angle_dist, area_dist) per triangle."""
    n_tri = F.shape[0]
    angle_dist = np.zeros(n_tri)
    area_dist = np.zeros(n_tri)

    for t in range(n_tri):
        a, b, c = F[t]
        phys = X[[a, b, c]]
        log = Y[[a, b, c]]

        # Угловое искажение (cond(J))
        dX = np.column_stack([phys[1] - phys[0], phys[2] - phys[0]])
        dU = np.column_stack([log[1] - log[0],   log[2] - log[0]])
        try:
            J = dX @ np.linalg.inv(dU)
            S = np.linalg.svd(J, compute_uv=False)
            cond = max(S[0], 1e-15) / max(S[-1], 1e-15)
            angle_dist[t] = min(max(cond, 1.0), 1000.0)
        except np.linalg.LinAlgError:
            angle_dist[t] = 1.0

        # Площадное искажение
        A_phys = 0.5 * abs(
            phys[1,0]*(phys[2,1]-phys[0,1]) +
            phys[2,0]*(phys[0,1]-phys[1,1]) +
            phys[0,0]*(phys[1,1]-phys[2,1])
        )
        A_log = 0.5 * abs(
            log[1,0]*(log[2,1]-log[0,1]) +
            log[2,0]*(log[0,1]-log[1,1]) +
            log[0,0]*(log[1,1]-log[2,1])
        )
        area_dist[t] = max(A_phys, 1e-15) / max(A_log, 1e-15)

    mean_area = np.mean(area_dist)
    if mean_area > 1e-15:
        area_dist = area_dist / mean_area

    return angle_dist, area_dist


def count_flipped(Y, F):
    """Число треугольников с отрицательной площадью в параметрическом пространстве."""
    n = 0
    for t in range(F.shape[0]):
        a, b, c = F[t]
        log = Y[[a, b, c]]
        A = 0.5 * (log[1,0]*(log[2,1]-log[0,1]) +
                   log[2,0]*(log[0,1]-log[1,1]) +
                   log[0,0]*(log[1,1]-log[2,1]))
        if A < -1e-12:
            n += 1
    return n


def distortions_table(angle_dist, area_dist):
    return {
        'angle_max': float(np.max(angle_dist)),
        'angle_mean': float(np.mean(angle_dist)),
        'angle_median': float(np.median(angle_dist)),
        'area_max': float(np.max(area_dist)),
        'area_mean': float(np.mean(area_dist)),
        'area_median': float(np.median(area_dist)),
    }


# 6.  MAIN
if __name__ == "__main__":
    msh_path = os.path.join(os.path.dirname(__file__),
                            "../test_domains/step_1.msh")
    print(f"Загружаем: {msh_path}")
    X, F, bnds = read_msh_file(msh_path)
    print(f"Узлов: {X.shape[0]}, треугольников: {F.shape[0]}")


    print("  ГАРМОНИЧЕСКОЕ ОТОБРАЖЕНИЕ")

    t0 = time.time()
    Y_harm = harmonic_map(X, F, bnds)
    t_harm = time.time() - t0
    nf_harm = count_flipped(Y_harm, F)
    print(f"  flipped={nf_harm}, время={t_harm:.4f} сек")

    # 2. Quasi-harmonic: 6 комбинаций
    combos = [
        ('sqrt',   'log',  'sqrt + log'),
        ('sqrt',   'phys', 'sqrt + phys'),
        ('invsqrt','log',  'invsqrt + log'),
        ('invsqrt','phys', 'invsqrt + phys'),
        ('inv',    'log',  'inv + log'),
        ('inv',    'phys', 'inv + phys'),
    ]

    results = {}
    for C_mode, eperp_mode, label in combos:
        print(f"\n  --- {label} ---")
        t0 = time.time()
        Y_qh = quasi_harmonic_map(X, F, bnds, Y_harm,
                                  C_mode=C_mode, eperp_mode=eperp_mode,
                                  n_iter=5, damping=0.3, verbose=True)
        t_qh = time.time() - t0
        nf_qh = count_flipped(Y_qh, F)
        angle_q, area_q = compute_distortions(X, Y_qh, F)
        s_q = distortions_table(angle_q, area_q)
        results[label] = {**s_q, 'flipped': nf_qh, 'time': t_qh, 'Y': Y_qh}

    # 3. Сводная таблица
    angle_h, area_h = compute_distortions(X, Y_harm, F)
    s_h = distortions_table(angle_h, area_h)

    print("\n" + "=" * 105)
    print(f"{'Метрика':<20} {'Harmonic':<12}", end="")
    for label in [c[2] for c in combos]:
        print(f"{label:<20}", end="")
    print()
    print("-" * 105)

    for key in ['angle_max', 'angle_mean', 'angle_median',
                'area_max', 'area_mean']:
        print(f"{key:<20} {s_h[key]:<12.4f}", end="")
        for _, _, label in combos:
            val = results[label][key]
            chg = (val - s_h[key]) / max(s_h[key], 1e-15) * 100
            print(f"{val:<12.4f} ({chg:+.1f}%)", end=" ")
        print()

    print(f"{'flipped':<20} {nf_harm:<12}", end="")
    for _, _, label in combos:
        print(f"{results[label]['flipped']:<20}", end="")
    print()
    print(f"{'time (sec)':<20} {t_harm:<12.4f}", end="")
    for _, _, label in combos:
        print(f"{results[label]['time']:<20.4f}", end="")
    print()
    print("=" * 105)

    # 4. Визуализация: лучшая и худшая по area_max
    def per_vertex_area(Y):
        dist = np.ones(X.shape[0])
        _, area_d = compute_distortions(X, Y, F)
        for t in range(F.shape[0]):
            for v in F[t]:
                dist[v] = min(dist[v], area_d[t])
        return dist

    # Сортируем по area_max (чем меньше — тем лучше)
    sorted_combos = sorted(combos, key=lambda c: results[c[2]]['area_max'])
    best_label = sorted_combos[0][2]
    worst_label = sorted_combos[-1][2]

    v_h = per_vertex_area(Y_harm)
    v_best = per_vertex_area(results[best_label]['Y'])
    v_worst = per_vertex_area(results[worst_label]['Y'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    panels = [
        (axes[0], Y_harm, v_h, f"Harmonic (flipped={nf_harm})"),
        (axes[1], results[best_label]['Y'], v_best,
         f"Best: {best_label} (flipped={results[best_label]['flipped']})"),
        (axes[2], results[worst_label]['Y'], v_worst,
         f"Worst: {worst_label} (flipped={results[worst_label]['flipped']})"),
    ]

    for ax, Y, vd, title in panels:
        sc = ax.scatter(Y[:, 0], Y[:, 1], s=4, c=vd,
                        cmap='plasma', alpha=0.7, edgecolors='none')
        for bn, cl in [('inlet', 'r'), ('outlet', 'b'),
                       ('bottom', 'g'), ('top', 'm')]:
            ids = bnds[bn]
            if ids:
                ax.scatter(Y[ids, 0], Y[ids, 1], c=cl, s=10,
                           edgecolors='k', linewidths=0.3, zorder=5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("ξ")
        ax.set_ylabel("η")
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, shrink=0.8, label="area distortion")

    plt.suptitle("Сравнение 6 комбинаций QH (step_1.msh)", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__),
                            "qh_6combos.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nСохранено: {out_path}")
