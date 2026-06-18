#!/usr/bin/env python3
"""
generate_data.py — Генерация stationary NS данных через Firedrake.

Пайплайн:
  1. Читает .msh из test_domains/
  2. Решает Stokes → Newton (ламинарный, Re ∈ [50, 400])
  3. Интерполирует на регулярную сетку N×N в [0,L]×[0,H]
  4. Строит маску (полигон жидкости)
  5. Сохраняет маску + поля в .npy для FNO

Зависимости: firedrake, numpy, scipy, matplotlib
"""
import os
import sys
import random
import numpy as np
from scipy.interpolate import griddata
import matplotlib.path as mpath

# Пути
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MSH_DIR = os.path.join(BASE_DIR, "create_dif_d_step", "test_domains")
PARAMS_PATH = os.path.join(MSH_DIR, "geometry_params.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "fno", "dataset_laminar")

N_GRID = 128
N_SAMPLES = 100   # все имеющиеся геометрии

# Firedrake импорты (тяжёлые, только в main)


# Полигон (копия из dataset.py)
def build_fluid_polygon(L, H, x_start, y_peak, frontSteps, backSteps, x_end):
    verts = []
    verts.append((0.0, 0.0))
    verts.append((x_start, 0.0))
    x_peak = (x_start + x_end) / 2.0
    n_f = frontSteps + 1
    dx_f = (x_peak - x_start) / n_f
    dy_f = y_peak / n_f
    cur_x, cur_y = x_start, 0.0
    for i in range(1, n_f + 1):
        cur_y += dy_f
        if i == n_f: cur_y = y_peak
        verts.append((cur_x, cur_y))
        cur_x += dx_f
        if i == n_f: cur_x = x_peak
        verts.append((cur_x, cur_y))
    n_b = backSteps + 1
    dx_b = (x_end - x_peak) / n_b
    dy_b = y_peak / n_b
    for i in range(1, n_b + 1):
        cur_x += dx_b
        if i == n_b: cur_x = x_end
        verts.append((cur_x, cur_y))
        target_y = y_peak - i * dy_b
        if i == n_b: target_y = 0.0
        verts.append((cur_x, target_y))
        cur_y = target_y
    verts.append((L, 0.0))
    verts.append((L, H))
    verts.append((0.0, H))
    return verts


def compute_x_end(geom_id, L):
    rng = random.Random(geom_id)
    _ = 0.5 + 0.5 * rng.random()
    _ = rng.uniform(2.0, 4.0) * _
    _ = rng.uniform(1.0, 2.5) * _
    rw = L
    _ = rng.uniform(0, rw / 8.0) * rng.choice([1, -1])
    de = rng.uniform(0, rw / 8.0) * rng.choice([1, -1])
    return (3 * rw / 4.0) + de


def load_geometry_params(path=PARAMS_PATH):
    params = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        vals = line.strip().split(',')
        if len(vals) >= 8:
            gid = int(vals[0])
            L = float(vals[2])
            params.append({
                'id': gid,
                'L': L,
                'H': float(vals[3]),
                'frontSteps': int(vals[4]),
                'backSteps': int(vals[5]),
                'x_start': float(vals[6]),
                'y_peak': float(vals[7]),
                'x_end': compute_x_end(gid, L),
            })
    return params


# Регулярная сетка + маска
def make_grid_and_mask(gp, N=N_GRID):
    """Возвращает (mask, XI, ETA, x_phys, y_phys) на сетке N×N."""
    verts = build_fluid_polygon(gp['L'], gp['H'],
                                 gp['x_start'], gp['y_peak'],
                                 gp['frontSteps'], gp['backSteps'],
                                 gp['x_end'])
    path = mpath.Path(verts)
    xi = np.linspace(0, 1, N)
    eta = np.linspace(0, 1, N)
    XI, ETA = np.meshgrid(xi, eta)
    x_phys = XI * gp['L']
    y_phys = ETA * gp['H']
    pts = np.column_stack([x_phys.ravel(), y_phys.ravel()])
    mask = path.contains_points(pts).reshape(N, N).astype(np.float32)
    return mask, XI, ETA, x_phys, y_phys


# Main: генерация через Firedrake
def main():
    import firedrake
    from firedrake import *

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Загружаем параметры геометрий
    all_gp = load_geometry_params()
    print(f"Загружено {len(all_gp)} геометрий")

    # Ищем .msh файлы
    msh_files = sorted(
        [f for f in os.listdir(MSH_DIR) if f.endswith('.msh') and 'step' in f],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    print(f"Найдено {len(msh_files)} .msh файлов")

    total = min(N_SAMPLES, len(msh_files), len(all_gp))

    # Итоговые массивы
    all_mask = np.zeros((total, 1, N_GRID, N_GRID), dtype=np.float32)
    all_xi   = np.zeros((total, 1, N_GRID, N_GRID), dtype=np.float32)
    all_eta  = np.zeros((total, 1, N_GRID, N_GRID), dtype=np.float32)
    all_u    = np.zeros((total, 1, N_GRID, N_GRID), dtype=np.float32)
    all_v    = np.zeros((total, 1, N_GRID, N_GRID), dtype=np.float32)
    all_p    = np.zeros((total, 1, N_GRID, N_GRID), dtype=np.float32)
    all_re   = np.zeros((total, 1), dtype=np.float32)

    for idx in range(total):
        gp = all_gp[idx]
        msh_path = os.path.join(MSH_DIR, msh_files[idx])
        print(f"\n[{idx+1}/{total}] id={gp['id']}, L={gp['L']:.2f}, H={gp['H']:.2f}")

        # Сетка + маска
        mask, XI, ETA, x_phys, y_phys = make_grid_and_mask(gp)

        # Рандомное Re
        # Re_val = random.uniform(50.0, 400.0)
        Re_val = 200.0  # пока фиксированное для отладки
        print(f"  Re = {Re_val:.1f}")

        try:
            # Firedrake: Mesh + Function Spaces
            mesh = Mesh(msh_path)
            V = VectorFunctionSpace(mesh, "CG", 2)
            Q = FunctionSpace(mesh, "CG", 1)
            Z = V * Q

            up = Function(Z)
            u, p = split(up)
            v, q = TestFunctions(Z)

            Re = Constant(Re_val)
            nu = 1.0 / Re

            coords = mesh.coordinates.dat.data_ro
            H_ch = Constant(coords[:, 1].max())
            U_max = Constant(1.0)
            x_spatial, y_spatial = SpatialCoordinate(mesh)

            u_inflow = as_vector([
                4.0 * U_max * y_spatial * (H_ch - y_spatial) / (H_ch**2),
                0.0
            ])

            bcs = [
                DirichletBC(Z.sub(0), u_inflow, 1),
                DirichletBC(Z.sub(0), Constant((0.0, 0.0)), 3),
                DirichletBC(Z.sub(0), Constant((0.0, 0.0)), 4),
            ]

            # Stokes (начальное приближение)
            F_stokes = (nu * inner(grad(u), grad(v)) * dx
                        - p * div(v) * dx + div(u) * q * dx)
            print("  Stokes...")
            solve(F_stokes == 0, up, bcs=bcs,
                  solver_parameters={
                      "ksp_type": "preonly",
                      "pc_type": "lu",
                      "pc_factor_mat_solver_type": "mumps"
                  })

            # Navier-Stokes (Newton)
            F_ns = F_stokes + inner(dot(grad(u), u), v) * dx
            print("  Navier-Stokes...")
            solve(F_ns == 0, up, bcs=bcs,
                  solver_parameters={
                      "snes_monitor": None,
                      "ksp_type": "preonly",
                      "pc_type": "lu",
                      "pc_factor_mat_solver_type": "mumps"
                  })
            u_sol, p_sol = up.subfunctions

            # Интерполяция на регулярную сетку
            print("  Interpolating...")
            fluid_mask = mask > 0.5
            fluid_pts = np.column_stack([
                x_phys[fluid_mask].ravel(),
                y_phys[fluid_mask].ravel()
            ])

            if len(fluid_pts) > 0:
                u_vec_vals = np.array(u_sol.at(fluid_pts))
                p_vals = np.array(p_sol.at(fluid_pts))

                # Раскладываем на U, V
                u_fluid = u_vec_vals[:, 0]
                v_fluid = u_vec_vals[:, 1]

                # Собираем в полные массивы N×N
                u_full = np.full((N_GRID, N_GRID), 0.0, dtype=np.float32)
                v_full = np.full((N_GRID, N_GRID), 0.0, dtype=np.float32)
                p_full = np.full((N_GRID, N_GRID), 0.0, dtype=np.float32)

                u_full[fluid_mask] = u_fluid
                v_full[fluid_mask] = v_fluid
                p_full[fluid_mask] = p_vals
            else:
                u_full = np.zeros((N_GRID, N_GRID))
                v_full = np.zeros((N_GRID, N_GRID))
                p_full = np.zeros((N_GRID, N_GRID))

            # Сохраняем
            all_mask[idx, 0] = mask
            all_xi[idx, 0] = XI
            all_eta[idx, 0] = ETA
            all_u[idx, 0] = u_full
            all_v[idx, 0] = v_full
            all_p[idx, 0] = p_full
            all_re[idx, 0] = Re_val

            print(f"  ✓ Сохранено (fluid={fluid_mask.sum()}/{N_GRID*N_GRID})")

        except Exception as e:
            print(f"  ✗ ОШИБКА: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Сохраняем датасет
    print(f"\nСохранение датасета в {OUTPUT_DIR}...")
    np.save(os.path.join(OUTPUT_DIR, "mask.npy"), all_mask)
    np.save(os.path.join(OUTPUT_DIR, "xi.npy"),   all_xi)
    np.save(os.path.join(OUTPUT_DIR, "eta.npy"),  all_eta)
    np.save(os.path.join(OUTPUT_DIR, "u.npy"),    all_u)
    np.save(os.path.join(OUTPUT_DIR, "v.npy"),    all_v)
    np.save(os.path.join(OUTPUT_DIR, "p.npy"),    all_p)
    np.save(os.path.join(OUTPUT_DIR, "re.npy"),   all_re)
    print(f"Готово! {total} сэмплов")


if __name__ == "__main__":
    main()
