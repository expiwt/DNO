#!/usr/bin/env python3
"""
dataset.py — Загрузка и конвертация данных для FNO.

Строит регулярную сетку в физическом пространстве [0,L]×[0,H],
маску через point-in-polygon (лестница = ступенчатая нижняя граница),
интерполирует u,v,p со scattered данных на эту сетку.
Вход FNO: [mask, ξ, η] + Re (FiLM). Выход: [u, v, p].
"""
import os
import random
import numpy as np
from scipy.interpolate import griddata
import matplotlib.path as mpath
import matplotlib.pyplot as plt


# Пути
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dns_averaged_dataset")
PARAMS_PATH = os.path.join(BASE_DIR, "create_dif_d_step", "test_domains",
                           "geometry_params.csv")

N_GRID = 128
TRAIN_RATIO = 0.9


# Построение полигона жидкости (повторяет generate_1000.py)
def _build_fluid_polygon(L, H, x_start, y_peak, frontSteps, backSteps, x_end):
    """
    Строит замкнутый полигон CCW: лестница снизу → правая стенка →
    верхняя стенка → левая стенка → замыкание.

    Параметры — как в generate_1000.py.
    x_end — конец лестницы (восстанавливаем по seed).
    """
    verts = []

    # Нижняя граница: от (0,0) до начала лестницы
    verts.append((0.0, 0.0))
    verts.append((x_start, 0.0))

    # x_peak — середина лестницы (по горизонтали)
    x_peak = (x_start + x_end) / 2.0

    # --- Ступеньки ВВЕРХ ---
    n_f = frontSteps + 1
    dx_f = (x_peak - x_start) / n_f
    dy_f = y_peak / n_f
    cur_x = x_start
    cur_y = 0.0

    for i in range(1, n_f + 1):
        cur_y += dy_f
        if i == n_f:
            cur_y = y_peak
        verts.append((cur_x, cur_y))          # вверх
        cur_x += dx_f
        if i == n_f:
            cur_x = x_peak
        verts.append((cur_x, cur_y))          # вправо

    # --- Ступеньки ВНИЗ ---
    n_b = backSteps + 1
    dx_b = (x_end - x_peak) / n_b
    dy_b = y_peak / n_b

    for i in range(1, n_b + 1):
        cur_x += dx_b
        if i == n_b:
            cur_x = x_end
        verts.append((cur_x, cur_y))          # вправо
        target_y = y_peak - i * dy_b
        if i == n_b:
            target_y = 0.0
        verts.append((cur_x, target_y))       # вниз
        cur_y = target_y

    # До правой стенки (дно канала)
    verts.append((L, 0.0))

    # Стенки канала (CCW: вверх → налево → вниз к (0,0))
    verts.append((L, H))
    verts.append((0.0, H))

    return verts  # Path сам замкнёт


# Восстановление x_end по seed (как в generate_1000.py)
def _compute_x_end(geom_id, L):
    """Восстанавливает x_end, используя тот же seed, что в generate_1000."""
    rng = random.Random(geom_id)
    _ = 0.5 + 0.5 * rng.random()        # scale
    _ = rng.uniform(2.0, 4.0) * _       # L (не используем)
    _ = rng.uniform(1.0, 2.5) * _       # H
    range_width = L
    _ = rng.uniform(0, range_width / 8.0) * rng.choice([1, -1])  # delta_start
    delta_end = rng.uniform(0, range_width / 8.0) * rng.choice([1, -1])
    x_end = (3 * range_width / 4.0) + delta_end
    return x_end


# Чтение параметров геометрий
def load_geometry_params(path=PARAMS_PATH):
    """Возвращает список словарей с key: id, L, H, frontSteps, backSteps,
    x_start, y_peak, x_end."""
    params = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        vals = line.strip().split(',')
        if len(vals) >= 8:
            geom_id = int(vals[0])
            L = float(vals[2])
            H = float(vals[3])
            x_start = float(vals[6])
            y_peak = float(vals[7])
            x_end = _compute_x_end(geom_id, L)
            params.append({
                'id': geom_id,
                'L': L,
                'H': H,
                'frontSteps': int(vals[4]),
                'backSteps': int(vals[5]),
                'x_start': x_start,
                'y_peak': y_peak,
                'x_end': x_end,
            })
    return params


# Загрузка CSV
def load_csv_fast(path):
    print(f"  Загрузка {os.path.basename(path)}...")
    return np.loadtxt(path, delimiter=",")


# Конвертация одной геометрии на регулярную физ. сетку + маска
def geometry_to_physical_grid(geom_params, x_scattered, y_scattered,
                               u_scattered, v_scattered, p_scattered,
                               N=N_GRID):
    """
    Строит регулярную сетку [0,L]×[0,H], маску, интерполированные u,v,p.
    """
    L = geom_params['L']
    H = geom_params['H']

    # 1. Полигон области жидкости
    polygon_verts = _build_fluid_polygon(
        L, H,
        geom_params['x_start'], geom_params['y_peak'],
        geom_params['frontSteps'], geom_params['backSteps'],
        geom_params['x_end']
    )
    fluid_path = mpath.Path(polygon_verts)

    # 2. Регулярная сетка в [0,1]² → маппинг в [0,L]×[0,H]
    xi = np.linspace(0, 1, N)
    eta = np.linspace(0, 1, N)
    XI, ETA = np.meshgrid(xi, eta)
    x_phys = XI * L
    y_phys = ETA * H

    # 3. Маска: point-in-polygon
    grid_pts = np.column_stack([x_phys.ravel(), y_phys.ravel()])
    mask_flat = fluid_path.contains_points(grid_pts)
    mask = mask_flat.reshape(N, N).astype(np.float32)

    # 4. Интерполяция через griddata
    valid = ~np.isnan(x_scattered)
    if valid.sum() < 3:
        return mask, XI, ETA, np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))

    points = np.column_stack([x_scattered[valid], y_scattered[valid]])
    grid_points = np.column_stack([x_phys.ravel(), y_phys.ravel()])

    u_flat = griddata(points, u_scattered[valid], grid_points, method='linear')
    v_flat = griddata(points, v_scattered[valid], grid_points, method='linear')
    p_flat = griddata(points, p_scattered[valid], grid_points, method='linear')

    u_flat = np.nan_to_num(u_flat, nan=0.0)
    v_flat = np.nan_to_num(v_flat, nan=0.0)
    p_flat = np.nan_to_num(p_flat, nan=0.0)

    u_grid = u_flat.reshape(N, N)
    v_grid = v_flat.reshape(N, N)
    p_grid = p_flat.reshape(N, N)

    return mask, XI, ETA, u_grid, v_grid, p_grid


# Загрузка всего датасета
def load_dataset(data_dir=DATA_DIR, params_path=PARAMS_PATH,
                 N=N_GRID, train_ratio=TRAIN_RATIO):
    geom_params_list = load_geometry_params(params_path)
    num_geoms = len(geom_params_list)
    print(f"Загружено {num_geoms} геометрий")

    x_data = load_csv_fast(os.path.join(data_dir, "x_data.csv"))
    y_data = load_csv_fast(os.path.join(data_dir, "y_data.csv"))
    u_data = load_csv_fast(os.path.join(data_dir, "u_data.csv"))
    v_data = load_csv_fast(os.path.join(data_dir, "v_data.csv"))
    p_data = load_csv_fast(os.path.join(data_dir, "p_data.csv"))
    re_data = load_csv_fast(os.path.join(data_dir, "re_data.csv"))

    num_samples = min(u_data.shape[0], num_geoms)
    print(f"Сэмплов: {num_samples}")

    inputs = np.zeros((num_samples, 3, N, N), dtype=np.float32)
    targets = np.zeros((num_samples, 3, N, N), dtype=np.float32)
    re_scalar = np.zeros((num_samples, 1), dtype=np.float32)

    for i in range(num_samples):
        if (i + 1) % 100 == 0:
            print(f"  Конвертация {i+1}/{num_samples}...")

        gp = geom_params_list[i] if i < len(geom_params_list) else geom_params_list[-1]
        mask, XI, ETA, u_g, v_g, p_g = geometry_to_physical_grid(
            gp, x_data[i], y_data[i], u_data[i], v_data[i], p_data[i], N=N
        )

        inputs[i, 0] = mask
        inputs[i, 1] = XI
        inputs[i, 2] = ETA
        targets[i, 0] = u_g
        targets[i, 1] = v_g
        targets[i, 2] = p_g
        re_scalar[i, 0] = re_data[i] if re_data.ndim == 1 else re_data[i, 0]

    # Нормализация полей (только по fluid точкам)
    scalers = {}
    for name, data in [('u', targets[:, 0]), ('v', targets[:, 1]),
                       ('p', targets[:, 2])]:
        valid_mask = inputs[:, 0] > 0.5
        vals = data[valid_mask]
        mean = float(np.mean(vals))
        std = float(np.std(vals)) if np.std(vals) > 1e-10 else 1.0
        scalers[name] = {'mean': mean, 'std': std}
        data[:] = (data - mean) / std

    # Re: Z-score
    re_mean = float(np.mean(re_scalar))
    re_std = float(np.std(re_scalar)) if np.std(re_scalar) > 1e-10 else 1.0
    scalers['re'] = {'mean': re_mean, 'std': re_std}
    re_scalar = (re_scalar - re_mean) / re_std

    # Train/test split
    ntrain = int(num_samples * train_ratio)
    indices = np.random.RandomState(42).permutation(num_samples)
    train_idx = indices[:ntrain]
    test_idx = indices[ntrain:]

    train_inputs = inputs[train_idx]
    train_targets = targets[train_idx]
    train_re = re_scalar[train_idx]
    test_inputs = inputs[test_idx]
    test_targets = targets[test_idx]
    test_re = re_scalar[test_idx]

    print(f"\nTrain: {len(train_idx)}, Test: {len(test_idx)}")
    print(f"Input: {train_inputs.shape}, Target: {train_targets.shape}")
    return (train_inputs, train_targets, train_re,
            test_inputs, test_targets, test_re, scalers)


# Визуализация
def visualize_sample(mask, u, v, p, re_val=None, save_path=None):
    speed = np.sqrt(u**2 + v**2)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fields = [
        (mask, 'Mask', 'gray'),
        (u, 'U', 'coolwarm'),
        (v, 'V', 'coolwarm'),
        (speed, 'Speed', 'inferno'),
        (p, 'P', 'viridis'),
    ]
    for (arr, title, cmap), ax in zip(fields, axes.flat[:5]):
        im = ax.imshow(arr, cmap=cmap, origin='lower', aspect='equal')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[1, 2].axis('off')
    title = "Sample" + (f"  Re={re_val:.1f}" if re_val is not None else "")
    axes[0, 0].set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Тест
if __name__ == "__main__":
    print("=" * 55)
    print("  ТЕСТ dataset.py")
    print("=" * 55)

    params = load_geometry_params()
    print(f"Геометрий: {len(params)}")
    gp = params[0]
    print(f"  id={gp['id']}, L={gp['L']:.4f}, H={gp['H']:.4f}")
    print(f"  x_start={gp['x_start']:.4f}, y_peak={gp['y_peak']:.4f}")
    print(f"  x_end={gp['x_end']:.4f}, steps={gp['frontSteps']}+{gp['backSteps']}")

    # Проверка маски
    mask, XI, ETA, u_g, v_g, p_g = geometry_to_physical_grid(
        gp, np.zeros(16384), np.zeros(16384),
        np.zeros(16384), np.zeros(16384), np.zeros(16384)
    )
    print(f"  Маска: {mask.sum():.0f}/{N_GRID*N_GRID} fluid точек "
          f"({100*mask.sum()/(N_GRID*N_GRID):.1f}%)")

    # Сохраняем маску для проверки
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(mask, cmap='gray', origin='lower', extent=[0, gp['L'], 0, gp['H']])
    # Границы полигона
    verts = _build_fluid_polygon(gp['L'], gp['H'], gp['x_start'], gp['y_peak'],
                                  gp['frontSteps'], gp['backSteps'], gp['x_end'])
    xs, ys = zip(*verts)
    ax.plot(xs, ys, 'r-', lw=1, label='fluid polygon')
    ax.set_title(f"Mask + polygon (id={gp['id']})")
    ax.set_xlim(0, gp['L'])
    ax.set_ylim(0, gp['H'])
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    out = os.path.join(os.path.dirname(__file__), "mask_check.png")
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Маска сохранена: {out}")
