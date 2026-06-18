
from firedrake import *
import numpy as np
import csv
import os
import glob
import random
import matplotlib.pyplot as plt

MSH_FOLDER = "train_domains_w_steps"
INTERP_RESULTS_DIR = "train_data_d_step"
OUTPUT_DIR = "dns_averaged_dataset"


# Физические параметры (Турбулентный режим)
RE_MIN, RE_MAX = 800.0, 1500.0 
DT = 0.01                                       # Шаг времени (0.01 стабилен для таких Re)
T_BURN = 20.0                                   # Время на "прогрев" (выход на режим)
T_SAMPLING = 25.0                               # Время на сбор статистики (усреднение)

# Файлы входной сетки (координаты диффеоморфизма)
GRID_X_FILE = os.path.join(INTERP_RESULTS_DIR, "x_data.csv")
GRID_Y_FILE = os.path.join(INTERP_RESULTS_DIR, "y_data.csv")

# --- 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def clear_output_files(paths):
    for p in paths:
        if os.path.exists(p):
            os.remove(p)
    print("Старые файлы очищены.")

def load_grid_row(row_idx):
    """Загружает конкретную строку из CSV файлов координат маппинга."""
    with open(GRID_X_FILE, 'r') as fx, open(GRID_Y_FILE, 'r') as fy:
        x_reader = list(csv.reader(fx))
        y_reader = list(csv.reader(fy))
    
    x_flat = np.array(x_reader[row_idx], dtype=float)
    y_flat = np.array(y_reader[row_idx], dtype=float)
    
    valid_mask = ~np.isnan(x_flat)
    valid_points = np.column_stack((x_flat[valid_mask], y_flat[valid_mask]))
    
    return x_flat, y_flat, valid_mask, valid_points

def append_to_csv(filename, data_row):
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data_row.tolist())

def visualize_results(u_data, v_data, p_data, save_path, title, size=128):
    """Создает диагностический PNG с полями."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    U = u_data.reshape((size, size))
    V = v_data.reshape((size, size))
    P = p_data.reshape((size, size))
    
    mag = np.sqrt(U**2 + V**2)
    
    im1 = axes[0].imshow(mag, cmap='inferno', origin='lower')
    axes[0].set_title(f"Velocity Mag: {title}")
    fig.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(U, cmap='coolwarm', origin='lower')
    axes[1].set_title("U component (Horizontal)")
    fig.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(P, cmap='viridis', origin='lower')
    axes[2].set_title("Pressure")
    fig.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --- 3. ОСНОВНОЙ ЦИКЛ ГЕНЕРАЦИИ ---

def main():
    # Настройка путей сохранения
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    paths = {
        'x': os.path.join(OUTPUT_DIR, "x_data.csv"),
        'y': os.path.join(OUTPUT_DIR, "y_data.csv"),
        'u': os.path.join(OUTPUT_DIR, "u_data.csv"),
        'v': os.path.join(OUTPUT_DIR, "v_data.csv"),
        'p': os.path.join(OUTPUT_DIR, "p_data.csv"),
        're': os.path.join(OUTPUT_DIR, "re_data.csv")
    }

    clear_output_files(list(paths.values()))

    # ПРАВИЛЬНАЯ СОРТИРОВКА (по числам после step_)
    raw_files = [f for f in os.listdir(MSH_FOLDER) if f.endswith(".msh") and "step" in f]
    sorted_names = sorted(raw_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    msh_files = [os.path.join(MSH_FOLDER, f) for f in sorted_names]

    print(f"Найдено файлов для обработки: {len(msh_files)}")

    for i, msh_path in enumerate(msh_files):
        base_name = os.path.basename(msh_path).replace('.msh', '')
        Re_val = random.uniform(RE_MIN, RE_MAX)
        
        print(f"\n>>> [{i+1}/{len(msh_files)}] Задача: {base_name} | Re = {Re_val:.1f}")

        try:
            # 1. Загрузка данных маппера
            x_flat, y_flat, valid_mask, valid_pts = load_grid_row(i)


            # 2. Настройка FEM пространства (Тейлор-Худ P2-P1)
            mesh = Mesh(msh_path)
            V = VectorFunctionSpace(mesh, "CG", 2) # Скорость (P2)
            Q = FunctionSpace(mesh, "CG", 1)       # Давление (P1)
            Z = V * Q

            up = Function(Z)
            u, p = split(up)
            up_old = Function(Z)
            u_old, p_old = split(up_old)
            v, q = TestFunctions(Z)
            
            # Параметры задачи
            nu = Constant(1.0 / Re_val)
            dt = Constant(DT)
            H = mesh.coordinates.dat.data_ro[:, 1].max()
            x_sp, y_sp = SpatialCoordinate(mesh)
            
            # Граничные условия
            u_inflow = as_vector([4.0 * 1.0 * y_sp * (H - y_sp) / (H**2), 0.0])
            bcs = [
                DirichletBC(Z.sub(0), u_inflow, 1),              # Вход
                DirichletBC(Z.sub(0), Constant((0.0, 0.0)), 3),  # Стенки/Ступеньки
                DirichletBC(Z.sub(0), Constant((0.0, 0.0)), 4)   # Потолок
            ]

            # 3. Решение Стокса (начальное приближение)
            F_stokes = (nu * inner(grad(u), grad(v)) * dx - p * div(v) * dx + div(u) * q * dx)
            solve(F_stokes == 0, up, bcs=bcs)
            up_old.assign(up)

            # 4. Нестационарный цикл DNS
            u_sum = Function(V)
            p_sum = Function(Q)
            steps_sampled = 0
            t = 0.0

            # Слабая форма (Неявный Эйлер)
            F_ns = (
                inner((u - u_old) / dt, v) * dx
                + inner(dot(u, grad(u)), v) * dx
                + nu * inner(grad(u), grad(v)) * dx
                - p * div(v) * dx
                + div(u) * q * dx
            )
            print(f"  Симуляция запущена...")
            while t < (T_BURN + T_SAMPLING):
                solve(F_ns == 0, up, bcs=bcs)
                up_old.assign(up)
                t += DT
                
                if t > T_BURN:
                    u_curr, p_curr = up.subfunctions
                    u_sum.assign(u_sum + u_curr)
                    p_sum.assign(p_sum + p_curr)
                    steps_sampled += 1
                
                if int(t/DT) % 500 == 0:
                    print(f"    Прогресс: {t:.1f}/{T_BURN+T_SAMPLING:.1f} сек.")

            # 5. Усреднение и Интерполяция
            u_mean = Function(V).assign(u_sum / float(steps_sampled))
            p_mean = Function(Q).assign(p_sum / float(steps_sampled))

            # Коррекция точек (чтобы не было ERROR: domain does not contain point)
            m_coords = mesh.coordinates.dat.data_ro
            x_min, y_min = m_coords.min(axis=0)
            x_max, y_max = m_coords.max(axis=0)
            eps = 1e-5
            valid_pts[:, 0] = np.clip(valid_pts[:, 0], x_min + eps, x_max - eps)
            valid_pts[:, 1] = np.clip(valid_pts[:, 1], y_min + eps, y_max - eps)

            print("  Интерполяция в сетку маппера...")
            u_vals = np.array(u_mean.at(valid_pts))
            p_vals = np.array(p_mean.at(valid_pts))

            # Сборка финальных массивов 128x128 (с учетом NaN маски)
            u_final = np.full_like(x_flat, np.nan)
            v_final = np.full_like(x_flat, np.nan)
            p_final = np.full_like(x_flat, np.nan)
            
            u_final[valid_mask] = u_vals[:, 0]
            v_final[valid_mask] = u_vals[:, 1]
            p_final[valid_mask] = p_vals

            # 6. Сохранение данных
            append_to_csv(paths['x'], x_flat)
            append_to_csv(paths['y'], y_flat)
            append_to_csv(paths['u'], u_final)
            append_to_csv(paths['v'], v_final)
            append_to_csv(paths['p'], p_final)
            append_to_csv(paths['re'], np.array([Re_val]))

            # Визуализация для контроля
            img_path = os.path.join(OUTPUT_DIR, "images", f"{base_name}.png")
            visualize_results(u_final, v_final, p_final, img_path, f"Re={Re_val:.1f}")
            
            print(f"  Готово. Данные сохранены.")

        except Exception as e:
            print(f"  !!! ОШИБКА на файле {base_name}: {e}")
            continue

if __name__ == "__main__":
    main()
