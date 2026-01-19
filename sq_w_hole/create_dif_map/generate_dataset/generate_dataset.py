import firedrake
from firedrake import *
import numpy as np
import csv
import os
import random
import glob
import re
import sys
import matplotlib.pyplot as plt
from firedrake.output import VTKFile

# --- КОНФИГУРАЦИЯ ---
# Папки с данными
MSH_FOLDER = "../../test_sq_part_obj"           # Где лежат MSH файлы
INTERP_RESULTS_DIR = "../final_test_dataset"         # Где лежат x_data.csv и y_data.csv
OUTPUT_DIR = "./test_dataset_output"                 # Куда сохранять U.csv, C.csv и картинки
PARAMS_CSV = os.path.join(MSH_FOLDER, "geometry_params.csv")

GRID_X_FILE = os.path.join(INTERP_RESULTS_DIR, "x_data.csv")
GRID_Y_FILE = os.path.join(INTERP_RESULTS_DIR, "y_data.csv")

# Параметры PDE

# --- ФУНКЦИИ ---

def load_geometry_params(csv_path):
    """
    Загружает параметры геометрий в словарь.
    Returns: dict { geometry_id (int) : scale (float) }
    """
    params_map = {}
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Файл параметров не найден: {csv_path}. Запустите generate_quad_geometries.py с обновлением!")
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                geom_id = int(row['id'])
                scale = float(row['scale'])
                params_map[geom_id] = scale
            except (ValueError, KeyError):
                continue
    print(f"Загружено {len(params_map)} параметров масштаба.")
    return params_map

def extract_id_from_filename(filename):
    """Извлекает ID из имени файла вида quad_123.msh"""
    basename = os.path.basename(filename)
    match = re.search(r'quad_(\d+)\.msh', basename)
    if match:
        return int(match.group(1))
    return None

def clear_output_files(output_u, output_c):
    """Удаляет старые файлы результатов перед запуском."""
    if os.path.exists(output_u):
        os.remove(output_u)
    if os.path.exists(output_c):
        os.remove(output_c)
    print("Старые U.csv и C.csv удалены.")

def get_random_coefficients():
    """Генерация случайных коэффициентов для PDE."""
    cof1 = random.uniform(0.20, 0.80)
    cof2 = random.uniform(0.20, 0.80)
    
    if random.random() > 0.5:
        val = random.randint(1, 50)
        scale_c = 0.01 + 0.02 * val
    else:
        val = random.randint(1, 60)
        scale_c = 1.0 + 0.05 * val
        
    return [cof1, cof2], scale_c

def load_grid_row(row_idx):
    """
    Загружает конкретную строку (номер row_idx) из CSV файлов координат.
    Это координаты интерполяции для конкретного объекта.
    """
    # Читаем X
    with open(GRID_X_FILE, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == row_idx:
                x_flat = np.array(row, dtype=float)
                break
    
    # Читаем Y
    with open(GRID_Y_FILE, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == row_idx:
                y_flat = np.array(row, dtype=float)
                break
                
    # Находим валидные точки (не дырка)
    valid_mask = ~np.isnan(x_flat)
    valid_points = np.column_stack((x_flat[valid_mask], y_flat[valid_mask]))
    
    return x_flat, valid_mask, valid_points

def visualize_heatmap(u_data, c_data, save_path, title_prefix, sampling_size=128):
    """Строит и сохраняет тепловую карту для одного объекта."""
    U_grid = u_data.reshape((sampling_size, sampling_size))
    C_grid = c_data.reshape((sampling_size, sampling_size))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(U_grid, cmap='inferno', origin='lower', interpolation='nearest')
    axes[0].set_title(f"{title_prefix}: U Solution")
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(C_grid, cmap='viridis', origin='lower', interpolation='nearest')
    axes[1].set_title(f"{title_prefix}: C Coefficient")
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

def append_to_csv(filename, data_row):
    """Добавляет строку в CSV."""
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data_row.tolist())

# --- MAIN ---

def main():
# --- 1. Чтение аргументов (start_idx, end_idx) ---
    if len(sys.argv) >= 3:
        try:
            start_idx = int(sys.argv[1])
            end_idx = int(sys.argv[2])
        except ValueError:
            print("Ошибка: индексы должны быть числами. Пример: python script.py 0 100")
            return
    else:
        # Режим по умолчанию (все файлы) - если запускаем просто так
        start_idx = 0
        end_idx = 999999
        print("Запуск без аргументов: обработка всех файлов.")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    path_u = os.path.join(OUTPUT_DIR, "U.csv")
    path_c = os.path.join(OUTPUT_DIR, "C.csv")

    # --- 2. Очистка файлов ---
    # Чистим ТОЛЬКО если это самый первый батч (начинаем с 0)
    if start_idx == 0:
        clear_output_files(path_u, path_c)
    else:
        print(f"Дописываем данные в существующие файлы (Batch {start_idx}-{end_idx})...")

    # Загружаем карту параметров {id: scale}
    try:
        scale_params = load_geometry_params(PARAMS_CSV)
    except Exception as e:
        print(f"ОШИБКА: {e}")
        return

    # Получаем список всех msh файлов (сортировка обязательна!)
    msh_files_all = sorted(glob.glob(os.path.join(MSH_FOLDER, "*.msh")))
    total_files = len(msh_files_all)
    print(f"Всего найдено {total_files} MSH файлов.")
    
    if not os.path.exists(GRID_X_FILE):
        print(f"ОШИБКА: Файл {GRID_X_FILE} не найден. Запустите сначала create_map.py!")
        return

    # --- 3. Выбор подмножества файлов для обработки ---
    # Ограничиваем список нашим диапазоном
    files_to_process = msh_files_all[start_idx : end_idx]
    
    if not files_to_process:
        print("Нет файлов для обработки в заданном диапазоне.")
        return

    print(f"Начинаю обработку диапазона: {start_idx} -> {min(end_idx, total_files)}")

    # ЦИКЛ ПО ВЫБРАННОЙ ПАРТИИ
    # Используем enumerate, но индекс i идет от 0.
    # Нам нужен реальный глобальный индекс для CSV: global_idx = start_idx + i
    for i, msh_file in enumerate(files_to_process):
        global_idx = start_idx + i
        
        base_name = os.path.splitext(os.path.basename(msh_file))[0]
        
        # Извлекаем ID для поиска масштаба
        geom_id = extract_id_from_filename(msh_file)
        if geom_id is None or geom_id not in scale_params:
            print(f"⚠ WARN: Не найден ID или масштаб для {base_name}. Пропускаем.")
            continue
            
        current_scale = scale_params[geom_id]
        
        print(f"\n--- [{global_idx+1}/{total_files}] {base_name} (ID={geom_id}, Scale={current_scale:.4f}) ---")
        
        # 1. Загрузка строки координат для ЭТОГО объекта
        # ВАЖНО: передаем global_idx, чтобы взять правильную строку из большого CSV
        try:
            x_flat_template, valid_mask, valid_points = load_grid_row(global_idx)
        except IndexError:
            print(f"ОШИБКА: В файлах x_data/y_data меньше строк ({global_idx}), чем файлов msh!")
            break
        except Exception as e:
             print(f"ОШИБКА чтения CSV строки {global_idx}: {e}")
             continue
            
        # 2. Генерация параметров PDE
        cof, scale_c = get_random_coefficients()
        # print(f"  Params: cof={cof}, scale={scale_c}")
        
        try:
            # 3. Загрузка FEM сетки
            mesh = Mesh(msh_file)
            V = FunctionSpace(mesh, "CG", 3)
            
            # 4. Формулировка задачи
            x, y = SpatialCoordinate(mesh)
            c_expression = (cof[0] * sin(pi * (x / current_scale / 10.0)) - 
                            cof[1] * (x / current_scale) * (x / current_scale - 10.0) + 2.0) * scale_c
            
            u = TrialFunction(V)
            v = TestFunction(V)
            f = Constant(1.0)
            
            a = c_expression * dot(grad(u), grad(v)) * dx
            L = f * v * dx
            bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]
            
            # 5. Решение
            # print("  Solving PDE...")
            u_sol = Function(V)
            solve(a == L, u_sol, bcs=bcs)
            
            c_sol = Function(V)
            c_sol.interpolate(c_expression)
            
            u_vals = u_sol.at(valid_points)
            c_vals = c_sol.at(valid_points)
            
            # Заполняем полные массивы (с NaN)
            u_final_row = np.full_like(x_flat_template, np.nan)
            c_final_row = np.full_like(x_flat_template, np.nan)
            
            u_final_row[valid_mask] = u_vals
            c_final_row[valid_mask] = c_vals
            
            # 7. Запись в общие CSV
            append_to_csv(path_u, u_final_row)
            append_to_csv(path_c, c_final_row)
            print("  Data saved to CSV.")
            
            # 8. Сохранение картинок
            # Папка для картинок конкретного объекта
            obj_img_dir = os.path.join(OUTPUT_DIR, "images", base_name)
            os.makedirs(obj_img_dir, exist_ok=True)
            
            # Тепловая карта
            img_path = os.path.join(obj_img_dir, "heatmap.png")
            visualize_heatmap(u_final_row, c_final_row, img_path, base_name)

            img_path = os.path.join(obj_img_dir, "heatmap.png")
            visualize_heatmap(u_final_row, c_final_row, img_path, base_name)
            
            # Сохраняем физическое решение для Paraview
            pvd_path = os.path.join(obj_img_dir, "solution.pvd")
            VTKFile(pvd_path).write(u_sol, c_sol)
            print(f"  VTK saved to {pvd_path}")
            plt.close('all')
        except Exception as e:
            print(f"  CRITICAL ERROR на объекте {base_name}: {e}")
            # Если произошла ошибка, чтобы не сбивать порядок строк в CSV,
            # можно записать пустую строку или строку NaN?
            # Лучше просто остановить, чтобы разобраться.
            continue

    print(f"\nПартия {start_idx}-{end_idx} завершена.")

if __name__ == "__main__":
    main()
