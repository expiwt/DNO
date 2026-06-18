import gmsh
import sys
import os
import random
import csv
import numpy as np

def generate_step_mesh(filename, geom_id, seed=None):
    """
    Генерирует один файл .msh с "лестницей" с разделенными границами Top/Bottom.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 1. Параметры геометрии    
    scale = 0.5 + 0.5 * random.random()

    L = random.uniform(2.0, 4.0) * scale
    H = random.uniform(1.0, 2.5) * scale
    
    lc = min(L, H) / 35.0 

    range_width = L 
    
    delta_start = random.uniform(0, range_width / 8.0) * random.choice([1, -1])
    x_start = (range_width / 4.0) + delta_start

    delta_end = random.uniform(0, range_width / 8.0) * random.choice([1, -1])
    x_end = (3 * range_width / 4.0) + delta_end

    if x_end <= x_start + 0.2:
        x_end = x_start + 0.2

    x_peak = (x_start + x_end) / 2.0
    
    delta_h = random.uniform(0, H / 8.0) * random.choice([1, -1])
    y_peak = (H / 2.0) + delta_h
    
    y_peak = max(0.2 * H, min(0.8 * H, y_peak))

    frontSteps = random.randint(1, 5)
    backSteps = random.randint(1, 5)

    # 2. Инициализация Gmsh
    gmsh.initialize()
    gmsh.model.add(f"step_{geom_id}")

    # 3. Построение точек
    p_inlet_bot = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p_inlet_top = gmsh.model.geo.addPoint(0, H, 0, lc)
    p_outlet_top = gmsh.model.geo.addPoint(L, H, 0, lc)
    p_outlet_bot = gmsh.model.geo.addPoint(L, 0, 0, lc)

    bottom_points = [p_inlet_bot]
    p_start = gmsh.model.geo.addPoint(x_start, 0, 0, lc)
    bottom_points.append(p_start)

    # --- Ступени ВВЕРХ ---
    n_seg_front = frontSteps + 1
    dx_f = (x_peak - x_start) / n_seg_front
    dy_f = y_peak / n_seg_front

    cur_x = x_start
    cur_y = 0.0

    for i in range(1, n_seg_front + 1):
        cur_y += dy_f
        if i == n_seg_front: cur_y = y_peak
        p_up = gmsh.model.geo.addPoint(cur_x, cur_y, 0, lc)
        bottom_points.append(p_up)
        
        cur_x += dx_f
        if i == n_seg_front: cur_x = x_peak
        p_right = gmsh.model.geo.addPoint(cur_x, cur_y, 0, lc)
        bottom_points.append(p_right)

    # --- Ступени ВНИЗ ---
    n_seg_back = backSteps + 1
    dx_b = (x_end - x_peak) / n_seg_back
    dy_b = y_peak / n_seg_back 

    for i in range(1, n_seg_back + 1):
        cur_x += dx_b
        if i == n_seg_back: cur_x = x_end
        p_right = gmsh.model.geo.addPoint(cur_x, cur_y, 0, lc)
        bottom_points.append(p_right)
        
        target_y = y_peak - i * dy_b
        if i == n_seg_back: target_y = 0.0
        p_down = gmsh.model.geo.addPoint(cur_x, target_y, 0, lc)
        bottom_points.append(p_down)
        cur_y = target_y

    bottom_points.append(p_outlet_bot)

    # 4. Линии и Поверхности
    walls_lines = []
    
    for k in range(len(bottom_points) - 1):
        l = gmsh.model.geo.addLine(bottom_points[k], bottom_points[k+1])
        walls_lines.append(l)

    l_outlet = gmsh.model.geo.addLine(p_outlet_bot, p_outlet_top)
    l_top = gmsh.model.geo.addLine(p_outlet_top, p_inlet_top)
    l_inlet = gmsh.model.geo.addLine(p_inlet_top, p_inlet_bot)

    curve_loop = gmsh.model.geo.addCurveLoop(walls_lines + [l_outlet, l_top, l_inlet])
    plane_surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    gmsh.model.geo.synchronize()

    # 5. Physical Groups 
    gmsh.model.addPhysicalGroup(1, [l_inlet], 1, "Inlet")
    gmsh.model.addPhysicalGroup(1, [l_outlet], 2, "Outlet")
    
    # Разделяем стены на Низ (3) и Верх (4)
    gmsh.model.addPhysicalGroup(1, walls_lines, 3, "Bottom") # Только зигзаг снизу
    gmsh.model.addPhysicalGroup(1, [l_top], 4, "Top")        # Только потолок
    
    # Жидкость теперь Tag 5
    gmsh.model.addPhysicalGroup(2, [plane_surface], 5, "FluidDomain")

    # 6. Генерация и Сохранение
    gmsh.model.mesh.generate(2)
    for _ in range(3):
        gmsh.model.mesh.optimize("Laplace2D")

    gmsh.write(filename)
    gmsh.finalize()
    
    return {
        "id": geom_id,
        "scale": round(scale, 5),
        "L": round(L, 4),
        "H": round(H, 4),
        "frontSteps": frontSteps,
        "backSteps": backSteps,
        "x_start": round(x_start, 4),
        "y_peak": round(y_peak, 4)
    }

def main():
    OUTPUT_DIR = "test_domains"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    csv_file = os.path.join(OUTPUT_DIR, "geometry_params.csv")
    keys = ["id", "scale", "L", "H", "frontSteps", "backSteps", "x_start", "y_peak"]
    
    all_params = []
    
    print(f"Генерация 1000 геометрий в папку '{OUTPUT_DIR}'...")
    
    for i in range(1, 101): # Генерируем с 0 до 99
        filename = os.path.join(OUTPUT_DIR, f"step_{i}.msh")
        # seed=i гарантирует воспроизводимость
        params = generate_step_mesh(filename, geom_id=i, seed=i)
        
        all_params.append(params)
        if i % 10 == 0:
            print(f"  [OK] step_{i}.msh generated")

    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_params)
        
    print(f"\nГотово! Параметры сохранены в {csv_file}")

if __name__ == "__main__":
    main()