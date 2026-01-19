#!/usr/bin/env python3
"""

generate_quad_geometries.py

Генерирует 5 четырёхугольников с параметрами из part_generate.py
Каждый с СЛУЧАЙНОЙ дыркой (смещение центра + безопасный радиус)
Сохраняет в quad_1.msh, quad_2.msh, ..., quad_5.msh
"""

import gmsh
import numpy as np
import os
import sys
import csv


# Параметры для 5 геометрий (из part_generate.py логики)
def generate_quad_parameters(n_geoms=30, seed=42):
    """
    Генерирует параметры четырёхугольников
    по логике part_generate.py
    """
    np.random.seed(seed)
    
    num = n_geoms
    
    # Координаты левой верхней точки
    leftpoint_x = np.random.randint(0, 21, num) / 10.0      # 0.0 - 2.0
    leftpoint_y = np.random.randint(40, 61, num) / 10.0     # 4.0 - 6.0
    
    # Координаты правой верхней точки
    rightpoint_x = np.random.randint(80, 101, num) / 10.0   # 8.0 - 10.0
    rightpoint_y = np.random.randint(40, 61, num) / 10.0    # 4.0 - 6.0
    
    # X координаты левой нижней точки
    down_leftpoint_x = np.random.randint(20, 41, num) / 10.0  # 2.0 - 4.0
    
    # X координаты правой нижней точки
    down_rightpoint_x = np.random.randint(60, 81, num) / 10.0 # 6.0 - 8.0
    
    # Масштаб (0.5 - 1.0)
    scaled = 0.5 + 0.5 * np.random.rand(num)
    
    # Формируем параметры для каждой геометрии
    geoms = []
    for i in range(num):
        # Масштабированные координаты
        x_left_up = leftpoint_x[i] * scaled[i]
        y_left_up = leftpoint_y[i] * scaled[i]
        
        x_right_up = rightpoint_x[i] * scaled[i]
        y_right_up = rightpoint_y[i] * scaled[i]
        
        x_left_down = down_leftpoint_x[i] * scaled[i]
        x_right_down = down_rightpoint_x[i] * scaled[i]
        
        # Y нижних точек всегда 0 (как в задании)
        y_down = 0.0
        
        geom = {
            "id": i + 1,
            "x1": x_left_down,      # левая нижняя X
            "y1": y_down,           # левая нижняя Y (всегда 0)
            "x2": x_right_down,     # правая нижняя X
            "y2": y_down,           # правая нижняя Y (всегда 0)
            "x3": x_right_up,       # правая верхняя X
            "y3": y_right_up,       # правая верхняя Y
            "x4": x_left_up,        # левая верхняя X
            "y4": y_left_up,        # левая верхняя Y
            "scale": scaled[i],
        }
        geoms.append(geom)
    
    return geoms

def calculate_safe_hole_params(x1, y1, x2, y2, x3, y3, x4, y4, 
                              hole_offset_ratio=0.2, hole_size_ratio=0.25):
    """
    Вычисляет параметры дырки для четырёхугольника
    
    Args:
        x1, y1, x2, y2, x3, y3, x4, y4: координаты четырёх вершин
        hole_offset_ratio: как далеко смещение от центра (0.2 = 20%)
        hole_size_ratio: размер дырки как % от min_dist (0.25 = 25%)
    
    Returns:
        cx, cy, r: центр и радиус дырки
    """
    hole_offset_ratio = np.random.uniform (0.2, 0.6)
    
    # Центр четырёхугольника
    cx_center = (x1 + x2 + x3 + x4) / 4
    cy_center = (y1 + y2 + y3 + y4) / 4
    
    # Расстояния до всех сторон (приблизительно)
    dist_left = min(abs(cx_center - x1), abs(cx_center - x4))
    dist_right = min(abs(cx_center - x2), abs(cx_center - x3))
    dist_bottom = min(abs(cy_center - y1), abs(cy_center - y2))
    dist_top = min(abs(cy_center - y3), abs(cy_center - y4))
    
    min_dist = min(dist_left, dist_right, dist_bottom, dist_top)
    
    # Безопасный радиус дырки
    r_safe = hole_size_ratio * min_dist
    
    # СЛУЧАЙНОЕ СМЕЩЕНИЕ центра дырки
    # Максимальное смещение = hole_offset_ratio * min_dist
    max_offset = hole_offset_ratio * min_dist
    
    # Случайные смещения (может быть в любую сторону)
    offset_x = np.random.uniform(-max_offset, max_offset)
    offset_y = np.random.uniform(-max_offset, max_offset)
    
    cx = cx_center + offset_x
    cy = cy_center + offset_y
    
    # Проверка: дырка не выходит за пределы
    # (пересчитаем радиус если нужно)
    dist_to_boundary_adjusted = min(
        abs(cx - x1), abs(cx - x2),  # расстояния по X
        abs(cy - y1), abs(cy - y3)   # расстояния по Y
    )
    
    r = min(r_safe, 0.95 * dist_to_boundary_adjusted)
    
    return cx, cy, r

def create_quad_with_random_hole(geom_params, output_path, mesh_size=0.25):
    """
    Создаёт четырёхугольник с СЛУЧАЙНОЙ дыркой
    
    Args:
        geom_params: словарь с координатами {x1, y1, x2, y2, x3, y3, x4, y4, id}
        output_path: путь для сохранения .msh
        mesh_size: размер элемента сетки
    """
    
    x1, y1 = geom_params["x1"], geom_params["y1"]
    x2, y2 = geom_params["x2"], geom_params["y2"]
    x3, y3 = geom_params["x3"], geom_params["y3"]
    x4, y4 = geom_params["x4"], geom_params["y4"]
    geom_id = geom_params["id"]
    
    print(f"\n{'='*60}")
    print(f"Геометрия {geom_id}")
    print(f"{'='*60}")
    
    # Вычислить параметры дырки
    cx, cy, r = calculate_safe_hole_params(x1, y1, x2, y2, x3, y3, x4, y4)
    
    print(f"Четырёхугольник (нижние точки y=0):")
    print(f"  Нижняя левая: ({x1:.3f}, {y1:.3f})")
    print(f"  Нижняя правая: ({x2:.3f}, {y2:.3f})")
    print(f"  Верхняя правая: ({x3:.3f}, {y3:.3f})")
    print(f"  Верхняя левая: ({x4:.3f}, {y4:.3f})")
    
    print(f"\nДырка (СЛУЧАЙНАЯ):")
    print(f"  Центр: ({cx:.3f}, {cy:.3f})")
    print(f"  Радиус: {r:.3f}")
    
    # Инициализировать Gmsh
    gmsh.initialize()
    gmsh.model.add(f"quad_{geom_id}")
    
    # ===== СОЗДАТЬ ЧЕТЫРЁХУГОЛЬНИК =====
    
    pts = [
        gmsh.model.geo.addPoint(x1, y1, 0, mesh_size),
        gmsh.model.geo.addPoint(x2, y2, 0, mesh_size),
        gmsh.model.geo.addPoint(x3, y3, 0, mesh_size),
        gmsh.model.geo.addPoint(x4, y4, 0, mesh_size),
    ]
    
    lines = [
        gmsh.model.geo.addLine(pts[0], pts[1]),  # нижнее
        gmsh.model.geo.addLine(pts[1], pts[2]),  # правое
        gmsh.model.geo.addLine(pts[2], pts[3]),  # верхнее
        gmsh.model.geo.addLine(pts[3], pts[0]),  # левое
    ]
    
    outer_loop = gmsh.model.geo.addCurveLoop(lines)
    
    # ===== СОЗДАТЬ ДЫРКУ =====
    
    center = gmsh.model.geo.addPoint(cx, cy, 0, mesh_size)
    
    circle_pt_right = gmsh.model.geo.addPoint(cx + r, cy, 0, mesh_size)
    circle_pt_left = gmsh.model.geo.addPoint(cx - r, cy, 0, mesh_size)
    
    # Две полудуги
    arc_top = gmsh.model.geo.addCircleArc(circle_pt_right, center, circle_pt_left)
    arc_bottom = gmsh.model.geo.addCircleArc(circle_pt_left, center, circle_pt_right)
    
    hole_loop = gmsh.model.geo.addCurveLoop([arc_top, arc_bottom])
    
    # ===== ПОВЕРХНОСТЬ =====
    
    surface = gmsh.model.geo.addPlaneSurface([outer_loop, hole_loop])
    
    gmsh.model.geo.synchronize()
    
    # ===== ФИЗИЧЕСКИЕ ГРУППЫ =====
    
    gmsh.model.addPhysicalGroup(2, [surface], name="Domain")
    gmsh.model.addPhysicalGroup(1, lines, name="OuterBoundary")
    gmsh.model.addPhysicalGroup(1, [arc_top, arc_bottom], name="InnerBoundary")
    
    # ===== СЕТКА =====
    
    gmsh.option.setNumber("Mesh.Algorithm", 6)                      # Delaunay 2D алгоритм
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.25 * geom_params["scale"])  # Адаптивный размер

    # Опции для оптимизации качества сетки
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)                 # Оптимизация Netgen (улучшает углы)
    gmsh.option.setNumber("Mesh.Smoothing", 100)                    # Сглаживание (100 итераций)

    # Параметры для лучшего качества
    gmsh.option.setNumber("Mesh.MinimumCirclePoints", 12)           # Минимум точек на окружности
    gmsh.option.setNumber("Mesh.MinimumCurvePoints", 2)             # Минимум точек на кривой

    # Теперь генерируем сетку
    gmsh.model.mesh.generate(2)
        
    # Информация о сетке
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, node_connectivity = gmsh.model.mesh.getElements()
    
    n_nodes = len(node_tags)
    n_triangles = len(node_connectivity[0]) // 3 if node_connectivity else 0
    
    print(f"\nСетка: {n_nodes} вершин, {n_triangles} треугольников")
    
    # ===== СОХРАНЕНИЕ =====
    
    gmsh.write(output_path)
    gmsh.finalize()
    
    print(f"✓ Сохранено: {output_path}")
    
    return {
        "id": geom_id,
        "file": output_path,
        "n_nodes": n_nodes,
        "n_triangles": n_triangles,
        "hole": (cx, cy, r)
    }

def main():
    """Главная функция"""
    
    print("\n" + "="*60)
    print("ГЕНЕРИРОВАНИЕ ГЕОМЕТРИЙ С ПАРАМЕТРАМИ")
    print("="*60)
    
    # Создать папку для MSH файлов
    output_dir = "train_sq_part_obj"
    os.makedirs(output_dir, exist_ok=True)
    
    # Генерировать параметры для 5 геометрий
    print("\n1  Генерирую параметры геометрий...")
    geoms = generate_quad_parameters(n_geoms=1000, seed=42)
    
    # Для каждой геометрии создать MSH с СЛУЧАЙНОЙ дыркой
    print("\n  Создаю MSH файлы с дырками...")
    
    results = []
    for geom in geoms:
        output_path = os.path.join(output_dir, f"quad_{geom['id']}.msh")
        result = create_quad_with_random_hole(geom, output_path)
        results.append(result)
            # ... (конец цикла for geom in geoms) ...
    
        # === НОВАЯ ЧАСТЬ: Сохранение параметров в CSV ===
        
        params_csv_path = os.path.join(output_dir, "geometry_params.csv")
        print(f"\nСохраняю параметры в {params_csv_path}...")
        
    with open(params_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Заголовок
            writer.writerow(["id", "scale", "hole_cx", "hole_cy", "hole_r"])
            
            # Данные
            for i, geom in enumerate(geoms):
                # geom содержит 'id' и 'scale'
                # result (из results) содержит параметры дырки
                res = results[i]
                cx, cy, r_hole = res['hole']
                
                writer.writerow([
                    geom['id'], 
                    f"{geom['scale']:.6f}", 
                    f"{cx:.6f}", 
                    f"{cy:.6f}", 
                    f"{r_hole:.6f}"
                ])
            
    
    print(f"\n Создано {len(results)} геометрий:\n")
    
    for r in results:
        print(f"  Геометрия {r['id']}:")
        print(f"    Файл: {r['file']}")
        print(f"    Вершин: {r['n_nodes']}, Треугольников: {r['n_triangles']}")
        cx, cy, r_hole = r['hole']
        print(f"    Дырка: центр=({cx:.3f}, {cy:.3f}), радиус={r_hole:.3f}")

if __name__ == "__main__":

    # Открываем "черную дыру" для данных
    f = open(os.devnull, 'w')

    # Перенаправляем стандартный вывод туда
    sys.stdout = f
    main()
