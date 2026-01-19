#!/usr/bin/env python3
"""
generate_sev_w_hole_geom.py

Генерирует семиугольники (7 углов) с параметрами
Каждый с СЛУЧАЙНОЙ дыркой (смещение центра + безопасный радиус)
Сохраняет в sev_1.msh, sev_2.msh, ..., sev_n.msh и параметры в CSV.

Семиугольник состоит из:
1. Левая нижняя точка (x1, y1)
2. Правая нижняя точка (x2, y2) 
3. Правая верхняя точка (x3, y3)
4. Левая верхняя точка (x4, y4)
5. Центральная верхняя точка (x5, y5) - между двумя верхними, немного выше
6. Левая боковая точка (x6, y6) - между двумя левыми, левее
7. Правая боковая точка (x7, y7) - между двумя правыми, правее
"""

import gmsh
import numpy as np
import os
import sys
import csv

# Параметры для семиугольников
def generate_sev_parameters(n_geoms=30, seed=42):
    """
    Генерирует параметры семиугольников
    """
    np.random.seed(seed)
    
    num = n_geoms
    
    # Координаты левой верхней точки
    leftpoint_x = np.random.randint(0, 21, num) / 10.0      # 0.0 - 2.0
    leftpoint_y = np.random.randint(40, 55, num) / 10.0     # 4.0 - 6.0
    
    # Координаты правой верхней точки
    rightpoint_x = np.random.randint(80, 101, num) / 10.0   # 8.0 - 10.0
    rightpoint_y = np.random.randint(40, 55, num) / 10.0    # 4.0 - 6.0
    
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
        
        # 5. Центральная верхняя точка - между двумя верхними, немного выше
        x_center_top = (x_left_up + x_right_up) / 2
        y_center_top = max(y_left_up, y_right_up) + np.random.uniform(0.1, 0.5) * scaled[i]
        
        # 6. Левая боковая точка - между двумя левыми, левее
        y_left_mid = (y_down + y_left_up) / 2
        x_left_mid = min(x_left_down, x_left_up)- np.random.uniform(0.1, 0.5) * scaled[i]
        
        # 7. Правая боковая точка - между двумя правыми, правее
        y_right_mid = (y_down + y_right_up) / 2
        x_right_mid = max(x_right_down, x_right_up) + np.random.uniform(0.1, 0.5) * scaled[i]
        
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
            "x5": x_center_top,     # центральная верхняя X
            "y5": y_center_top,     # центральная верхняя Y
            "x6": x_left_mid,       # левая боковая X
            "y6": y_left_mid,       # левая боковая Y
            "x7": x_right_mid,      # правая боковая X
            "y7": y_right_mid,      # правая боковая Y
            "scale": scaled[i],
        }
        geoms.append(geom)
    
    return geoms


def calculate_safe_hole_params(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7,
                              hole_offset_ratio=0.5, hole_size_ratio=0.25):
    """
    Вычисляет параметры дырки для семиугольника
    """
    hole_size_ratio = np.random.uniform (0.4, 0.5)
    hole_offset_ratio = np.random.uniform (0.1, 0.2)
    # Центр семиугольника
    cx_center = (x1 + x2 + x3 + x4 + x5 + x6 + x7) / 7
    cy_center = (y1 + y2 + y3 + y4 + y5 + y6 + y7) / 7
    
    # Расстояния до всех сторон (приблизительно)
    dist_left = min(abs(cx_center - x1), abs(cx_center - x4), abs(cx_center - x6))
    dist_right = min(abs(cx_center - x2), abs(cx_center - x3), abs(cx_center - x7))
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
        abs(cx - x1), abs(cx - x2), abs(cx - x3), abs(cx - x4),
        abs(cx - x6), abs(cx - x7), abs(cy - y1), abs(cy - y2), abs(cy - y3), abs(cy - y4)
    )
    
    r = min(r_safe, 0.95 * dist_to_boundary_adjusted)
    # print(f"{min_dist, r_safe, r} - min_dist, r_safe, r")
    return cx, cy, r


def create_sev_with_random_hole(geom_params, output_path, mesh_size=0.25):
    """
    Создаёт семиугольник с СЛУЧАЙНОЙ дыркой
    """
    
    x1, y1 = geom_params["x1"], geom_params["y1"]
    x2, y2 = geom_params["x2"], geom_params["y2"]
    x3, y3 = geom_params["x3"], geom_params["y3"]
    x4, y4 = geom_params["x4"], geom_params["y4"]
    x5, y5 = geom_params["x5"], geom_params["y5"]
    x6, y6 = geom_params["x6"], geom_params["y6"]
    x7, y7 = geom_params["x7"], geom_params["y7"]
    geom_id = geom_params["id"]
    
    print(f"\n{'='*60}")
    print(f"Геометрия {geom_id}")
    print(f"{'='*60}")
    
    # Вычислить параметры дырки
    cx, cy, r = calculate_safe_hole_params(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7)
    
    print(f"Семиугольник (нижние точки y=0):")
    print(f"  1. Нижняя левая: ({x1:.3f}, {y1:.3f})")
    print(f"  2. Нижняя правая: ({x2:.3f}, {y2:.3f})")
    print(f"  3. Верхняя правая: ({x3:.3f}, {y3:.3f})")
    print(f"  4. Верхняя левая: ({x4:.3f}, {y4:.3f})")
    print(f"  5. Центральная верхняя: ({x5:.3f}, {y5:.3f})")
    print(f"  6. Левая боковая: ({x6:.3f}, {y6:.3f})")
    print(f"  7. Правая боковая: ({x7:.3f}, {y7:.3f})")
    
    print(f"\nДырка (СЛУЧАЙНАЯ):")
    print(f"  Центр: ({cx:.3f}, {cy:.3f})")
    print(f"  Радиус: {r:.3f}")
    
    # Инициализировать Gmsh
    gmsh.initialize()
    gmsh.model.add(f"sev_{geom_id}")
    
    # ===== СОЗДАТЬ СЕМИУГОЛЬНИК =====
    
    pts = [
        gmsh.model.geo.addPoint(x1, y1, 0, mesh_size),   # 1. Нижняя левая
        gmsh.model.geo.addPoint(x2, y2, 0, mesh_size),   # 2. Нижняя правая
        gmsh.model.geo.addPoint(x7, y7, 0, mesh_size),   # 7. Правая боковая
        gmsh.model.geo.addPoint(x3, y3, 0, mesh_size),   # 3. Верхняя правая
        gmsh.model.geo.addPoint(x5, y5, 0, mesh_size),   # 5. Центральная верхняя
        gmsh.model.geo.addPoint(x4, y4, 0, mesh_size),   # 4. Верхняя левая
        gmsh.model.geo.addPoint(x6, y6, 0, mesh_size),   # 6. Левая боковая
    ]
    
    lines = [
        gmsh.model.geo.addLine(pts[0], pts[1]),  # нижняя грань (1→2)
        gmsh.model.geo.addLine(pts[1], pts[2]),  # правый нижний склон (2→7)
        gmsh.model.geo.addLine(pts[2], pts[3]),  # правый боковой склон (7→3)
        gmsh.model.geo.addLine(pts[3], pts[4]),  # верхняя правая грань (3→5)
        gmsh.model.geo.addLine(pts[4], pts[5]),  # верхняя левая грань (5→4)
        gmsh.model.geo.addLine(pts[5], pts[6]),  # левый верхний склон (4→6)
        gmsh.model.geo.addLine(pts[6], pts[0]),  # левый нижний склон (6→1)
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
    print("ГЕНЕРИРОВАНИЕ СЕМИУГОЛЬНИКОВ С ПАРАМЕТРАМИ")
    print("="*60)
    
    # Создать папку для MSH файлов
    output_dir = "test_sev_part_obj"
    os.makedirs(output_dir, exist_ok=True)
    
    # Генерировать параметры для геометрий
    print("\n1  Генерирую параметры семиугольников...")
    geoms = generate_sev_parameters(n_geoms=30, seed=42)
    
    # Для каждой геометрии создать MSH с СЛУЧАЙНОЙ дыркой
    print("\n  Создаю MSH файлы с дырками...")
    
    results = []
    for geom in geoms:
        output_path = os.path.join(output_dir, f"sev_{geom['id']}.msh")
        result = create_sev_with_random_hole(geom, output_path)
        results.append(result)
    
    # === СОХРАНЕНИЕ ПАРАМЕТРОВ В CSV ===
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
            
    # Итоги
    print("\n" + "="*60)
    print("ИТОГИ")
    print("="*60)
    
    print(f"\n Создано {len(results)} семиугольников:\n")
    
    for r in results:
        print(f"  Семиугольник {r['id']}:")
        print(f"    Файл: {r['file']}")
        print(f"    Вершин: {r['n_nodes']}, Треугольников: {r['n_triangles']}")
        cx, cy, r_hole = r['hole']
        print(f"    Дырка: центр=({cx:.3f}, {cy:.3f}), радиус={r_hole:.3f}")


if __name__ == "__main__":
    # Для отладки вывод в консоль полезен, поэтому devnull закомментирован
    # f = open(os.devnull, 'w')
    # sys.stdout = f
    main()

