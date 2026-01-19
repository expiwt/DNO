#!/usr/bin/env python3
"""
Тестирование улучшенного алгоритма сопоставления точек на дырке
"""

import sys
import os
sys.path.append('.')

from create_dif_map.read_msh import read_msh_file
from create_dif_map.create_map import DiffeomorphismMapperHeptagon

# Тестируем на первом файле
test_dir = "test_sev_part_obj"
if os.path.exists(test_dir):
    msh_files = [f for f in os.listdir(test_dir) if f.endswith('.msh')]
    if msh_files:
        filename = os.path.join(test_dir, msh_files[0])
        print(f"Тестируем файл: {filename}")
        print(f"{'='*70}")
        
        try:
            # Читаем файл
            X, F, B_outer, B_inner, info = read_msh_file(filename)
            
            print(f"\nИнформация о сетке:")
            print(f"  Всего узлов: {info['n_nodes']}")
            print(f"  Всего треугольников: {info['n_triangles']}")
            print(f"  Узлов на внешней границе: {info['n_outer']}")
            print(f"  Узлов на внутренней границе: {info['n_inner']}")
            print(f"  Центр дырки: {info['center_hole']}")
            print(f"  Радиус дырки: {info['radius_hole']:.6f}")
            
            # Создаем маппер
            mapper = DiffeomorphismMapperHeptagon(X, F, B_outer, B_inner)
            
            # Вычисляем параметры дырки
            center, radius = mapper.compute_hole_parameters()
            print(f"\nВычисленные параметры дырки:")
            print(f"  Центр: {center}")
            print(f"  Радиус: {radius:.6f}")
            
            # Находим соответствие углов
            corners = mapper.find_corner_correspondence()
            print(f"\nСоответствие углов:")
            print(f"  Индексы углов: {corners}")
            print(f"  Координаты углов:")
            for i, idx in enumerate(corners):
                x, y = X[idx]
                print(f"    Угол {i} (узел {idx}): ({x:.4f}, {y:.4f})")
            
            # Находим соответствие точек на дырке (УЛУЧШЕННЫЙ АЛГОРИТМ)
            print(f"\n{'='*70}")
            print("НАХОЖДЕНИЕ ТОЧЕК НА ДЫРКЕ (улучшенный алгоритм):")
            print(f"{'='*70}")
            
            hole_indices, hole_coords = mapper.find_hole_boundary_correspondence()
            
            print(f"\nРезультаты:")
            print(f"  Выбранные точки на дырке: {hole_indices}")
            print(f"  Их координаты:")
            for i, (idx, coord) in enumerate(zip(hole_indices, hole_coords)):
                print(f"    Точка {i} (узел {idx}): ({coord[0]:.4f}, {coord[1]:.4f})")
            
            # Проверяем расстояния между выбранными точками
            print(f"\nПроверка физических расстояний:")
            for i in range(len(hole_indices)):
                for j in range(i+1, len(hole_indices)):
                    idx_i = hole_indices[i]
                    idx_j = hole_indices[j]
                    coord_i = X[idx_i]
                    coord_j = X[idx_j]
                    distance = np.linalg.norm(coord_i - coord_j)
                    print(f"  Расстояние между точками {i} и {j}: {distance:.4f}")
            
            # Параметризуем внутреннюю границу
            print(f"\n{'='*70}")
            print("ПАРАМЕТРИЗАЦИЯ ВНУТРЕННЕЙ ГРАНИЦЫ:")
            print(f"{'='*70}")
            
            inner_params = mapper._parametrize_inner_boundary()
            
            if inner_params is not None:
                print(f"\nПервые 10 точек параметризации:")
                for i in range(min(10, len(inner_params))):
                    node_idx = mapper.B_inner[i]
                    phys_coord = X[node_idx]
                    target_coord = inner_params[i]
                    print(f"  Узел {node_idx}: физич. ({phys_coord[0]:.4f}, {phys_coord[1]:.4f}) -> целевое ({target_coord[0]:.4f}, {target_coord[1]:.4f})")
                
                # Проверяем, что все целевые точки находятся на окружности радиусом 0.1
                print(f"\nПроверка целевых координат:")
                center_target = np.array([0.5, 0.5])
                for i in range(len(inner_params)):
                    dist = np.linalg.norm(inner_params[i] - center_target)
                    if abs(dist - 0.1) > 0.01:
                        print(f"  ВНИМАНИЕ: точка {i} на расстоянии {dist:.4f} от центра (ожидается 0.1)")
                
                print(f"\n✓ Параметризация успешно завершена!")
            
        except Exception as e:
            print(f"Ошибка: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"В папке {test_dir} нет MSH файлов")
else:
    print(f"Папка {test_dir} не существует")