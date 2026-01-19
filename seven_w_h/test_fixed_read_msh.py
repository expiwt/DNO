#!/usr/bin/env python3
"""
Тестирование исправленного read_msh.py
"""

import sys
sys.path.append('.')

# Импортируем исправленную функцию
from create_dif_map.read_msh import read_msh_file

# Тестируем на первом файле
import os

test_dir = "test_sev_part_obj"
if os.path.exists(test_dir):
    msh_files = [f for f in os.listdir(test_dir) if f.endswith('.msh')]
    if msh_files:
        filename = os.path.join(test_dir, msh_files[0])
        print(f"Тестируем файл: {filename}")
        
        try:
            X, F, B_outer, B_inner, info = read_msh_file(filename)
            
            print(f"\nРезультаты:")
            print(f"  Всего узлов: {info['n_nodes']}")
            print(f"  Всего треугольников: {info['n_triangles']}")
            print(f"  Узлов на внешней границе: {info['n_outer']}")
            print(f"  Узлов на внутренней границе: {info['n_inner']}")
            print(f"  Центр дырки: {info['center_hole']}")
            print(f"  Радиус дырки: {info['radius_hole']:.6f}")
            
            print(f"\nПервые 5 узлов внутренней границы (индексы): {B_inner[:5]}")
            print(f"Координаты первых 5 узлов внутренней границы:")
            for i in range(min(5, len(B_inner))):
                idx = B_inner[i]
                x, y = X[idx]
                print(f"  Узел {idx}: ({x:.4f}, {y:.4f})")
            
            print(f"\nПервые 5 узлов внешней границы (индексы): {B_outer[:5]}")
            print(f"Координаты первых 5 узлов внешней границы:")
            for i in range(min(5, len(B_outer))):
                idx = B_outer[i]
                x, y = X[idx]
                print(f"  Узел {idx}: ({x:.4f}, {y:.4f})")
            
            # Проверим, что внутренняя граница действительно круг
            print(f"\nПроверка формы внутренней границы:")
            inner_coords = X[B_inner]
            center = info['center_hole']
            distances = []
            for i, (x, y) in enumerate(inner_coords):
                dist = ((x - center[0])**2 + (y - center[1])**2)**0.5
                distances.append(dist)
                if i < 5:
                    print(f"  Узел {i}: расстояние до центра = {dist:.4f}")
            
            print(f"  Среднее расстояние: {sum(distances)/len(distances):.4f}")
            print(f"  Минимальное расстояние: {min(distances):.4f}")
            print(f"  Максимальное расстояние: {max(distances):.4f}")
            
        except Exception as e:
            print(f"Ошибка: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"В папке {test_dir} нет MSH файлов")
else:
    print(f"Папка {test_dir} не существует")
    
    # Попробуем другую папку
    train_dir = "train_sev_part_obj"
    if os.path.exists(train_dir):
        msh_files = [f for f in os.listdir(train_dir) if f.endswith('.msh')]
        if msh_files:
            filename = os.path.join(train_dir, msh_files[0])
            print(f"Тестируем файл: {filename}")
            
            try:
                X, F, B_outer, B_inner, info = read_msh_file(filename)
                
                print(f"\nРезультаты:")
                print(f"  Всего узлов: {info['n_nodes']}")
                print(f"  Всего треугольников: {info['n_triangles']}")
                print(f"  Узлов на внешней границе: {info['n_outer']}")
                print(f"  Узлов на внутренней границе: {info['n_inner']}")
                print(f"  Центр дырки: {info['center_hole']}")
                print(f"  Радиус дырки: {info['radius_hole']:.6f}")
                
            except Exception as e:
                print(f"Ошибка: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"В папке {train_dir} нет MSH файлов")
    else:
        print(f"Папки {test_dir} и {train_dir} не существуют")