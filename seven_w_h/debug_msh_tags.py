#!/usr/bin/env python3
"""
Скрипт для отладки тегов в MSH файле
"""

import numpy as np

def debug_msh_tags(filename: str):
    """
    Отладочная функция для просмотра тегов в MSH файле
    """
    print(f"\n{'='*70}")
    print(f"DEBUG: Анализ тегов в файле {filename}")
    print(f"{'='*70}")
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Найдем секцию $Elements
    elem_start = None
    elem_end = None
    for i, line in enumerate(lines):
        if line.strip() == "$Elements":
            elem_start = i
        if line.strip() == "$EndElements":
            elem_end = i
            break
    
    if elem_start is None or elem_end is None:
        print("ERROR: Не найдена секция $Elements")
        return
    
    # Читаем заголовок
    header = lines[elem_start + 1].split()
    num_entity_blocks = int(header[0])
    num_elements_total = int(header[1])
    
    print(f"Количество entity blocks: {num_entity_blocks}")
    print(f"Всего элементов: {num_elements_total}")
    
    idx = elem_start + 2
    entity_stats = {}
    
    for entity_block in range(num_entity_blocks):
        parts = lines[idx].split()
        entity_dim = int(parts[0])
        entity_tag = int(parts[1])
        element_type = int(parts[2])
        num_elements_in_entity = int(parts[3])
        
        print(f"\nEntity block {entity_block}:")
        print(f"  dim={entity_dim}, tag={entity_tag}, type={element_type}, count={num_elements_in_entity}")
        
        # Сохраняем статистику
        key = (entity_dim, entity_tag, element_type)
        if key not in entity_stats:
            entity_stats[key] = 0
        entity_stats[key] += num_elements_in_entity
        
        idx += 1
        
        # Пропускаем элементы (не читаем их содержимое для простоты)
        for _ in range(num_elements_in_entity):
            idx += 1
    
    print(f"\n{'='*70}")
    print("Сводка по тегам:")
    print(f"{'='*70}")
    
    for (dim, tag, etype), count in entity_stats.items():
        element_name = "Unknown"
        if etype == 1:
            element_name = "Line (1D)"
        elif etype == 2:
            element_name = "Triangle (2D)"
        elif etype == 15:
            element_name = "Point (0D)"
        
        boundary_type = "Unknown"
        if dim == 2 and tag == 1:
            boundary_type = "Domain (поверхность)"
        elif dim == 1 and tag in [1, 2, 3, 4, 5, 6]:
            boundary_type = "OuterBoundary (предположительно)"
        elif dim == 1 and tag in [7, 8]:
            boundary_type = "InnerBoundary (предположительно)"
        
        print(f"  dim={dim}, tag={tag}, type={etype} ({element_name}): {count} элементов - {boundary_type}")

if __name__ == "__main__":
    # Протестируем на первом файле
    import os
    
    # Найдем первый MSH файл в папке test_sev_part_obj
    test_dir = "test_sev_part_obj"
    if os.path.exists(test_dir):
        msh_files = [f for f in os.listdir(test_dir) if f.endswith('.msh')]
        if msh_files:
            filename = os.path.join(test_dir, msh_files[0])
            debug_msh_tags(filename)
        else:
            print(f"В папке {test_dir} нет MSH файлов")
    else:
        print(f"Папка {test_dir} не существует")
        
        # Попробуем train_sev_part_obj
        train_dir = "train_sev_part_obj"
        if os.path.exists(train_dir):
            msh_files = [f for f in os.listdir(train_dir) if f.endswith('.msh')]
            if msh_files:
                filename = os.path.join(train_dir, msh_files[0])
                debug_msh_tags(filename)
            else:
                print(f"В папке {train_dir} нет MSH файлов")
        else:
            print(f"Папки {test_dir} и {train_dir} не существуют")
            print("Сначала запустите generate_sev_w_hole_geom.py для генерации файлов")