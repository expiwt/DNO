#!/usr/bin/env python3
"""
Скрипт для детального анализа тегов в MSH файле
"""

import numpy as np

def debug_msh_tags_detailed(filename: str):
    """
    Детальная отладочная функция для просмотра тегов в MSH файле
    """
    print(f"\n{'='*70}")
    print(f"DETAILED DEBUG: Анализ тегов в файле {filename}")
    print(f"{'='*70}")
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Найдем секцию $Nodes
    nodes_start = None
    nodes_end = None
    for i, line in enumerate(lines):
        if line.strip() == "$Nodes":
            nodes_start = i
        if line.strip() == "$EndNodes":
            nodes_end = i
            break
    
    # Создадим mapping индексов
    if nodes_start is not None and nodes_end is not None:
        idx = nodes_start + 2
        header = lines[nodes_start + 1].split()
        num_entity_blocks = int(header[0])
        
        nodes_dict = {}
        for entity_block in range(num_entity_blocks):
            parts = lines[idx].split()
            entity_dim = int(parts[0])
            entity_tag = int(parts[1])
            entity_parametric = int(parts[2])
            num_nodes_in_entity = int(parts[3])
            
            idx += 1
            node_indices = []
            while len(node_indices) < num_nodes_in_entity:
                node_indices.extend(map(int, lines[idx].split()))
                idx += 1
            node_indices = node_indices[:num_nodes_in_entity]
            
            for node_id in node_indices:
                parts = lines[idx].split()
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                nodes_dict[node_id] = (x, y)
                idx += 1
        
        # Создаем mapping
        sorted_indices = sorted(nodes_dict.keys())
        index_mapping = {msh_idx: py_idx for py_idx, msh_idx in enumerate(sorted_indices)}
    
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
    
    # Собираем элементы по тегам
    elements_by_tag = {}
    
    for entity_block in range(num_entity_blocks):
        parts = lines[idx].split()
        entity_dim = int(parts[0])
        entity_tag = int(parts[1])
        element_type = int(parts[2])
        num_elements_in_entity = int(parts[3])
        
        print(f"\nEntity block {entity_block}: dim={entity_dim}, tag={entity_tag}, type={element_type}")
        
        idx += 1
        
        elements = []
        for _ in range(num_elements_in_entity):
            parts = list(map(int, lines[idx].split()))
            element_id = parts[0]
            nodes = parts[1:]
            
            # Конвертируем индексы
            if 'index_mapping' in locals():
                nodes_py = [index_mapping[n] for n in nodes]
                elements.append(nodes_py)
            
            idx += 1
        
        elements_by_tag[entity_tag] = {
            'dim': entity_dim,
            'type': element_type,
            'count': num_elements_in_entity,
            'elements': elements
        }
    
    # Анализируем каждый тег
    print(f"\n{'='*70}")
    print("Детальный анализ тегов:")
    print(f"{'='*70}")
    
    for tag, data in sorted(elements_by_tag.items()):
        dim = data['dim']
        etype = data['type']
        count = data['count']
        elements = data['elements']
        
        element_name = "Unknown"
        if etype == 1:
            element_name = "Line (1D)"
        elif etype == 2:
            element_name = "Triangle (2D)"
        
        print(f"\nТег {tag}: dim={dim}, type={etype} ({element_name}), элементов: {count}")
        
        if elements and len(elements) > 0:
            # Покажем первые 3 элемента
            print(f"  Первые 3 элемента (индексы Python):")
            for i in range(min(3, len(elements))):
                elem = elements[i]
                print(f"    Элемент {i+1}: {elem}")
                
                # Если у нас есть координаты, покажем их
                if 'nodes_dict' in locals() and 'index_mapping' in locals():
                    # Найдем обратный mapping
                    reverse_mapping = {v: k for k, v in index_mapping.items()}
                    print(f"      MSH индексы: {[reverse_mapping[n] for n in elem]}")
                    
                    # Покажем координаты первой точки
                    if elem:
                        msh_idx = reverse_mapping[elem[0]]
                        x, y = nodes_dict[msh_idx]
                        print(f"      Координаты первой точки: ({x:.4f}, {y:.4f})")
            
            # Для тега 9 покажем все элементы
            if tag == 9:
                print(f"  \n  ВСЕ элементы тега 9:")
                for i, elem in enumerate(elements):
                    print(f"    Элемент {i+1}: {elem}")
                    if 'nodes_dict' in locals() and 'index_mapping' in locals():
                        reverse_mapping = {v: k for k, v in index_mapping.items()}
                        msh_idx = reverse_mapping[elem[0]]
                        x, y = nodes_dict[msh_idx]
                        print(f"      Координаты: ({x:.4f}, {y:.4f})")

if __name__ == "__main__":
    # Протестируем на первом файле
    import os
    
    # Найдем первый MSH файл в папке test_sev_part_obj
    test_dir = "test_sev_part_obj"
    if os.path.exists(test_dir):
        msh_files = [f for f in os.listdir(test_dir) if f.endswith('.msh')]
        if msh_files:
            filename = os.path.join(test_dir, msh_files[0])
            debug_msh_tags_detailed(filename)
        else:
            print(f"В папке {test_dir} нет MSH файлов")
    else:
        print(f"Папка {test_dir} не существует")