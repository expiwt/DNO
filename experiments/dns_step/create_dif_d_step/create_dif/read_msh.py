import gmsh
import numpy as np

def read_msh_file(filename):
    """
    Читает MSH файл (формат 4.1) для задачи "Ступенька".
    
    Returns:
    X : np.array (N, 2)
        Координаты всех узлов [x, y].
    F : np.array (M, 3)
        Индексы узлов, образующих треугольники (связность).
        Нужно для построения графа Лапласа.
    boundaries : dict
        Словарь списков индексов узлов:
        {
            'inlet':  [idx1, idx2, ...],  # Tag 1
            'outlet': [idx1, idx2, ...],  # Tag 2
            'bottom': [idx1, idx2, ...],  # Tag 3 (Пол + Ступени)
            'top':    [idx1, idx2, ...]   # Tag 4 (Потолок)
        }
    """
    gmsh.initialize()
    try:
        gmsh.open(filename)
        
        # 1. Читаем ВСЕ узлы
        # getNodes возвращает (nodeTags, nodeCoords, parametricCoord)
        # nodeCoords это плоский список [x1, y1, z1, x2, y2, z2, ...]
        nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
        
        # Превращаем координаты в массив (N, 3) и отбрасываем Z (N, 2)
        X = np.array(nodeCoords).reshape(-1, 3)[:, :2]
        
        # Важно: Gmsh теги (ID) могут быть любыми (не обязательно 0, 1, 2...).
        # А Python массивы требуют индексов 0..N-1.
        # Создаем карту: GmshTag -> PythonIndex
        tag2idx = {tag: i for i, tag in enumerate(nodeTags)}
        
        # 2. Читаем треугольники (Элементы типа 2)
        # Это нужно, чтобы знать, какие узлы соединены (для Лапласиана)
        # Берем все 2D элементы
        elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(2)
        
        # elemNodeTags - плоский список. Решейпим в (M, 3)
        if len(elemNodeTags) > 0:
            F_tags = np.array(elemNodeTags).reshape(-1, 3)
            # Переводим теги Gmsh в наши индексы Python
            # Используем векторизацию словаря для скорости
            vec_map = np.vectorize(tag2idx.get)
            F = vec_map(F_tags)
        else:
            F = np.empty((0, 3), dtype=int)
            print(f"Warning: No triangles found in {filename}!")

        # 3. Читаем границы по Physical Groups
        # Мы знаем наши ID из генератора:
        # 1=Inlet, 2=Outlet, 3=Bottom, 4=Top
        
        boundaries = {}
        target_groups = {
            'inlet': 1,
            'outlet': 2,
            'bottom': 3,
            'top': 4
        }
        
        for name, tag_id in target_groups.items():
            # Получаем узлы для физической группы (dim=1, так как это линии)
            try:
                bnodes_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, tag_id)
                
                if len(bnodes_tags) == 0:
                    print(f"Warning: Boundary '{name}' (Tag {tag_id}) is empty in {filename}")
                    boundaries[name] = []
                else:
                    # Конвертируем теги в индексы и сохраняем
                    boundaries[name] = [tag2idx[t] for t in bnodes_tags]
                    
            except Exception as e:
                # Если группы нет в файле
                print(f"Error reading group {name}: {e}")
                boundaries[name] = []

    except Exception as e:
        print(f"Critical error reading MSH: {e}")
        return None, None, None
        
    finally:
        gmsh.finalize()
    
    return X, F, boundaries

# --- Блок для проверки (можно запустить скрипт отдельно) ---
if __name__ == "__main__":
    # Тест на первом файле
    test_file = "train_domains_w_steps/step_0.msh"
    import os
    if os.path.exists(test_file):
        X, F, bnds = read_msh_file(test_file)
        if X is not None:
            print(f"Успешно прочитан {test_file}")
            print(f"Всего узлов: {X.shape[0]}")
            print(f"Всего треугольников: {F.shape[0]}")
            for name, nodes in bnds.items():
                print(f"  Граница '{name}': {len(nodes)} узлов")
    else:
        print(f"Файл {test_file} не найден. Сначала запусти генератор!")