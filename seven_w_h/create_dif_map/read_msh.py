import numpy as np
from typing import Tuple, List, Dict
import math


def read_msh_file(filename: str) -> Tuple[np.ndarray, np.ndarray, List[int], List[int], Dict]:
    """
    Читает MSH 4.1 файл и возвращает триангуляцию с границами.
    
    Parameters
    ----------
    filename : str
        Путь к MSH файлу
    
    Returns
    -------
    X : (n_nodes, 2) ndarray
        Координаты узлов [x, y] (0-indexed)
    
    F : (n_triangles, 3) ndarray
        Индексы вершин треугольников (0-indexed)
    
    B_outer : List[int]
        Индексы узлов на внешней границе в циклическом порядке
    
    B_inner : List[int]
        Индексы узлов на внутренней границе в циклическом порядке
    
    info : Dict
        Дополнительная информация:
        - 'center_hole': центр дыры [x, y]
        - 'radius_hole': радиус дыры (среднее расстояние)
        - 'n_nodes': всего узлов
        - 'n_triangles': всего треугольников
        - 'n_outer': узлов на внешней границе
        - 'n_inner': узлов на внутренней границе
    """
    
    # print(f"\n{'='*70}")
    # print(f"Reading MSH file: {filename}")
    # print(f"{'='*70}")
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # ========================================================================
    # STEP 1: Parse $Nodes section
    # ========================================================================
    # print("\n[STEP 1] Parsing $Nodes section...")
    
    nodes_start = None
    nodes_end = None
    for i, line in enumerate(lines):
        if line.strip() == "$Nodes":
            nodes_start = i
        if line.strip() == "$EndNodes":
            nodes_end = i
            break
    
    if nodes_start is None or nodes_end is None:
        raise ValueError("Could not find $Nodes section in MSH file")
    
    # Read nodes header
    header = lines[nodes_start + 1].split()
    num_entity_blocks, num_nodes_total = int(header[0]), int(header[1])
    # print(f"  Entity blocks: {num_entity_blocks}, Total nodes: {num_nodes_total}")
    
    # Dictionary to store nodes: msh_index -> (x, y)
    nodes_dict = {}
    
    idx = nodes_start + 2
    for entity_block in range(num_entity_blocks):
        parts = lines[idx].split()
        entity_dim = int(parts[0])  # 0=point, 1=curve, 2=surface
        entity_tag = int(parts[1])
        entity_parametric = int(parts[2])
        num_nodes_in_entity = int(parts[3])
        
        # Read node indices for this entity
        idx += 1
        node_indices = []
        while len(node_indices) < num_nodes_in_entity:
            node_indices.extend(map(int, lines[idx].split()))
            idx += 1
        node_indices = node_indices[:num_nodes_in_entity]
        
        # Read node coordinates
        for node_id in node_indices:
            parts = lines[idx].split()
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            nodes_dict[node_id] = (x, y)
            idx += 1
    
    # Convert to 0-indexed numpy array
    sorted_indices = sorted(nodes_dict.keys())
    index_mapping = {msh_idx: py_idx for py_idx, msh_idx in enumerate(sorted_indices)}
    
    X = np.array([nodes_dict[msh_idx] for msh_idx in sorted_indices], dtype=np.float64)
    # print(f"  Loaded {len(X)} nodes")
    # print(f"  X shape: {X.shape}, bounds: x=[{X[:,0].min():.4f}, {X[:,0].max():.4f}], "
    #       f"y=[{X[:,1].min():.4f}, {X[:,1].max():.4f}]")
    
    # ========================================================================
    # STEP 2: Parse $Elements section
    # ========================================================================
    # print("\n[STEP 2] Parsing $Elements section...")
    
    elem_start = None
    elem_end = None
    for i, line in enumerate(lines):
        if line.strip() == "$Elements":
            elem_start = i
        if line.strip() == "$EndElements":
            elem_end = i
            break
    
    if elem_start is None or elem_end is None:
        raise ValueError("Could not find $Elements section in MSH file")
    
    # Read elements header
    header = lines[elem_start + 1].split()
    num_entity_blocks = int(header[0])
    num_elements_total = int(header[1])
    # print(f"  Entity blocks: {num_entity_blocks}, Total elements: {num_elements_total}")
    
    # Storage for different element types
    triangles = []  # Elements of type 2 (triangles) in Domain (tag 1)
    outer_edges = []  # Elements of type 1 (lines) in OuterBoundary (tag 2)
    inner_edges = []  # Elements of type 1 (lines) in InnerBoundary (tag 3)
    
    idx = elem_start + 2
    for entity_block in range(num_entity_blocks):
        parts = lines[idx].split()
        entity_dim = int(parts[0])
        entity_tag = int(parts[1])
        element_type = int(parts[2])
        num_elements_in_entity = int(parts[3])
        
        print(f"  Entity dim={entity_dim}, tag={entity_tag}, type={element_type}, "
              f"count={num_elements_in_entity}")
        
        idx += 1
        
        for _ in range(num_elements_in_entity):
            parts = list(map(int, lines[idx].split()))
            element_id = parts[0]
            nodes = parts[1:]
            
            # Convert MSH indices to Python indices
            nodes_py = [index_mapping[n] for n in nodes]
            
            if element_type == 2 and entity_tag == 1:  # Triangle in Domain
                triangles.append(nodes_py)
            elif element_type == 1 and entity_tag in [1, 2, 3, 4, 5, 6, 7]:  # OuterBoundary (все 4 кривые!)
                outer_edges.append(nodes_py)
            elif element_type == 1 and entity_tag in [8, 9]:  # InnerBoundary (2 кривые)
                inner_edges.append(nodes_py)
            
            idx += 1
    
    F = np.array(triangles, dtype=np.int32)
    # print(f"  Loaded {len(F)} triangles")
    # print(f"  Loaded {len(outer_edges)} outer boundary edges")
    # print(f"  Loaded {len(inner_edges)} inner boundary edges")
    
    # ========================================================================
    # STEP 3: Reconstruct boundary curves from edges
    # ========================================================================
    # print("\n[STEP 3] Reconstructing boundary curves...")
    
    def reconstruct_curve(edges):
        """
        Восстанавливает ориентированную кривую из набора рёбер.
        Рёбра - это пары вершин [(v0, v1), (v2, v3), ...]
        Результат - упорядоченный список вершин.
        """
        if not edges:
            return []
        
        # Build adjacency
        edge_dict = {}
        for v1, v2 in edges:
            if v1 not in edge_dict:
                edge_dict[v1] = []
            edge_dict[v1].append(v2)
        
        # Start from first vertex and follow the chain
        curve = [edges[0][0]]  # Start vertex
        current = edges[0][1]
        
        while current != curve[0] and len(curve) < len(edges) + 10:  # Prevent infinite loop
            curve.append(current)
            if current in edge_dict and edge_dict[current]:
                current = edge_dict[current][0]
            else:
                break
        
        return curve
    
    B_outer_ordered = reconstruct_curve(outer_edges)
    B_inner_ordered = reconstruct_curve(inner_edges)
    
    print(f"  Outer boundary: {len(B_outer_ordered)} nodes")
    print(f"  Inner boundary: {len(B_inner_ordered)} nodes")
    
    # ========================================================================
    # STEP 4: Check and fix orientation
    # ========================================================================
    print("\n[STEP 4] Checking and fixing orientation...")
    
    def compute_signed_area(coords):
        """
        Вычисляет ориентированную площадь замкнутого контура.
        > 0: против часовой стрелки
        < 0: по часовой стрелке
        """
        n = len(coords)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += (coords[j][0] - coords[i][0]) * (coords[j][1] + coords[i][1])
        return area / 2.0
    
    # Check outer boundary
    outer_coords = X[B_outer_ordered]
    outer_signed_area = compute_signed_area(outer_coords)
    print(f"  Outer boundary signed area: {outer_signed_area:.6f}")
    if outer_signed_area < 0:
        print(f"  ✓ Outer boundary is counter-clockwise (correct)")
    else:
        print(f" ! Outer boundary is clockwise, reversing... ")
        B_outer_ordered = B_outer_ordered[::-1]
    
    # Check inner boundary
    inner_coords = X[B_inner_ordered]
    inner_signed_area = compute_signed_area(inner_coords)
    print(f"  Inner boundary signed area: {inner_signed_area:.6f}")
    if inner_signed_area < 0:
        print(f"  ! Inner boundary is counter-clockwise, reversing...")
        B_inner_ordered = B_inner_ordered[::-1]
    else:
        print(f"  ✓ Inner boundary is clockwise (correct)")
    
    B_outer = B_outer_ordered
    B_inner = B_inner_ordered
    
    # STEP 5: Compute hole information
    # print("\n[STEP 5] Computing hole information...")
    
    inner_coords = X[B_inner]
    center_hole = inner_coords.mean(axis=0)
    distances = np.linalg.norm(inner_coords - center_hole, axis=1)
    radius_hole = distances.mean()
    
    # print(f"  Hole center: ({center_hole[0]:.6f}, {center_hole[1]:.6f})")
    # print(f"  Hole radius: {radius_hole:.6f}")
    # print(f"  Hole radius range: [{distances.min():.6f}, {distances.max():.6f}]")
    # print(f"  Hole boundary bbox: x=[{inner_coords[:,0].min():.6f}, {inner_coords[:,0].max():.6f}], "
    #       f"y=[{inner_coords[:,1].min():.6f}, {inner_coords[:,1].max():.6f}]")
    
    # ========================================================================
    # STEP 6: Find "lower-left" point on inner boundary
    # ========================================================================
    print("\n[STEP 6] Finding lower-left point on inner boundary...")
    
    # Find point with minimum x, then minimum y
    min_idx = 0
    min_val = (inner_coords[0][0], inner_coords[0][1])
    for i in range(1, len(inner_coords)):
        if (inner_coords[i][0] < min_val[0] or 
            (inner_coords[i][0] == min_val[0] and inner_coords[i][1] < min_val[1])):
            min_idx = i
            min_val = (inner_coords[i][0], inner_coords[i][1])
    
    
    # Rotate B_inner so that lower-left point is first
    B_inner = B_inner[min_idx:] + B_inner[:min_idx]
    # print(f"  Lower-left point at index {min_idx}: {X[B_inner[0]]}")
    
    # Similarly for outer boundary
    outer_coords = X[B_outer]
    min_idx = 0
    min_val = (outer_coords[0][0], outer_coords[0][1])
    for i in range(1, len(outer_coords)):
        if (outer_coords[i][0] < min_val[0] and outer_coords[i][1] ==  0):
            min_idx = i
            min_val = (outer_coords[i][0], outer_coords[i][1])
    
    B_outer = B_outer[min_idx:] + B_outer[:min_idx]
    # print(f"  Outer lower-left point at index {min_idx}: {X[B_outer[0]]}")
    
    # ========================================================================
    # FINAL INFO
    # ========================================================================
    info = {
        'center_hole': center_hole,
        'radius_hole': radius_hole,
        'n_nodes': len(X),
        'n_triangles': len(F),
        'n_outer': len(B_outer),
        'n_inner': len(B_inner),
    }
    
    # print(f"\n{'='*70}")
    # print(f"Summary:")
    # print(f"  Nodes: {info['n_nodes']}")
    # print(f"  Triangles: {info['n_triangles']}")
    # print(f"  Outer boundary nodes: {info['n_outer']}")
    # print(f"  Inner boundary nodes: {info['n_inner']}")
    # print(f"{'='*70}\n")
    
    return X, F, B_outer, B_inner, info
