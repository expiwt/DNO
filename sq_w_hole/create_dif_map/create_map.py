import numpy as np
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, diags
from scipy.sparse.linalg import spsolve
import os
import glob
import csv
import shutil # Для очистки старых данных
import gc

import matplotlib
matplotlib.use('Agg')  # Включаем "безэкранный" режим. Это экономит кучу памяти.
import matplotlib.pyplot as plt

from read_msh import read_msh_file
from map_utils.interpolate import GridInterpolator



class DiffeomorphismMapper:
    """
    Класс для построения диффеоморфизма из тфизического пространства
    (четырёхугольник с круглой дыркой) в универсальное пространство
    (квадрат [0,1]x[0,1] с круглой дыркой в центре).
    
    Физическое пространство:
    - Четырёхугольник (произвольной формы)
    - Круглая дыра внутри
    
    Универсальное пространство:
    - Квадрат [0,1]x[0,1]
    - Круг радиусом r=0.1 с центром [0.5, 0.5]
    """
    
    def __init__(self, X, F, B_outer, B_inner):
        """
        X: (n_nodes, 2) массив координат узлов
        F: (n_triangles, 3) массив индексов треугольников
        B_outer: индексы узлов на внешней границе (против часовой)
        B_inner: индексы узлов на внутренней границе (по часовой)
        """
        self.X = X
        self.F = F
        self.B_outer = B_outer
        self.B_inner = B_inner
        self.n_nodes = X.shape[0]
        
        # Параметры универсального пространства
        self.target_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.hole_center_target = np.array([0.5, 0.5])
        self.hole_radius_target = 0.1
        self.Y = None
        
        # print("\n" + "="*70)
        # print("Инициализация диффеоморфизма")
        # print("="*70)
        # print(f"Узлов в сетке: {self.n_nodes}")
        # print(f"Треугольников: {F.shape[0]}")
        # print(f"Узлов на внешней границе: {len(B_outer)}")
        # print(f"Узлов на внутренней границе: {len(B_inner)}")
        # print(f"B_outer {B_outer}")
    def compute_hole_parameters(self):
        """
        Вычисляет центр и радиус дыры в физическом пространстве.
        """
        inner_coords = self.X[self.B_inner]
        center = np.mean(inner_coords, axis=0)
        radius = np.mean(np.linalg.norm(inner_coords - center, axis=1))
        
        self.hole_center_physical = center
        self.hole_radius_physical = radius
        
        # print(f"\nПараметры дыры в физическом пространстве:")
        # print(f"  Центр: {center}")
        # print(f"  Радиус: {radius:.6f}")
        
        return center, radius
    def find_corner_correspondence(self):
        """
        Находит 4 угла четырёхугольника в физическом пространстве.
        Сопоставляет их с углами квадрата в универсальном пространстве.
        """
        # 4 направления от центра дыры к углам: -x-y, +x-y, +x+y, -x+y
        corner_directions_ideal = np.array([
            [-1, -1],
            [1, -1],
            [1, 1],
            [-1, 1]
        ], dtype=float)
        
        # Нормализуем
        for i in range(4):
            corner_directions_ideal[i] /= np.linalg.norm(corner_directions_ideal[i])
        
        outer_coords = self.X[self.B_outer]
        
        corner_indices_physical = []
        corner_coords_physical = []
        
        # Для каждого направления найдём узел на внешней границе
        for direction in corner_directions_ideal:
            projections = np.dot(outer_coords - self.hole_center_physical, direction)
            max_proj_idx = np.argmax(projections)
            
            corner_node_idx = self.B_outer[max_proj_idx]
            corner_coords_physical.append(self.X[corner_node_idx])
            corner_indices_physical.append(corner_node_idx)
        
        self.corner_indices_physical = corner_indices_physical
        self.corner_coords_physical = np.array(corner_coords_physical)
        
        # Целевые углы квадрата [0,1]x[0,1]
        self.target_corners = np.array([
            [0.0, 0.0],  # НЛ
            [1.0, 0.0],  # НП
            [1.0, 1.0],  # ВП
            [0.0, 1.0]   # ВЛ
        ])
        
        # print(f"\nНайденные углы в физическом пространстве:")
        for i, (idx, coord) in enumerate(zip(corner_indices_physical, corner_coords_physical)):
            print(f"  Угол {i} (узел {idx}): {coord} -> {self.target_corners[i]}")
        # print(corner_indices_physical, corner_coords_physical)
        
        return corner_indices_physical, corner_coords_physical


    
    def find_hole_boundary_correspondence(self):
        """
        Находит точки на дыре в физическом пространстве в направлениях углов.
        Сопоставляет их с точками на окружности радиусом 0.1 в универсальном пространстве.
        
        Логика:
        1. Из диагональных точек [0.4,0.4], [0.6,0.4], [0.6,0.6], [0.4,0.6] вычисляем направления
        2. Нормализуем направления
        3. Масштабируем на радиус 0.1: целевая_точка = [0.5,0.5] + 0.1 * нормированное_направление
        4. Это гарантирует, что целевые точки лежат на окружности радиусом 0.1
        """
        
        # Вычисляем направления от центра дыры к каждому углу в физическом пространстве
        corner_directions = []
        # print(f"{self.corner_coords_physical} - corners_coord")

        invers_corner_coords_physical = self.corner_coords_physical[::-1]
        invers_corner_coords_physical = np.roll(invers_corner_coords_physical, 1, axis=0)
        # print(f"{invers_corner_coords_physical} - invers_corners_coord")

        for corner_coord in invers_corner_coords_physical:
            direction = corner_coord - self.hole_center_physical
            direction = direction / np.linalg.norm(direction)
            corner_directions.append(direction)
        

        inner_coords = self.X[self.B_inner]
        
        hole_indices_physical = []
        hole_coords_physical = []
        
        # Диагональные точки - опорные для вычисления направлений в универсальном пространстве
        diagonal_points = np.array([
            [0.4, 0.4],  # НЛ
            [0.4, 0.6],   # ВЛ
            [0.6, 0.6],  # ВП
            [0.6, 0.4]  # НП
        ])
        
        # Целевые точки на окружности радиусом 0.1
        target_center = np.array([0.5, 0.5])
        target_radius = 0.1
        hole_coords_target = []
        
        for diag_point in diagonal_points:
            # Вычисляем направление от центра к диагональной точке
            t_direction = diag_point - target_center
            # Нормализуем
            t_direction_norm = t_direction / np.linalg.norm(t_direction)
            # Масштабируем на радиус
            t_point = target_center + target_radius * t_direction_norm
            hole_coords_target.append(t_point)
        
        # Находим соответствующие точки на дыре в физическом пространстве
        for i, direction in enumerate(corner_directions):
            # Проецируем все точки дыры на направление к углу
            projections = np.dot(inner_coords - self.hole_center_physical, direction)
            max_proj_idx = np.argmax(projections)
            
            hole_node_idx = self.B_inner[max_proj_idx]
            hole_coords_physical.append(self.X[hole_node_idx])
            hole_indices_physical.append(hole_node_idx)
        
        self.hole_indices_physical = hole_indices_physical
        self.hole_coords_physical = np.array(hole_coords_physical)
        self.hole_coords_target = np.array(hole_coords_target)
        
        # print(f"\nЦелевые точки на окружности радиусом 0.1:")
        for i, point in enumerate(self.hole_coords_target):
            dist_from_center = np.linalg.norm(point - target_center)
            # print(f"  Угол {i}: {point}, расстояние: {dist_from_center:.6f}")
        
        # print(f"\nСоответствие точек на дыре:")
        for i, (phys_idx, phys_coord) in enumerate(zip(hole_indices_physical, hole_coords_physical)):
            target_coord = hole_coords_target[i]
            # print(f"  Физическое (узел {phys_idx}): {phys_coord}")
            # print(f"  Целевое: {target_coord}")
            # print()
        return hole_indices_physical, hole_coords_physical
    
    def construct_boundary_conditions(self):
        """
        Устанавливает целевые координаты на границах для уравнения Лапласа.
        """
        Z_boundary = np.zeros((self.n_nodes, 2))
        
        # Параметризуем внешнюю границу
        outer_params = self._parametrize_outer_boundary()
        if outer_params is None:
            return None
        
        for i, node_idx in enumerate(self.B_outer):
            Z_boundary[int(node_idx)] = outer_params[i]
        
        # Параметризуем внутреннюю границу
        inner_params = self._parametrize_inner_boundary()
        if inner_params is None:
            return None
        
        for i, node_idx in enumerate(self.B_inner):
            Z_boundary[int(node_idx)] = inner_params[i]
        
        self.Z_boundary = Z_boundary
        self.boundary_nodes = list(set(self.B_outer) | set(self.B_inner))
        
        # Проверка порядка и соответствия
        # print(f"\nПроверка граничных условий:")
        # print(f"  Внешняя граница ({len(self.B_outer)} узлов):")
        for i, node_idx in enumerate(self.corner_indices_physical):
            bc = Z_boundary[int(node_idx)]
            # print(f"    Угол {i} (узел {node_idx}): {bc} ✓" if np.linalg.norm(bc - self.target_corners[i]) < 0.01 else f"    Угол {i} (узел {node_idx}): {bc} ✗")
        
        # print(f"  Внутренняя граница ({len(self.B_inner)} узлов):")
        for i, node_idx in enumerate(self.hole_indices_physical):
            bc = Z_boundary[int(node_idx)]
            dist = np.linalg.norm(bc - np.array([0.5, 0.5]))
            # print(f"    Точка {i} (узел {node_idx}): {bc}, расстояние={dist:.6f}" + (" ✓" if abs(dist - 0.1) < 0.01 else " ✗"))
        

        # print("\n=== ПРОВЕРКА ГРАНИЧНЫХ УСЛОВИЙ (RAW DUMP) ===")
        # print(f"{'Node':<6} | {'Phys (X, Y)':<20} | {'Target (U, V)':<20}")
        # print("-" * 50)

        # Выводим внешнюю границу
        for node_idx in self.B_outer:
            phys = self.X[int(node_idx)]
            target = Z_boundary[int(node_idx)]
            # print(f"{node_idx:<6} | [{phys[0]:.4f}, {phys[1]:.4f}]    | [{target[0]:.4f}, {target[1]:.4f}]")

        # Выводим внутреннюю границу
        # print("\n--- ВНУТРЕННЯЯ ГРАНИЦА ---")
        for node_idx in self.B_inner:
            phys = self.X[int(node_idx)]
            target = Z_boundary[int(node_idx)]
            print(f"{node_idx:<6} | [{phys[0]:.4f}, {phys[1]:.4f}]    | [{target[0]:.4f}, {target[1]:.4f}]")
        # print("=" * 50 + "\n")


        return Z_boundary
        
    def _parametrize_outer_boundary(self):
        """
        Параметризует внешнюю границу на стороны квадрата [0,1]x[0,1].
        """
        n_outer = len(self.B_outer)
        params = np.zeros((n_outer, 2))
        first_corner_id = self.corner_indices_physical[0]
        
        # 2. Находим, где сейчас находится первый угол
        shift_idx = self.B_outer.index(first_corner_id)
        
        # print(f"\n[Outer Parametrization] Сдвигаем B_outer. Старт был на {shift_idx}, теперь будет 0.")
        # print(f"{self.B_outer} - self.B_outer\n {self.X[self.B_outer]} - self.X[self.B_outer]")
        # 3. Сдвигаем массив B_outer
        # Было: [..., FirstCorner, ...] -> Стало: [FirstCorner, ..., ...]
        self.B_outer = self.B_outer[shift_idx:] + self.B_outer[:shift_idx]
        # print(f"{self.B_outer} - self.B_outer_shifted")
        # print(f"{self.corner_indices_physical}-self.corner_indices_physical")
        corner_positions = []
        for indx in self.corner_indices_physical:
            for i in range(n_outer):
                if indx == self.B_outer[i]:
                    corner_positions.append(i)
                    break

        # print(f"[Parametrization] Позиции углов в массиве B_outer: {corner_positions}")

        # Четыре стороны квадрата
        sides = [
            (corner_positions[0], corner_positions[1], np.array([0, 0]), np.array([1, 0])),
            
            (corner_positions[1], corner_positions[2], np.array([1, 0]), np.array([1, 1])),
            
            (corner_positions[2], corner_positions[3], np.array([1, 1]), np.array([0, 1])),
            
            (corner_positions[3], corner_positions[0], np.array([0, 1]), np.array([0, 0]))
        ]
        
        # Обработаем каждую сторону
        for side_num, (start_idx, end_idx, start_target, end_target) in enumerate(sides):
            
            # print(f"\nСТОРОНА {side_num}: от позиции {start_idx} к позиции {end_idx}")
            # print(f"  от координаты {start_target} к координаде {end_target}")
            
            # ШАГ 1: Найти все позиции между start_idx и end_idx
            if start_idx < end_idx:
                # Нормальный случай: просто от start к end
                indices = np.arange(start_idx, end_idx + 1)
                # print(f"  Нормальный диапазон: {indices[:3]}...{indices[-3:]}")
            
            elif start_idx > end_idx:
                # Циклический случай: от start к концу, потом от начала к end
                indices = np.concatenate([
                    np.arange(start_idx, n_outer),     # [start, ..., 98]
                    np.arange(0, end_idx + 1)          # [0, ..., end]
                ])
                # print(f"  Циклический переход: {indices[:3]}...{indices[-3:]}")
            
            else:
                # start_idx == end_idx (одна точка - угол)
                indices = np.array([start_idx])
                # print(f"  Одна точка (угол): {indices}")
            
            # ШАГ 2: Создаём "линейку" от 0 до 1
            n_on_side = len(indices)
            t = np.linspace(0, 1, n_on_side)
            # print(f"  Узлов на стороне: {n_on_side}")
            # print(f"  t = {t[:3]}...{t[-3:]}")
            
            # ШАГ 3: Для каждого узла вычисляем его целевую позицию
            for j, idx in enumerate(indices):
                # Линейная интерполяция
                params[idx] = (1 - t[j]) * start_target + t[j] * end_target
                
                # if j == 0 or j == n_on_side - 1 or j == n_on_side // 2:
                    # print(f"    j={j}, idx={idx}: t={t[j]:.4f} → params[{idx}] = {params[idx]}")
        
        return params
    
    def _parametrize_inner_boundary(self):
        """
        Параметризация внутренней границы:
        1. Сдвигаем массив B_inner, чтобы начать с 1-го угла.
        2. Идем по порядку, уменьшая угол (против часовой стрелки в вашей логике).
        """
        # --- ШАГ 1: Сдвиг массива B_inner ---
        # Находим, где сейчас лежит первый угол
        start_node_id = self.hole_indices_physical[0]
        
        if start_node_id not in self.B_inner:
            print(f"Error: Start node {start_node_id} not found in B_inner")
            return None
            
        start_pos = self.B_inner.index(start_node_id)
        
        # Делаем циклический сдвиг (rotate), чтобы start_node_id стал нулевым элементом
        # Было: [..., start, ..., end] -> Стало: [start, ..., end, ...]
        self.B_inner = self.B_inner[start_pos:] + self.B_inner[:start_pos]
        
        # print(f"\n[Inner Parametrization] Массив B_inner сдвинут. Начало: {self.B_inner[0]} (должен быть {start_node_id})")
        # print(self.X[self.B_inner])
        # print(self.X[self.hole_indices_physical])

        # --- ШАГ 2: Поиск позиций остальных углов в новом массиве ---
        n_inner = len(self.B_inner)
        hole_positions = []
        
        # Теперь self.hole_indices_physical[0] точно находится на позиции 0
        for node_id in self.hole_indices_physical:
            pos = self.B_inner.index(node_id)
            hole_positions.append(pos)
            
        # print(f"Позиции углов в сдвинутом массиве: {hole_positions}")
        # Ожидается что-то вроде [0, 12, 25, 38] (цифры по возрастанию)

        # --- ШАГ 3: Расчет длин интервалов (delts) ---
        # Так как массив упорядочен и начинается с 0, логика упрощается:
        # delt0 - количество шагов от угла 0 до угла 1
        delt0 = hole_positions[1] - hole_positions[0]
        delt1 = hole_positions[2] - hole_positions[1]
        delt2 = hole_positions[3] - hole_positions[2]
        delt3 = n_inner - hole_positions[3] # Оставшийся кусок до конца круга (замыкание на 0)
        
        delts = [delt0, delt1, delt2, delt3]
        # print(f"Количество шагов в секторах: {delts}")

        # --- ШАГ 4: Инициализация параметров ---
        params = np.zeros((n_inner, 2))
        center = self.hole_center_target # [0.5, 0.5]
        radius = self.hole_radius_target # 0.1
        pi = np.pi

        # Начальные углы (как в вашем примере)
        # Угол 0: 225 градусов
        init_ang_1 = pi + pi/4 
        init_ang_2 = init_ang_1 - pi/2
        init_ang_3 = init_ang_2 - pi/2
        init_ang_4 = init_ang_3 - pi/2
        
        # Шаг угла для каждого сектора
        # Делим pi/2 на количество отрезков (delt)
        # Защита от деления на ноль (max(d, 1))
        step_angle = [
            (pi/2) / delts[0] if delts[0] > 0 else 0,
            (pi/2) / delts[1] if delts[1] > 0 else 0,
            (pi/2) / delts[2] if delts[2] > 0 else 0,
            (pi/2) / delts[3] if delts[3] > 0 else 0
        ]

        # --- ШАГ 5: Заполнение координат (Ваша логика с if/elif) ---
        for i in range(n_inner):
            # Сектор 1 (от 0 до hole_positions[1])
            if i <= hole_positions[1]:
                # Локальный индекс внутри сектора = i
                params[i, 0] = center[0] + radius * np.cos(init_ang_1 - i * step_angle[0])
                params[i, 1] = center[1] + radius * np.sin(init_ang_1 - i * step_angle[0])
            
            # Сектор 2 (от hole_positions[1] до hole_positions[2])
            elif i <= hole_positions[2]:
                # Нам нужно, чтобы при i = hole_positions[1] угол совпадал.
                # Поэтому считаем шаги (k) относительно начала ЭТОГО сектора
                k = i - hole_positions[1]
                params[i, 0] = center[0] + radius * np.cos(init_ang_2 - k * step_angle[1])
                params[i, 1] = center[1] + radius * np.sin(init_ang_2 - k * step_angle[1])
            
            # Сектор 3 (от hole_positions[2] до hole_positions[3])
            elif i <= hole_positions[3]:
                k = i - hole_positions[2]
                params[i, 0] = center[0] + radius * np.cos(init_ang_3 - k * step_angle[2])
                params[i, 1] = center[1] + radius * np.sin(init_ang_3 - k * step_angle[2])
            
            # Сектор 4 (от hole_positions[3] до конца)
            else:
                k = i - hole_positions[3]
                params[i, 0] = center[0] + radius * np.cos(init_ang_4 - k * step_angle[3])
                params[i, 1] = center[1] + radius * np.sin(init_ang_4 - k * step_angle[3])

        # print(f"{params} - params")
        # print(f"{self.X[self.B_inner]} - self.X[self.B_inner]")

        return params

    
    def build_cotangent_laplacian(self):
        """
        Строит матрицу Лапласиана с использованием котангенсных весов.
        Это стандартный подход для конформных отображений на триангулированных поверхностях.
        """
        # print(f"\nПостроение матрицы Лапласиана...")
        
        W = coo_matrix((self.n_nodes, self.n_nodes))
        
        # Для каждого угла в каждом треугольнике вычисляем котангенс
        for i in range(3):
            i2 = (i + 1) % 3
            i3 = (i + 2) % 3
            
            F_int = self.F.astype('int64')
            
            # Векторы сторон треугольника
            u = self.X[F_int[:, i2]] - self.X[F_int[:, i]]
            v = self.X[F_int[:, i3]] - self.X[F_int[:, i]]
            
            # Нормализация векторов
            u_norm = np.sqrt(np.sum(u**2, axis=1))
            v_norm = np.sqrt(np.sum(v**2, axis=1))
            
            u = u / u_norm[:, np.newaxis]
            v = v / v_norm[:, np.newaxis]
            
            # Косинус угла
            cos_angle = np.sum(u * v, axis=1)
            cos_angle = np.clip(cos_angle, -0.9999, 0.9999)
            
            # Котангенс угла
            angle = np.arccos(cos_angle)
            alpha = 1.0 / np.tan(angle)
            alpha = np.maximum(alpha, 1e-2 * np.ones(len(alpha)))
            
            # Добавляем в матрицу весов
            W = W + coo_matrix((alpha, (F_int[:, i2], F_int[:, i3])), shape=(self.n_nodes, self.n_nodes))
            W = W + coo_matrix((alpha, (F_int[:, i3], F_int[:, i2])), shape=(self.n_nodes, self.n_nodes))
        
        # Матрица Лапласиана L = D - W
        d = np.array(W.sum(axis=0)).flatten()
        D = diags(d, 0)
        L = D - W
        
        self.L = L
        # print(f"Матрица Лапласиана построена: {L.shape}")
        
        return L
    
    def apply_boundary_conditions(self):
        """
        Применяет граничные условия Дирихле к матрице Лапласиана.
        """
        L_bc = self.L.toarray().copy()
        
        for node_idx in self.boundary_nodes:
            L_bc[int(node_idx), :] = 0
            L_bc[int(node_idx), int(node_idx)] = 1
        
        # Защита от сингулярности
        for i in range(L_bc.shape[0]):
            if L_bc[i, i] == 0:
                L_bc[i, i] = 1e-6
        
        self.L_bc = L_bc
        return L_bc

    
    def solve_laplace_equation(self):
        """
        Решает L_bc * Y = R
        где R[граница] = Z_boundary, R[внутри] = 0
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import spsolve
        
        R = np.zeros((self.n_nodes, 2))
        # print(f"{self.Z_boundary}- Z_bound")
        for node_idx in self.boundary_nodes:
            R[int(node_idx)] = self.Z_boundary[int(node_idx)]
        L_bc_sparse = csr_matrix(self.L_bc)
        
        Y_x = spsolve(L_bc_sparse, R[:, 0])
        Y_y = spsolve(L_bc_sparse, R[:, 1])
        
        self.Y = np.column_stack([Y_x, Y_y])
        
        return self.Y
    
    
    def build_diffeomorphism(self):
        """
        Основной метод для построения всего диффеоморфизма.
        Объединяет все этапы.
        """
        # Этап 1: вычисляем параметры дыры
        self.compute_hole_parameters()
        
        # Этап 2: находим соответствие углов
        self.find_corner_correspondence()
        
        # Этап 3: находим соответствие точек на дыре
        self.find_hole_boundary_correspondence()
        
        # Этап 4: устанавливаем граничные условия
        self.construct_boundary_conditions()
        
        # Этап 5: строим матрицу Лапласиана
        self.build_cotangent_laplacian()
        
        # Этап 6: применяем граничные условия к матрице
        self.apply_boundary_conditions()
        
        # Этап 7: решаем уравнение Лапласа
        Y = self.solve_laplace_equation()
        
        if Y is None:
            print("Ошибка построения диффеоморфизма!")
            return False
        
        return True
    
    def visualize(self, output_prefix="result"):
        """
        Визуализирует результаты диффеоморфизма.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Физическое пространство
        ax = axes[0]
        ax.scatter(self.X[:, 0], self.X[:, 1], s=1, alpha=0.3)
        
        # Рисуем границы
        outer_coords = self.X[self.B_outer]
        ax.plot(outer_coords[:, 0], outer_coords[:, 1], 'b-', linewidth=2, label='Внешняя граница')
        
        inner_coords = self.X[self.B_inner]
        ax.plot(inner_coords[:, 0], inner_coords[:, 1], 'r-', linewidth=2, label='Внутренняя граница')
        
        # Рисуем углы
        ax.scatter(self.corner_coords_physical[:, 0], self.corner_coords_physical[:, 1], 
                  c='green', s=100, marker='s', label='Углы', zorder=5)
        
        # Рисуем точки на дыре
        ax.scatter(self.hole_coords_physical[:, 0], self.hole_coords_physical[:, 1],
                  c='orange', s=100, marker='D', label='Точки дыры', zorder=5)
        
        ax.set_title('Физическое пространство')
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Универсальное пространство
        ax = axes[1]
        ax.scatter(self.Y[:, 0], self.Y[:, 1], s=1, alpha=0.3)
        
        # Рисуем целевой квадрат
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        ax.plot(square[:, 0], square[:, 1], 'b-', linewidth=2, label='Квадрат')
        
        # Рисуем целевую дыру
        angles = np.linspace(0, 2*np.pi, 100)
        hole_x = 0.5 + 0.1 * np.cos(angles)
        hole_y = 0.5 + 0.1 * np.sin(angles)
        ax.plot(hole_x, hole_y, 'r-', linewidth=2, label='Дыра')
        
        # Рисуем углы в целевом пространстве
        ax.scatter(self.target_corners[:, 0], self.target_corners[:, 1],
                  c='green', s=100, marker='s', label='Целевые углы', zorder=5)
        
        # Рисуем целевые точки на дыре
        ax.scatter(self.hole_coords_target[:, 0], self.hole_coords_target[:, 1],
                  c='orange', s=100, marker='D', label='Целевые точки дыры', zorder=5)
        
        ax.set_title('Универсальное пространство')
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_visualization.png", dpi=150, bbox_inches='tight')
        # print(f"\nВизуализация сохранена в {output_prefix}_visualization.png")
        plt.close()

def clear_output_files(output_x, output_y):
    """Удаляет старые файлы результатов перед запуском."""
    if os.path.exists(output_x):
        os.remove(output_x)
    if os.path.exists(output_y):
        os.remove(output_y)
    print("Старые данные удалены.")

def append_to_master_csv(filename, data_row):
    """Добавляет одну строку (flatten array) в CSV файл."""
    # Создаем папку если её нет (для надежности)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data_row.flatten().tolist())

def main():
    """
    Пример использования: загружаем mesh из MSH файла и строим диффеоморфизм.
    """
    MSH_FOLDER = "../train_sq_part_obj"  # Папка с .msh файлами
    FINAL_OUTPUT_DIR = "./final_train_dataset" # Куда класть итоговые CSV
    OUTPUT_CSV_X = os.path.join(FINAL_OUTPUT_DIR, "x_data.csv")
    OUTPUT_CSV_Y = os.path.join(FINAL_OUTPUT_DIR, "y_data.csv")
    if not os.path.exists(FINAL_OUTPUT_DIR):
        os.makedirs(FINAL_OUTPUT_DIR)

    # Очистка старых данных
    clear_output_files(OUTPUT_CSV_X, OUTPUT_CSV_Y)

    # Поиск всех msh файлов
    msh_files = sorted(glob.glob(os.path.join(MSH_FOLDER, "*.msh")))
    print(f"Найдено {len(msh_files)} файлов .msh для обработки.")

    if not msh_files:
        print("Внимание! Файлы не найдены. Проверьте путь.")
        return

    # Загружаем mesh
    #import io
    #from contextlib import redirect_stdout

    #f = io.StringIO()
    #with redirect_stdout(f):

    # Цикл по файлам
    for idx, msh_file in enumerate(msh_files):
        base_name = os.path.splitext(os.path.basename(msh_file))[0]
        print(f"\n[{idx+1}/{len(msh_files)}] Обработка: {base_name}")

        try:
            X, F, B_outer, B_inner, info = read_msh_file(msh_file)
        except Exception as e:
            print(f"Ошибка при чтении MSH файла: {e}")
            return
        
        # Создаём объект диффеоморфизма
        mapper = DiffeomorphismMapper(X, F, B_outer, B_inner)
        
        # Строим диффеоморфизм
        success = mapper.build_diffeomorphism()
        
        if success:
            
            # 2. Вызываем визуализацию маппинга
            # Это создаст файл .../debug_images/quad_X/mapping_visualization.png
            #print(f"{mapper.Y}- mapper.Y[1] ")
            # Сохраняем координаты в универсальном пространстве
            # np.save("universal_coordinates.npy", mapper.Y)
            print("\nКоординаты в универсальном пространстве сохранены в universal_coordinates.npy")
            try:
                # Создаем временную папку для картинок этого объекта, чтобы не мусорить
                obj_debug_dir = os.path.join(FINAL_OUTPUT_DIR, "debug_images", base_name)
                os.makedirs(obj_debug_dir, exist_ok=True)
                mapper.visualize(output_prefix=os.path.join(obj_debug_dir, "mapping"))

                interpolator = GridInterpolator(mapper)
                M_x, M_y = interpolator.interpolate(sampling_size=128, output_dir=obj_debug_dir)
            
                # Запись в общие файлы
                append_to_master_csv(OUTPUT_CSV_X, M_x)
                append_to_master_csv(OUTPUT_CSV_Y, M_y)
                print(f"  ✓ Успешно. Данные добавлены в {OUTPUT_CSV_X}")
                
                #очистка
                del mapper
                del interpolator
                del X, F, B_outer, B_inner
                del M_x, M_y
                
                # Закрываем ВСЕ графики matplotlib (на всякий случай)
                plt.close('all')
                
                # Принудительно вызываем сборщик мусора
                gc.collect()
            except Exception as e:
                print(f"   CRITICAL ERROR: {e}")
                import traceback
                traceback.print_exc()
                # Тоже чистим память при ошибке
                plt.close('all')
                gc.collect()
                continue
        else:
            print(f"   Ошибка построения диффеоморфизма для {base_name}")
            continue       
    print("Обработка всех файлов завершена.")
    print(f"Итоговые файлы:\n  {OUTPUT_CSV_X}\n  {OUTPUT_CSV_Y}")

if __name__ == "__main__":
    main()
