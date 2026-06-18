import numpy as np
import csv
import os
import matplotlib.pyplot as plt

class GridInterpolator:
    """
    Интерполятор: переносит физические координаты (X, Y) на регулярную сетку
    в логическом пространстве (Xi, Eta) методом растеризации треугольников.
    """
    
    def __init__(self, mapper):
        # mapper должен содержать:
        # .X -> Физические координаты (N, 2)
        # .F -> Индексы треугольников (M, 3)
        # .Y -> Логические координаты (Xi, Eta) (N, 2)
        self.phys_coords = mapper.X
        self.faces = mapper.F
        self.uv_coords = mapper.Y 
        
        if self.uv_coords is None:
            raise ValueError("Mapper object has no UV coordinates (Y). Run build_mapping() first.")

    def _compute_dual_interp(self, face, vertex, v_x, v_y, n):
        """
        Растеризует треугольники на сетку n x n.
        vertex: координаты вершин в логическом пространстве (0..1)
        v_x, v_y: значения физических координат в вершинах
        """
        # Транспонируем, если нужно (ожидаем 2xN для вершин, 3xM для граней)
        if face.shape[0] != 3 and face.shape[1] == 3:
            face = face.T
        if vertex.shape[0] != 2 and vertex.shape[1] == 2:
            vertex = vertex.T
        
        # Обрезаем координаты 0..1 (защита от float погрешностей)
        vertex = np.clip(vertex, 0.0, 1.0)
        
        nface = face.shape[1]
        
        # Буферы для накопления значений
        Mx = np.zeros(n * n)
        My = np.zeros(n * n)
        Mnb = np.zeros(n * n) # Счетчик попаданий (для усреднения на границах)
        
        scale_factor = n - 1
        
        # Проходим по всем треугольникам
        for i in range(nface):
            T = face[:, i]       # Индексы вершин треугольника
            P = vertex[:, T]     # Логические координаты вершин (Xi, Eta)
            
            Vx = v_x[T]          # Физические X в вершинах
            Vy = v_y[T]          # Физические Y в вершинах
            
            # 1. Bounding Box треугольника (в пикселях)
            min_u, max_u = P[0, :].min(), P[0, :].max()
            min_v, max_v = P[1, :].min(), P[1, :].max()
            
            idx_min_u = int(np.floor(min_u * scale_factor))
            idx_max_u = int(np.ceil(max_u * scale_factor))
            idx_min_v = int(np.floor(min_v * scale_factor))
            idx_max_v = int(np.ceil(max_v * scale_factor))
            
            idx_min_u = max(0, idx_min_u); idx_max_u = min(n - 1, idx_max_u)
            idx_min_v = max(0, idx_min_v); idx_max_v = min(n - 1, idx_max_v)
            
            if idx_min_u > idx_max_u or idx_min_v > idx_max_v:
                continue

            # 2. Генерируем пиксели внутри BB
            ix = np.arange(idx_min_u, idx_max_u + 1)
            iy = np.arange(idx_min_v, idx_max_v + 1)
            Iy_grid, Ix_grid = np.meshgrid(iy, ix)
            
            # Координаты пикселей в пространстве 0..1
            pos_u = Ix_grid.flatten() / scale_factor
            pos_v = Iy_grid.flatten() / scale_factor
            pos = np.vstack((pos_u, pos_v))
            
            p_count = pos.shape[1]
            if p_count == 0: continue
            
            # 3. Барицентрические координаты
            # P * c = pos => c = P_inv * pos
            # Добавляем строку 1 для нормализации (сумма весов = 1)
            a = np.vstack(([1, 1, 1], P))
            try:
                inva = np.linalg.pinv(a)
            except np.linalg.LinAlgError:
                continue
            
            b = np.vstack((np.ones([1, p_count]), pos))
            c = np.dot(inva, b)
            
            # 4. Проверка: точка внутри треугольника? (все веса >= 0)
            eps = -1e-9
            I_in = np.where((c[0, :] >= eps) & (c[1, :] >= eps) & (c[2, :] >= eps))[0]
            
            if len(I_in) == 0: continue
            
            c_final = c[:, I_in]
            final_ix = Ix_grid.flatten()[I_in]
            final_iy = Iy_grid.flatten()[I_in]
            
            # Плоский индекс пикселя
            J = np.ravel_multi_index([final_iy, final_ix], (n, n))
            
            # 5. Интерполяция значений
            vals_x = Vx[0]*c_final[0] + Vx[1]*c_final[1] + Vx[2]*c_final[2]
            vals_y = Vy[0]*c_final[0] + Vy[1]*c_final[1] + Vy[2]*c_final[2]
            
            np.add.at(Mx, J, vals_x)
            np.add.at(My, J, vals_y)
            np.add.at(Mnb, J, 1)
        
        # Нормализация (если в пиксель попало несколько треугольников на стыке)
        Mx = Mx.reshape(n, n)
        My = My.reshape(n, n)
        Mnb = Mnb.reshape(n, n)
        
        mask = Mnb > 0
        Mx[mask] /= Mnb[mask]
        My[mask] /= Mnb[mask]
        
        # Заполняем пустоты (если есть) NaN (или можно ближайшим)
        Mx[~mask] = np.nan
        My[~mask] = np.nan
        
        return Mx, My

    def interpolate(self, sampling_size=128):
        """Возвращает матрицы Mx, My размером (size, size)"""
        # Транспонируем для _compute_dual_interp
        faces = self.faces.T
        uv = self.uv_coords.T
        phys_x = self.phys_coords[:, 0]
        phys_y = self.phys_coords[:, 1]
        
        M_x, M_y = self._compute_dual_interp(faces, uv, phys_x, phys_y, sampling_size)
        return M_x, M_y