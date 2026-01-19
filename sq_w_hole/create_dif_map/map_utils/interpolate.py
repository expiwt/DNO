import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import csv
import os

class GridInterpolator:
    """
    Класс для интерполяции физических координат с нерегулярной треугольной сетки
    на регулярную сетку в универсальном пространстве (квадрат).
    Работает в естественных координатах [0, 1].
    """
    
    def __init__(self, mapper):
        self.phys_coords = mapper.X  # (N, 2)
        self.faces = mapper.F        # (M, 3)
        self.uv_coords = mapper.Y    # (N, 2)
        
        if self.uv_coords is None:
            raise ValueError("Mapper object has no UV coordinates (Y). Run build_diffeomorphism() first.")

    def _compute_dual_interp(self, face, vertex, v_x, v_y, n):
        """
        Интерполирует сразу два канала (X и Y).
        Работает в нормализованных координатах 0..1.
        """
        # Приводим размерности (2 x N_vert, 3 x N_tri)
        if face.shape[1] == 3 and face.shape[0] != 3:
            face = face.T
        if vertex.shape[1] == 2 and vertex.shape[0] != 2:
            vertex = vertex.T
        
        # ВАЖНО: Мы НЕ масштабируем vertex в 1..n. Оставляем 0..1.
        # Но нам нужно защититься от вылета за 0.0 и 1.0 из-за флоат-погрешности
        vertex = np.clip(vertex, 0.0, 1.0)
        
        nface = face.shape[1]
        
        # Аккумуляторы (плоские)
        Mx = np.zeros(n * n)
        My = np.zeros(n * n)
        Mnb = np.zeros(n * n)
        
        # Шаг сетки (dx)
        # Если у нас n точек от 0 до 1, то шаг = 1 / (n-1)
        # Координата x_i = i * dx
        # Обратно: индекс i = round(x / dx) = round(x * (n-1))
        scale_factor = n - 1
        
        print(f"    Rasterizing {nface} triangles (Normalized 0..1)...")
        
        for i in range(nface):
            T = face[:, i]
            P = vertex[:, T] # Координаты вершин (0..1)
            
            Vx = v_x[T]
            Vy = v_y[T]
            
            # --- Хак для вырожденных треугольников (адаптирован под малые масштабы) ---
            # Раньше сдвигали на 0.01 пикселя. Теперь это 0.01 / 128 ≈ 0.0001
            # Но если вершины совпадают, pinv все равно упадет.
            # Простейшая защита: если площадь ~0, пропускаем или чуть двигаем.
            # Оставим пока как есть, pinv обычно справляется или кидает ошибку.
            
            # --- Bounding Box (в индексах) ---
            # Находим мин/макс координаты (0..1)
            min_u, max_u = P[0, :].min(), P[0, :].max()
            min_v, max_v = P[1, :].min(), P[1, :].max()
            
            # Переводим в индексы сетки (0..n-1)
            idx_min_u = int(np.floor(min_u * scale_factor))
            idx_max_u = int(np.ceil(max_u * scale_factor))
            idx_min_v = int(np.floor(min_v * scale_factor))
            idx_max_v = int(np.ceil(max_v * scale_factor))
            
            # Защита границ
            idx_min_u = max(0, idx_min_u); idx_max_u = min(n - 1, idx_max_u)
            idx_min_v = max(0, idx_min_v); idx_max_v = min(n - 1, idx_max_v)
            
            if idx_min_u > idx_max_u or idx_min_v > idx_max_v:
                continue

            # Создаем сетку индексов внутри bbox
            # range не включает правую границу, поэтому +1
            ix = np.arange(idx_min_u, idx_max_u + 1)
            iy = np.arange(idx_min_v, idx_max_v + 1)
            
            # Meshgrid ИНДЕКСОВ
            Iy_grid, Ix_grid = np.meshgrid(iy, ix)
            
            # Переводим индексы обратно в координаты (0..1) для проверки барицентрики
            # pos_u = ix / scale_factor
            pos_u = Ix_grid.flatten() / scale_factor
            pos_v = Iy_grid.flatten() / scale_factor
            
            pos = np.vstack((pos_u, pos_v)) # (2, N_pixels)
            p_count = pos.shape[1]
            
            if p_count == 0: continue
            
            # --- Барицентрические координаты ---
            # Решаем P * c = pos
            # Добавляем строку 1 для суммы весов
            a = np.vstack(([1, 1, 1], P))
            
            try:
                inva = np.linalg.pinv(a)
            except np.linalg.LinAlgError:
                continue
            
            b = np.vstack((np.ones([1, p_count]), pos))
            c = np.dot(inva, b)
            
            # --- Проверка: точка внутри? ---
            eps = -1e-9
            # Условие: все веса >= 0
            I_in = np.where((c[0, :] >= eps) & (c[1, :] >= eps) & (c[2, :] >= eps))[0]
            
            if len(I_in) == 0: continue
            
            # Отбираем валидные точки
            c_final = c[:, I_in]
            
            # Индексы в плоском массиве
            # Нам нужны индексы ix, iy, которые соответствуют отобранным точкам.
            # Мы расплющили Ix_grid и Iy_grid, поэтому берем те же индексы I_in
            final_ix = Ix_grid.flatten()[I_in]
            final_iy = Iy_grid.flatten()[I_in]
            
            # Плоский индекс J = ix * n + iy (или iy*n + ix, зависит от порядка)
            # В предыдущем коде было ravel_multi_index([y, x]).
            # Давайте держаться стандарта: J = row * width + col = iy * n + ix
            J = np.ravel_multi_index([final_iy, final_ix], (n, n))
            
            # --- Интерполяция ---
            vals_x = Vx[0]*c_final[0] + Vx[1]*c_final[1] + Vx[2]*c_final[2]
            vals_y = Vy[0]*c_final[0] + Vy[1]*c_final[1] + Vy[2]*c_final[2]
            
            np.add.at(Mx, J, vals_x)
            np.add.at(My, J, vals_y)
            np.add.at(Mnb, J, 1)
        
        # --- Нормализация ---
        # Мы заполняли J = iy * n + ix. 
        # reshape(n, n) даст массив, где 0-я ось - это iy (строки), 1-я - ix (столбцы).
        Mx = Mx.reshape(n, n)
        My = My.reshape(n, n)
        Mnb = Mnb.reshape(n, n)
        
        # Усреднение
        mask = Mnb > 0
        Mx[mask] /= Mnb[mask]
        My[mask] /= Mnb[mask]
        
        # NaN для пустот
        Mx[~mask] = np.nan
        My[~mask] = np.nan
        
        # В оригинале делался Transpose в конце.
        # Если вы хотите, чтобы CSV шел построчно (y=0, x=0..1; y=dy, x=0..1...), 
        # то текущий Mx (строки=y, столбцы=x) правильный.
        # Если вам нужно наоборот (строки=x), раскомментируйте .T
        # return Mx.T, My.T
        
        return Mx, My

    def interpolate(self, sampling_size=128, output_dir="interpolation_results"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"\n[Interpolator] Начинаем интерполяцию на сетку {sampling_size}x{sampling_size} (0..1)")
        
        faces = self.faces.T
        uv = self.uv_coords.T
        phys_x = self.phys_coords[:, 0]
        phys_y = self.phys_coords[:, 1]
        
        M_x, M_y = self._compute_dual_interp(faces, uv, phys_x, phys_y, sampling_size)
        
        # Сохранение
        path_x = os.path.join(output_dir, "x_data.csv")
        path_y = os.path.join(output_dir, "y_data.csv")
        
        # Flatten row-major (строка за строкой)
        with open(path_x, 'w', newline='') as f:
            csv.writer(f).writerow(M_x.flatten().tolist())
            
        with open(path_y, 'w', newline='') as f:
            csv.writer(f).writerow(M_y.flatten().tolist())
            
        # Визуализация
        print(f"  > Generating visual check...")
        flat_x = M_x.flatten()
        flat_y = M_y.flatten()
        mask_valid = ~np.isnan(flat_x)
        
        plt.figure(figsize=(6, 6))
        # Отрисуем только валидные точки
        plt.scatter(flat_x[mask_valid], flat_y[mask_valid], s=1, c='blue', alpha=0.1)
        plt.axis('equal')
        plt.title("Reconstructed Physical Domain")
        plt.savefig(os.path.join(output_dir, "reconstruction_check.png"))
        plt.close()
        
        print(f"  ✓ Готово. Результаты в папке '{output_dir}'")
        return M_x, M_y
