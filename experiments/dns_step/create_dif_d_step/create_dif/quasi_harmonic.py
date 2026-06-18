"""
Quasi-Harmonic Mapping (Zayer, Rössl, Seidel, 2005)

Реализация итерационного тензорного отображения на плоских триангулированных
областях. Сравнение с гармоническим (котангенсным) отображением.

Формулы из Discrete Tensorial Quasi-Harmonic Maps:
  - div(C · grad f) = 0  — quasi-harmonic equation
  - C = sqrtm(JᵀJ)       — квадратный корень из первой фундаментальной формы
  - Веса: w_ij = Σ_T (e_i⊥)ᵀ · C_T · (e_j⊥) / (4·A_T)

Детали:
  - Все треугольники предполагаются остроугольными (положительные веса)
  - Границы: Дирихле (те же, что и у гармонического отображения)
  - Итерации: 3-5 обычно достаточно
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import sqrtm
import time


class QuasiHarmonicMapper:
    """
    Итерационный quasi-harmonic mapper.

    Параметры
    X : ndarray (N, 2)
        Физические координаты вершин.
    F : ndarray (M, 3)
        Треугольники (индексы вершин).
    boundaries : dict
        Словарь с ключами 'inlet', 'outlet', 'bottom', 'top'.
        Каждый — список индексов граничных узлов.
    n_iter : int
        Количество итераций quasi-harmonic (default=5).
    """

    def __init__(self, X, F, boundaries, n_iter=5, mode='sqrt',
                 eperp_mode='log', damping=0.3):
        self.X = X              # физические координаты (N, 2)
        self.F = F              # треугольники (M, 3)
        self.boundaries = boundaries
        self.n_iter = n_iter
        self.n_nodes = X.shape[0]
        self.n_tri = F.shape[0]
        self.mode = mode        # 'sqrt', 'invsqrt', или 'inv'
        self.eperp_mode = eperp_mode  # 'log' — из парам. про-ва, 'phys' — из физ.
        self.damping = damping  # релаксация (0..1)
        # Текущее отображение: (ξ, η) для каждой вершины
        self.Y = None

        # Для замеров
        self.timings = {}

        # Предвычисленные площади треугольников в физическом пространстве
        self._phys_areas = None

    
    # 1. Якобиан и тензор C для треугольника
    

    def _compute_jacobian(self, phys_tri, log_tri):
        """
        Якобиан отображения (ξ,η) → (x,y) для треугольника.

        J = [∂x/∂ξ  ∂x/∂η]
            [∂y/∂ξ  ∂y/∂η]

        Треугольник линейный → J постоянен на треугольнике.
        Вычисляем через разности вершин:

          [Δx_ab  Δx_ac] = J · [Δξ_ab  Δξ_ac]
                               [Δη_ab  Δη_ac]

          J = [Δx] · [Δ(ξ,η)]^(-1)

        Parameters
        phys_tri : ndarray (3, 2)
            Физические координаты (x, y) трёх вершин треугольника.
        log_tri : ndarray (3, 2)
            Логические координаты (ξ, η) трёх вершин.

        Returns
        J : ndarray (2, 2)
            Якобиан.
        """
        # Базис в физическом пространстве
        dX = np.column_stack([phys_tri[1] - phys_tri[0],
                              phys_tri[2] - phys_tri[0]])  # (2, 2)

        # Базис в логическом пространстве
        dU = np.column_stack([log_tri[1] - log_tri[0],
                              log_tri[2] - log_tri[0]])  # (2, 2)

        # J = dX · inv(dU)
        try:
            J = dX @ np.linalg.inv(dU)
        except np.linalg.LinAlgError:
            # Вырожденный треугольник — возвращаем единичную матрицу
            J = np.eye(2)
            print("Вырождена")
        return J

    def _compute_tensor_C(self, phys_tri, log_tri, eps_reg=1e-10, mode='sqrt'):
        """
        Тензор C для quasi-harmonic уравнения div(C·grad f) = 0.

        По статье: C = sqrtm(JᵀJ)
        Но на плоскости может лучше работать C = inv(sqrtm(JᵀJ))

        Параметры
        mode : str
            'sqrt' — C = sqrtm(JᵀJ) как в статье (подавляет растяжение)
            'invsqrt' — C = inv(sqrtm(JᵀJ)) (усиливает сжатые области)
            'inv' — C = inv(JᵀJ)

        Returns
        C : ndarray (2, 2)
        """
        J = self._compute_jacobian(phys_tri, log_tri)

        # Первая фундаментальная форма
        I = J.T @ J  # (2, 2)
        # eps_reg * np.eye(2) добавляется для избежания вырожденности
        I_reg = I + eps_reg * np.eye(2)

        with np.errstate(all='ignore'):
            try:
                if mode == 'sqrt':
                    C = sqrtm(I_reg)
                elif mode == 'invsqrt':
                    C = np.linalg.inv(sqrtm(I_reg))
                elif mode == 'inv':
                    C = np.linalg.inv(I_reg)
                else:
                    C = np.eye(2)

                #Заменяет NaN → 1.0, +∞ → 1e12, -∞ → -1e12 
                C = np.real(np.nan_to_num(C, nan=1.0, posinf=1e12, neginf=-1e12))
            
            except Exception:
                C = np.eye(2)

        return C

    
    # 2. Площадь треугольника
    

    @staticmethod
    def _triangle_area(pts):
        """
        Площадь треугольника по координатам вершин (N, 2) или (3, 2).
        формула шнурков
        """
        return 0.5 * abs(
            pts[1, 0] * (pts[2, 1] - pts[0, 1]) +
            pts[2, 0] * (pts[0, 1] - pts[1, 1]) +
            pts[0, 0] * (pts[1, 1] - pts[2, 1])
        )

    def _compute_phys_areas(self):
        """Предвычисляем площади треугольников в физическом пространстве."""
        if self._phys_areas is None:
            areas = np.zeros(self.n_tri)
            for t in range(self.n_tri):
                v = self.F[t]
                areas[t] = self._triangle_area(self.X[v])
            # Защита от нулевых площадей
            areas = np.maximum(areas, 1e-15)
            self._phys_areas = areas
        return self._phys_areas

    
    # 3. Построение quasi-harmonic весов
    

    def _build_weights(self, Y):
        """
        Строит весовую матрицу W для quasi-harmonic отображения.
        Формула (12) из Zayer, Rössl, Seidel 2005.

        eperp_mode='log': перпендикуляры из параметрического (ξ,η) пространства,
                          площадь — |A_log|. Исходная формула из статьи.
        eperp_mode='phys': перпендикуляры из физического (x,y) пространства,
                          площадь — A_phys. Альтернативный вариант.

        Parameters
        Y : ndarray (N, 2)
            Текущие логические координаты (ξ, η).

        Returns
        W : csr_matrix (N, N)
            Весовая матрица (нулевая диагональ).
        """
        rows, cols, data = [], [], []
        n_flipped = 0
        phys_areas = self._compute_phys_areas()

        def ccw(v):
            """CCW поворот на 90°: (x,y) → (-y, x)."""
            return np.array([-v[1], v[0]])

        for t in range(self.n_tri):
            a, b, c = self.F[t]
            phys = self.X[[a, b, c]]    # (3, 2) физические
            log = Y[[a, b, c]]          # (3, 2) логические (ξ, η)

            # ----- Площади и ориентация -----
            A_log_signed = 0.5 * (
                log[1, 0] * (log[2, 1] - log[0, 1]) +
                log[2, 0] * (log[0, 1] - log[1, 1]) +
                log[0, 0] * (log[1, 1] - log[2, 1])
            )
            if abs(A_log_signed) < 1e-15:
                n_flipped += 1
                continue

            if A_log_signed < 0:
                n_flipped += 1
            A_log_abs = abs(A_log_signed)

            # ----- Тензор C -----
            C_tensor = self._compute_tensor_C(phys, log, mode=self.mode)

            # ----- Перпендикуляры и площадь -----
            if self.eperp_mode == 'phys':
                # Перпендикуляры из физического пространства
                e_ab = phys[1] - phys[0]
                e_bc = phys[2] - phys[1]
                e_ca = phys[0] - phys[2]
                # a←bc, b←ca, c←ab
                perps = [ccw(e_bc), ccw(e_ca), ccw(e_ab)]
                area = phys_areas[t]
                # phys треугольники всегда CCW, w = -K
                signed = -1.0
            else:
                # Перпендикуляры из параметрического (ξ,η) — формула (12)
                # Для CCW (A_log>0): w = -K, для CW (A_log<0): w = +K
                # (потому что градиент ∇φ_i = e_i⊥ / (2·A_log), знак A_log важен)
                p_ab = log[1] - log[0]
                p_bc = log[2] - log[1]
                p_ca = log[0] - log[2]
                perps = [ccw(p_bc), ccw(p_ca), ccw(p_ab)]
                area = A_log_abs
                signed = -1.0 if A_log_signed > 0 else 1.0

            if area < 1e-15:
                continue

            # ----- Веса: w = signed · (e_i⊥ · C · e_j⊥) / (4·A) -----
            for idx1, idx2 in [(0, 1), (1, 2), (2, 0)]:
                vi = [a, b, c][idx1]
                vj = [a, b, c][idx2]

                K_ij = (perps[idx1] @ (C_tensor @ perps[idx2])) / (4.0 * area)
                val = signed * K_ij

                if abs(val) < 1e-15:
                    continue

                rows.extend([vi, vj])
                cols.extend([vj, vi])
                data.extend([val, val])

        if len(data) == 0:
            return coo_matrix((self.n_nodes, self.n_nodes)).tocsr()

        return coo_matrix((data, (rows, cols)),
                          shape=(self.n_nodes, self.n_nodes)).tocsr()

    
    # 7. Диагностика: подсчёт перевёрнутых треугольников
    

    def _count_flipped(self, Y):
        """Сколько треугольников имеют отрицательную площадь в логическом пространстве."""
        n_flip = 0
        for t in range(self.n_tri):
            a, b, c = self.F[t]
            log = Y[[a, b, c]]
            # Знаковая площадь
            A_s = 0.5 * (
                log[1, 0] * (log[2, 1] - log[0, 1]) +
                log[2, 0] * (log[0, 1] - log[1, 1]) +
                log[0, 0] * (log[1, 1] - log[2, 1])
            )
            if A_s < -1e-12:
                n_flip += 1
        return n_flip

    
    # 6. Котангенсный Лапласиан (для начального приближения)
    

    def _build_cotangent_laplacian(self):
        """Строит матрицу Лапласа с котангенсными весами (дубль из create_map)."""
        v0 = self.X[self.F[:, 0]]
        v1 = self.X[self.F[:, 1]]
        v2 = self.X[self.F[:, 2]]

        def get_cotan(u, v):
            dot = np.sum(u * v, axis=1)
            cross = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
            cross = np.maximum(cross, 1e-12)
            return dot / cross

        cot0 = get_cotan(v1 - v0, v2 - v0)
        cot1 = get_cotan(v2 - v1, v0 - v1)
        cot2 = get_cotan(v0 - v2, v1 - v2)

        rows, cols, data = [], [], []
        for k, cot in enumerate([cot0, cot1, cot2]):
            i = self.F[:, (k + 1) % 3]
            j = self.F[:, (k + 2) % 3]
            rows.extend(i); cols.extend(j); data.extend(cot * 0.5)
            rows.extend(j); cols.extend(i); data.extend(cot * 0.5)

        W = coo_matrix((data, (rows, cols)),
                       shape=(self.n_nodes, self.n_nodes)).tocsr()
        return W

    
    # 7. Решение системы Дирихле
    

    def _solve_dirichlet(self, L, fixed_nodes, fixed_values):
        """
        Решает L·u = 0 с граничными условиями Дирихле (метод штрафа).

        Parameters
        L : csr_matrix (N, N)
            Матрица Лапласа (D - W).
        fixed_nodes : list[int]
            Индексы узлов с фиксированными значениями.
        fixed_values : list[float]
            Значения для фиксированных узлов.

        Returns
        u : ndarray (N,)
            Решение.
        """
        A = L.copy()
        b = np.zeros(self.n_nodes)
        penalty = 1e15

        idx = np.array(fixed_nodes, dtype=int)
        val = np.array(fixed_values, dtype=float)

        diag = A.diagonal().copy()
        diag[idx] += penalty
        A.setdiag(diag)
        b[idx] = val * penalty

        try:
            u = spsolve(A, b)
        except Exception:
            from scipy.sparse.linalg import bicgstab
            u, info = bicgstab(A, b, atol=1e-12, maxiter=10000)
            if info < 0:
                u = np.zeros(self.n_nodes)
        return u

    
    # 8. Основной цикл: гармоническое / quasi-harmonic
    

    def build_mapping(self, Y_init=None, verbose=True):
        """
        Построение отображения.
        Если n_iter=0 — решает Laplace(ξ)=0, Laplace(η)=0 (гармоническое).
        Если n_iter>0 — итерационное quasi-harmonic уточнение.

        Параметры
        Y_init : ndarray (N, 2) or None
            Начальное приближение (для QH — гармоническое).
            Если None и n_iter>0 — строится гармоническое и от него QH.
        verbose : bool

        Returns
        Y : ndarray (N, 2)
            Логические координаты (ξ, η).
        """
        # --- Гармоническое отображение (n_iter=0 или как init для QH) ---
        if Y_init is None or self.n_iter == 0:
            W_cot = self._build_cotangent_laplacian()
            deg = np.array(W_cot.sum(axis=1)).flatten()
            L_cot = (diags(deg) - W_cot).tocsr()

            fix_xi = list(self.boundaries['inlet']) + list(self.boundaries['outlet'])
            val_xi = [0.0]*len(self.boundaries['inlet']) + [1.0]*len(self.boundaries['outlet'])
            fix_eta = list(self.boundaries['bottom']) + list(self.boundaries['top'])
            val_eta = [0.0]*len(self.boundaries['bottom']) + [1.0]*len(self.boundaries['top'])

            xi = self._solve_dirichlet(L_cot, fix_xi, val_xi)
            eta = self._solve_dirichlet(L_cot, fix_eta, val_eta)
            Y = np.column_stack([xi, eta])
        else:
            Y = Y_init.copy()

        if self.n_iter == 0:
            self.Y = Y
            return Y

        # --- Quasi-harmonic итерации ---
        for it in range(self.n_iter):
            W = self._build_weights(Y)
            deg = np.array(W.sum(axis=1)).flatten()
            L = (diags(deg) - W).tocsr()

            fix_xi = list(self.boundaries['inlet']) + list(self.boundaries['outlet'])
            val_xi = [0.0]*len(self.boundaries['inlet']) + [1.0]*len(self.boundaries['outlet'])
            fix_eta = list(self.boundaries['bottom']) + list(self.boundaries['top'])
            val_eta = [0.0]*len(self.boundaries['bottom']) + [1.0]*len(self.boundaries['top'])

            xi = self._solve_dirichlet(L, fix_xi, val_xi)
            eta = self._solve_dirichlet(L, fix_eta, val_eta)
            Y_new = np.column_stack([xi, eta])

            if np.any(np.isnan(Y_new)) or np.any(np.isinf(Y_new)):
                if verbose:
                    print(f"  QH iter {it+1}/{self.n_iter}: NaN/Inf — стоп")
                break

            diff = np.max(np.abs(Y_new - Y))
            Y = self.damping * Y_new + (1 - self.damping) * Y

            if verbose:
                nf = self._count_flipped(Y)
                print(f"  QH iter {it+1}/{self.n_iter}: Δ={diff:.2e} flipped={nf}")

        self.Y = Y
        return Y


# Метрики искажения

def compute_distortions(X, Y, F):
    """
    Вычисляет угловые и площадные искажения для каждого треугольника.

    Параметры
    X : ndarray (N, 2)
        Физические координаты.
    Y : ndarray (N, 2)
        Логические координаты (ξ, η).
    F : ndarray (M, 3)
        Треугольники.

    Returns
    angle_dist : ndarray (M,)
        Угловое искажение: max(α/α', α'/α) для каждого угла треугольника.
        Значение 1 = без искажений, обрезано до [1, 100] для робастности.
    area_dist : ndarray (M,)
        Площадное искажение: отношение площадей (phys/log).
    """
    n_tri = F.shape[0]
    angle_dist = np.zeros(n_tri)
    area_phys = np.zeros(n_tri)
    area_log = np.zeros(n_tri)

    for t in range(n_tri):
        a, b, c = F[t]
        phys = X[[a, b, c]]
        log = Y[[a, b, c]]

        # Площади (берём абсолютные)
        A_phys_signed = 0.5 * (
            phys[1, 0] * (phys[2, 1] - phys[0, 1]) +
            phys[2, 0] * (phys[0, 1] - phys[1, 1]) +
            phys[0, 0] * (phys[1, 1] - phys[2, 1])
        )
        A_log_signed = 0.5 * (
            log[1, 0] * (log[2, 1] - log[0, 1]) +
            log[2, 0] * (log[0, 1] - log[1, 1]) +
            log[0, 0] * (log[1, 1] - log[2, 1])
        )
        A_phys = max(abs(A_phys_signed), 1e-15)
        A_log = max(abs(A_log_signed), 1e-15)
        area_phys[t] = A_phys
        area_log[t] = A_log

        # Угловое искажение — через соотношение углов (conformal distortion)
        # Для пары сопоставленных треугольников считаем distortion по формуле:
        #   D_conf = (σ_max / σ_min)  где σ — сингулярные числа J
        # что эквивалентно sqrt(κ_max / κ_min) для C = sqrtm(JᵀJ)
        #
        # Но проще — через собственные числа C, которые равны σ:
        # J = U·Σ·Vᵀ, C = U·Σ·Uᵀ
        # σ_max/σ_min = cond(J) — число обусловленности

        # Векторы физического базиса
        dX = np.column_stack([phys[1] - phys[0], phys[2] - phys[0]])
        dU = np.column_stack([log[1] - log[0], log[2] - log[0]])

        try:
            J = dX @ np.linalg.inv(dU)
            S = np.linalg.svd(J, compute_uv=False)
            # Защита от деления на ноль
            sigma_max = max(S[0], 1e-15)
            sigma_min = max(S[-1], 1e-15)
            cond = sigma_max / sigma_min
            # Ограничиваем разумным диапазоном
            angle_dist[t] = min(max(cond, 1.0), 1000.0)
        except np.linalg.LinAlgError:
            angle_dist[t] = 1.0

    # Площадное искажение (отношение площадей)
    area_dist = area_phys / (area_log + 1e-15)
    # Нормализуем так, чтобы среднее = 1
    mean_area_dist = np.mean(area_dist)
    if mean_area_dist > 1e-15:
        area_dist = area_dist / mean_area_dist

    return angle_dist, area_dist


def summarize_distortions(angle_dist, area_dist):
    """
    Возвращает сводную статистику по искажениям.

    Returns
    dict с ключами:
      angle_max, angle_mean, angle_median, angle_std,
      area_max, area_mean, area_median, area_std
    """
    return {
        'angle_max': float(np.max(angle_dist)),
        'angle_mean': float(np.mean(angle_dist)),
        'angle_median': float(np.median(angle_dist)),
        'angle_std': float(np.std(angle_dist)),
        'area_max': float(np.max(area_dist)),
        'area_mean': float(np.mean(area_dist)),
        'area_median': float(np.median(area_dist)),
        'area_std': float(np.std(area_dist)),
    }


# Функция сравнения: гармоническое vs quasi-гармоническое

def compare_mappings(X, F, boundaries, n_iter=5, verbose=True):
    """
    Запускает оба метода на одной геометрии и возвращает результат сравнения.

    Parameters
    X : ndarray (N, 2)
    F : ndarray (M, 3)
    boundaries : dict
    n_iter : int
        Число quasi-harmonic итераций.
    verbose : bool

    Returns
    result : dict
        Содержит Y_harm, Y_qharm, distortion_harm, distortion_qharm, timings.
    """
    # --- 1. Гармоническое отображение ---
    if verbose:
        print("=" * 55)
        print("  ГАРМОНИЧЕСКОЕ ОТОБРАЖЕНИЕ (котангенсные веса)")
        print("=" * 55)

    t_harm_start = time.time()
    mapper_harm = QuasiHarmonicMapper(X, F, boundaries, n_iter=0)
    Y_harm = mapper_harm.build_mapping(Y_init=None, verbose=verbose)
    t_harm = time.time() - t_harm_start
    if verbose:
        print(f"  Время гармонического: {t_harm:.4f} сек\n")

    dist_harm_angle, dist_harm_area = compute_distortions(X, Y_harm, F)
    stats_harm = summarize_distortions(dist_harm_angle, dist_harm_area)

    # --- 2. Quasi-гармоническое отображение ---
    if verbose:
        print("=" * 55)
        print("  КВАЗИ-ГАРМОНИЧЕСКОЕ ОТОБРАЖЕНИЕ")
        print("=" * 55)

    t_qh_start = time.time()
    mapper_qh = QuasiHarmonicMapper(X, F, boundaries, n_iter=n_iter,
                                     mode='sqrt', eperp_mode='log')
    Y_qh = mapper_qh.build_mapping(Y_init=Y_harm, verbose=verbose)
    t_qh = time.time() - t_qh_start
    if verbose:
        print(f"  Время quasi-гармонического: {t_qh:.4f} сек\n")

    dist_qh_angle, dist_qh_area = compute_distortions(X, Y_qh, F)
    stats_qh = summarize_distortions(dist_qh_angle, dist_qh_area)

    # --- 3. Сводка ---
    if verbose:
        print("=" * 55)
        print("  СРАВНЕНИЕ ИСКАЖЕНИЙ")
        print("=" * 55)
        print(f"{'Метрика':<25} {'Гармоническое':<18} {'Quasi-Harmonic':<18}")
        print("-" * 55)
        print(f"{'Угол: max':<25} {stats_harm['angle_max']:<18.4f} {stats_qh['angle_max']:<18.4f}")
        print(f"{'Угол: mean':<25} {stats_harm['angle_mean']:<18.4f} {stats_qh['angle_mean']:<18.4f}")
        print(f"{'Угол: median':<25} {stats_harm['angle_median']:<18.4f} {stats_qh['angle_median']:<18.4f}")
        print(f"{'Площадь: max':<25} {stats_harm['area_max']:<18.4f} {stats_qh['area_max']:<18.4f}")
        print(f"{'Площадь: mean':<25} {stats_harm['area_mean']:<18.4f} {stats_qh['area_mean']:<18.4f}")
        print(f"{'Площадь: median':<25} {stats_harm['area_median']:<18.4f} {stats_qh['area_median']:<18.4f}")
        print(f"{'Время (сек)':<25} {t_harm:<18.4f} {t_qh:<18.4f}")
        print("=" * 55)

    return {
        'Y_harm': Y_harm,
        'Y_qh': Y_qh,
        'dist_harm': (dist_harm_angle, dist_harm_area),
        'dist_qh': (dist_qh_angle, dist_qh_area),
        'stats_harm': stats_harm,
        'stats_qh': stats_qh,
        'timings': {'harmonic': t_harm, 'quasi_harmonic': t_qh,
                     'qh_iterations': mapper_qh.timings},
    }


# Тест при запуске модуля

if __name__ == "__main__":
    import os
    import sys

    # Путь к тестовым данным
    test_dir = os.path.join(os.path.dirname(__file__),
                            "../test_domains")
    if not os.path.isdir(test_dir):
        print(f"Директория {test_dir} не найдена.")
        print("Сначала сгенерируй сетки через generate_1000.py")
        sys.exit(1)

    # Тест на первом файле
    files = sorted([f for f in os.listdir(test_dir) if f.endswith(".msh")])
    if not files:
        print("Нет .msh файлов в", test_dir)
        sys.exit(1)

    test_file = os.path.join(test_dir, files[0])
    print(f"Тестовый файл: {test_file}")

    try:
        from read_msh import read_msh_file
    except ImportError:
        print("Не найден read_msh.py")
        sys.exit(1)

    X, F, bnds = read_msh_file(test_file)
    if X is None:
        sys.exit(1)

    print(f"Узлов: {X.shape[0]}, треугольников: {F.shape[0]}")
    print(f"Границы: inlet={len(bnds['inlet'])}, outlet={len(bnds['outlet'])}, "
          f"bottom={len(bnds['bottom'])}, top={len(bnds['top'])}")

    # Запускаем сравнение
    result = compare_mappings(X, F, bnds, n_iter=15, verbose=True)
