import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import matplotlib.tri as tri
from scipy import sparse
from scipy.sparse.linalg import spsolve

###############################################################################
# НАСТРОЙКИ ГЕНЕРАЦИИ СЕТКИ И ПАРАМЕТРОВ ДАРСИ
###############################################################################

# Основные параметры
TOTAL_POINTS = 150
BOUNDARY_POINTS = 35
SEED = 42

# Параметры пятиугольника
NUM_VERTICES = 5
RADIUS_MEAN = 4.5
RADIUS_VARIATION = 0.5

# Параметры уравнения Дарси (как в статье)
COF_RANGE = [0.2, 0.8]  # диапазон для cof1, cof2
SCALE_C_RANGE_SMALL = [0.01, 1.0]  # малый масштаб
SCALE_C_RANGE_LARGE = [1.0, 4.0]   # большой масштаб
SCALE_C_PROB = 0.5     # вероятность выбора малого масштаба

# Параметры файлов
OBJ_FILENAME = 'darcy_pentagon.obj'
PERMEABILITY_FILENAME = 'C.csv'
PRESSURE_FILENAME = 'U.csv'
COORDS_X_FILENAME = 'x_data.csv'
COORDS_Y_FILENAME = 'y_data.csv'

###############################################################################
# ФУНКЦИИ ДЛЯ ГЕНЕРАЦИИ СЕТКИ
###############################################################################

def point_in_polygon(x, y, poly):
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def create_enhanced_pentagon(num_points=TOTAL_POINTS, seed=SEED):
    np.random.seed(seed)
    
    n_vertices = NUM_VERTICES
    base_angles = np.linspace(0, 2*np.pi, n_vertices, endpoint=False)
    radii = RADIUS_MEAN + RADIUS_VARIATION * np.random.random(n_vertices)
    
    vertices = np.column_stack([radii * np.cos(base_angles), radii * np.sin(base_angles)])
    
    points_list = []
    n_boundary_points = BOUNDARY_POINTS
    
    # Точки на границе
    for i in range(n_vertices):
        start = vertices[i]
        end = vertices[(i + 1) % n_vertices]
        t = np.linspace(0, 1, n_boundary_points // n_vertices + 2)[1:-1]
        boundary_points = start + t[:, np.newaxis] * (end - start)
        points_list.extend(boundary_points)
    
    # Внутренние точки
    remaining_points = num_points - len(points_list)
    bbox = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    
    interior_points = []
    grid_size = int(np.sqrt(remaining_points) * 2)
    x_grid = np.linspace(bbox[0, 0], bbox[1, 0], grid_size)
    y_grid = np.linspace(bbox[0, 1], bbox[1, 1], grid_size)
    
    for x in x_grid:
        for y in y_grid:
            if point_in_polygon(x, y, vertices):
                interior_points.append([x, y])
    
    if len(interior_points) < remaining_points:
        additional_needed = remaining_points - len(interior_points)
        added = 0
        max_attempts = additional_needed * 10
        
        while added < additional_needed:
            x = np.random.uniform(bbox[0, 0], bbox[1, 0])
            y = np.random.uniform(bbox[0, 1], bbox[1, 1])
            
            if point_in_polygon(x, y, vertices):
                interior_points.append([x, y])
                added += 1
    
    all_points = np.vstack([vertices, points_list, interior_points])
    return all_points, vertices

###############################################################################
# ФУНКЦИИ ДЛЯ УРАВНЕНИЯ ДАРСИ (МЕТОД 2)
###############################################################################

def generate_random_parameters():
    """
    Генерирует случайные параметры как в статье
    """
    # cof1, cof2 в диапазоне [0.2, 0.8]
    cof = np.random.uniform(COF_RANGE[0], COF_RANGE[1], 2)
    
    # scale_c - как в статье
    if np.random.random() > SCALE_C_PROB:
        scale_c = np.random.uniform(SCALE_C_RANGE_SMALL[0], SCALE_C_RANGE_SMALL[1])
    else:
        scale_c = np.random.uniform(SCALE_C_RANGE_LARGE[0], SCALE_C_RANGE_LARGE[1])
    
    # scaled параметр (аналог их scaled)
    scaled = RADIUS_MEAN
    
    return cof, scale_c, scaled

def darcy_coefficient_function(x, y, cof, scaled, scale_c):
    """
    Коэффициент проницаемости C(x,y) по формуле из статьи:
    c = cof1*sin(pi*x/(scaled*10)) - cof2*(x/scaled)*(x/scaled - 10) + 2
    """
    x_norm = x / scaled
    c = cof[0] * np.sin(np.pi * x / (scaled * 10)) - cof[1] * x_norm * (x_norm - 10) + 2
    return c * scale_c

def compute_element_stiffness_matrix(points, element, cof, scaled, scale_c):
    """
    Вычисляет локальную матрицу жесткости для одного треугольника
    """
    p1, p2, p3 = points[element]
    
    # Площадь треугольника
    area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
    
    if area < 1e-12:
        return np.zeros((3, 3)), np.zeros(3)
    
    # Градиенты базисных функций
    grad_phi = np.array([
        [p2[1]-p3[1], p3[1]-p1[1], p1[1]-p2[1]],
        [p3[0]-p2[0], p1[0]-p3[0], p2[0]-p1[0]]
    ]) / (2 * area)
    
    # Центр элемента для вычисления коэффициента
    center_x = (p1[0] + p2[0] + p3[0]) / 3
    center_y = (p1[1] + p2[1] + p3[1]) / 3
    
    # Коэффициент проницаемости в центре элемента
    C_elem = darcy_coefficient_function(center_x, center_y, cof, scaled, scale_c)
    
    # Локальная матрица жесткости
    local_A = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            local_A[i, j] = C_elem * area * np.dot(grad_phi[:, i], grad_phi[:, j])
    
    # Локальный вектор правой части (f=1 как в статье)
    local_b = (area / 3) * np.ones(3)
    
    return local_A, local_b

def solve_darcy_fem(points, triangles, cof, scale_c, scaled):
    """
    Решает уравнение Дарси методом конечных элементов на нашей триангуляции
    Уравнение: -∇·(C∇u) = 1 с u=0 на границе
    """
    n_points = len(points)
    n_triangles = len(triangles)
    
    # Инициализация глобальной матрицы и вектора
    A = sparse.lil_matrix((n_points, n_points))
    b = np.zeros(n_points)
    
    # Сборка глобальной системы
    for element in triangles:
        local_A, local_b = compute_element_stiffness_matrix(points, element, cof, scaled, scale_c)
        
        # Добавление в глобальную систему
        for i, local_i in enumerate(element):
            for j, local_j in enumerate(element):
                A[local_i, local_j] += local_A[i, j]
            b[local_i] += local_b[i]
    
    # Определение граничных узлов (все точки на границе пятиугольника)
    boundary_indices = list(range(NUM_VERTICES))  # вершины
    # Добавляем точки на ребрах
    for i in range(NUM_VERTICES, NUM_VERTICES + BOUNDARY_POINTS):
        boundary_indices.append(i)
    
    # Применение граничных условий Дирихле (u=0)
    for idx in boundary_indices:
        # Зануляем строку и ставим 1 на диагонали
        A[idx, :] = 0
        A[idx, idx] = 1
        b[idx] = 0
    
    # Решение системы
    A_csr = A.tocsr()
    pressure = spsolve(A_csr, b)
    
    return pressure

def compute_permeability_field(points, cof, scaled, scale_c):
    """
    Вычисляет поле проницаемости C во всех точках
    """
    permeability = np.zeros(len(points))
    for i, (x, y) in enumerate(points):
        permeability[i] = darcy_coefficient_function(x, y, cof, scaled, scale_c)
    return permeability

def export_darcy_data(points, permeability, pressure):
    """
    Экспорт данных в формате CSV
    """
    # Координаты
    np.savetxt(COORDS_X_FILENAME, points[:, 0], delimiter=',')
    np.savetxt(COORDS_Y_FILENAME, points[:, 1], delimiter=',')
    
    # Поля
    np.savetxt(PERMEABILITY_FILENAME, permeability, delimiter=',')
    np.savetxt(PRESSURE_FILENAME, pressure, delimiter=',')
    
    print("Данные экспортированы:")
    print(f"  Координаты: {COORDS_X_FILENAME}, {COORDS_Y_FILENAME}")
    print(f"  Проницаемость: {PERMEABILITY_FILENAME}")
    print(f"  Давление: {PRESSURE_FILENAME}")

###############################################################################
# ОСНОВНОЙ КОД
###############################################################################

# Генерация геометрии
print("Создание пятиугольника и триангуляции...")
points, boundary = create_enhanced_pentagon()
triangulation = Delaunay(points)

# Создаем триангуляцию для matplotlib
tri_mesh = tri.Triangulation(points[:, 0], points[:, 1], triangulation.simplices)

# Случайные параметры (как в статье)
print("Генерация случайных параметров...")
cof, scale_c, scaled = generate_random_parameters()

print(f"Параметры: cof={cof}, scale_c={scale_c:.3f}, scaled={scaled}")

# Вычисление поля проницаемости
print("Вычисление поля проницаемости C...")
permeability = compute_permeability_field(points, cof, scaled, scale_c)

# Решение уравнения Дарси
print("Решение уравнения Дарси методом FEM...")
pressure = solve_darcy_fem(points, triangulation.simplices, cof, scale_c, scaled)

# Экспорт данных
print("Экспорт данных...")
export_darcy_data(points, permeability, pressure)

# Визуализация (исправленная)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Триангуляция
axes[0,0].triplot(tri_mesh, 'b-', alpha=0.5, linewidth=0.5)
axes[0,0].plot(points[:, 0], points[:, 1], 'o', markersize=2, alpha=0.6)
axes[0,0].plot(np.append(boundary[:, 0], boundary[0, 0]), 
               np.append(boundary[:, 1], boundary[0, 1]), 'r-', linewidth=2)
axes[0,0].set_title('Триангуляция сетки')
axes[0,0].axis('equal')

# 2. Поле проницаемости C
tcf1 = axes[0,1].tricontourf(tri_mesh, permeability, levels=50, cmap='viridis')
plt.colorbar(tcf1, ax=axes[0,1], label='Проницаемость C')
axes[0,1].set_title('Поле проницаемости C(x,y)')
axes[0,1].axis('equal')

# 3. Поле давления U
tcf2 = axes[1,0].tricontourf(tri_mesh, pressure, levels=50, cmap='plasma')
plt.colorbar(tcf2, ax=axes[1,0], label='Давление U')
axes[1,0].set_title('Поле давления U(x,y)')
axes[1,0].axis('equal')

# 4. Распределение значений
axes[1,1].hist(permeability, bins=30, alpha=0.7, label='Проницаемость C', color='green', density=True)
axes[1,1].hist(pressure, bins=30, alpha=0.7, label='Давление U', color='orange', density=True)
axes[1,1].set_xlabel('Значение')
axes[1,1].set_ylabel('Плотность вероятности')
axes[1,1].legend()
axes[1,1].set_title('Распределение значений')

plt.tight_layout()
plt.savefig('darcy_fem_solution.png', dpi=300, bbox_inches='tight')
plt.show()

# Дополнительная визуализация с tripcolor (альтернатива)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Триангуляция
axes[0].triplot(tri_mesh, 'b-', alpha=0.3, linewidth=0.5)
axes[0].plot(np.append(boundary[:, 0], boundary[0, 0]), 
             np.append(boundary[:, 1], boundary[0, 1]), 'r-', linewidth=2)
axes[0].set_title('Триангуляция')
axes[0].axis('equal')

# 2. Проницаемость (tripcolor)
tpc1 = axes[1].tripcolor(tri_mesh, permeability, cmap='viridis', shading='flat')
plt.colorbar(tpc1, ax=axes[1], label='Проницаемость C')
axes[1].set_title('Проницаемость C (tripcolor)')
axes[1].axis('equal')

# 3. Давление (tripcolor)
tpc2 = axes[2].tripcolor(tri_mesh, pressure, cmap='plasma', shading='flat')
plt.colorbar(tpc2, ax=axes[2], label='Давление U')
axes[2].set_title('Давление U (tripcolor)')
axes[2].axis('equal')

plt.tight_layout()
plt.savefig('darcy_fem_tripcolor.png', dpi=300, bbox_inches='tight')
plt.show()

# Статистика
print("\n" + "="*60)
print("СТАТИСТИКА РЕШЕНИЯ УРАВНЕНИЯ ДАРСИ")
print("="*60)
print(f"Количество точек: {len(points)}")
print(f"Количество треугольников: {len(triangulation.simplices)}")
print(f"Проницаемость C: min={permeability.min():.3f}, max={permeability.max():.3f}, mean={permeability.mean():.3f}")
print(f"Давление U: min={pressure.min():.3f}, max={pressure.max():.3f}, mean={pressure.mean():.3f}")

# Проверка граничных условий
boundary_indices = list(range(NUM_VERTICES + BOUNDARY_POINTS))
boundary_pressure = pressure[boundary_indices]
print(f"Давление на границе: max|U|={np.max(np.abs(boundary_pressure)):.2e}")

# Пример данных
print("\nПервые 5 точек данных:")
print("X\t\tY\t\tC\t\tU")
for i in range(5):
    print(f"{points[i,0]:.3f}\t{points[i,1]:.3f}\t{permeability[i]:.3f}\t{pressure[i]:.3f}")

print(f"\nФайлы успешно созданы:")
print(f"- {COORDS_X_FILENAME} (координаты X)")
print(f"- {COORDS_Y_FILENAME} (координаты Y)") 
print(f"- {PERMEABILITY_FILENAME} (проницаемость C)")
print(f"- {PRESSURE_FILENAME} (давление U)")
print(f"- darcy_fem_solution.png (визуализация)")
print(f"- darcy_fem_tripcolor.png (альтернативная визуализация)")
