import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')

###############################################################################
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ИЗ СТАТЬИ (адаптированные)
###############################################################################

def read_obj(obj_name):
    """
    Чтение OBJ файла - адаптированная версия
    """
    vertices = []
    faces = []
    
    with open(obj_name, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # vertex data
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                # face data
                parts = line.strip().split()
                face = []
                for part in parts[1:]:
                    # handle format "v/vt/vn" or just "v"
                    vertex_index = part.split('/')[0]
                    face.append(int(vertex_index) - 1)  # OBJ indices start at 1
                faces.append(face)
    
    vertices = np.array(vertices).T  # shape: (3, n_vertices)
    faces = np.array(faces).T        # shape: (3, n_faces)
    
    return vertices, faces

def compute_boundary(F, is_closed=True, start_point=0):
    """
    Вычисление границы из триангуляции - адаптированная версия
    """
    edges = {}
    
    # Собираем все ребра
    for face in F.T:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1)%3]]))
            if edge in edges:
                edges[edge] += 1
            else:
                edges[edge] = 1
    
    # Граничные ребра встречаются только один раз
    boundary_edges = [edge for edge, count in edges.items() if count == 1]
    
    # Строим упорядоченную границу
    if not boundary_edges:
        return np.array([])
    
    # Создаем граф границы
    graph = {}
    for edge in boundary_edges:
        u, v = edge
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
        graph[v].append(u)
    
    # Находим граничный цикл
    boundary = []
    current = start_point
    visited = set()
    
    while current not in visited:
        visited.add(current)
        boundary.append(current)
        neighbors = [n for n in graph[current] if n not in visited]
        if not neighbors:
            break
        current = neighbors[0]
    
    return np.array(boundary)

def calculate_angle(point1, point2):
    """
    Вычисление угла между двумя точками
    """
    x1, y1 = point1
    x2, y2 = point2
    return np.arctan2(y2 - y1, x2 - x1)

def find_corner_points(B, X):
    """
    Нахождение угловых точек на границе
    """
    corner_points = []
    corner_indices = []
    
    B_points = np.array([X[0, B], X[1, B]]).T
    
    for i in range(len(B)):
        # Точка i-1, i, i+1
        prev_point = B_points[i-1]
        current_point = B_points[i]
        if i == len(B)-1:
            next_point = B_points[0]
        else:
            next_point = B_points[i+1]
        
        # Векторы
        v1 = prev_point - current_point
        v2 = next_point - current_point
        
        # Нормализация
        v1 = v1 / (np.linalg.norm(v1) + 1e-8)
        v2 = v2 / (np.linalg.norm(v2) + 1e-8)
        
        # Угол между векторами
        dot_product = np.dot(v1, v2)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)
        
        # Если угол значительно отличается от 180 градусов - это угол
        if angle < 2.0:  # примерно 115 градусов
            corner_points.append(current_point)
            corner_indices.append(B[i])
    
    return corner_points, corner_indices

def compute_triang_interp(F, Y, values, q):
    """
    Интерполяция значений на регулярную сетку
    """
    # Создаем регулярную сетку в единичном квадрате
    x_grid = np.linspace(0, 1, q)
    y_grid = np.linspace(0, 1, q)
    XX, YY = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([XX.ravel(), YY.ravel()]).T
    
    # Интерполируем значения на сетку
    from scipy.interpolate import LinearNDInterpolator
    interp = LinearNDInterpolator(Y.T, values)
    interpolated = interp(grid_points)
    
    # Возвращаем в форме изображения
    return interpolated.reshape(q, q)

###############################################################################
# ОСНОВНАЯ ФУНКЦИЯ ДИФФЕОМОРФИЗМА
###############################################################################

def create_harmonic_map(obj_name, sampling_size=128, output_prefix="mapped"):
    """
    Создает гармоническое отображение OBJ файла в единичный квадрат
    """
    # Чтение OBJ файла
    [X, F] = read_obj(obj_name)
    F = F.astype('int64')
    
    # Центрирование (опционально)
    x_max = X[0].max()
    y_max = X[1].max()
    X[0] = X[0] - x_max/2
    X[1] = X[1] - y_max/2
    
    n = X.shape[1]  # количество вершин
    
    # Находим границу
    SB = compute_boundary(F, True, 0)
    if len(SB) == 0:
        raise ValueError("Не удалось найти границу")
    
    # Находим угловые точки
    B_points = np.array([X[0, SB], X[1, SB]]).T
    corners_points, corners_indices = find_corner_points(SB, X)
    
    # Если не нашли 4 угла, используем равномерное распределение
    if len(corners_indices) != 4:
        print(f"Найдено {len(corners_indices)} углов, используем равномерное распределение")
        step = len(SB) // 4
        corners_indices = [SB[i*step] for i in range(4)]
    
    # Находим индексы углов в границе
    corner_boundary_indices = [np.where(SB == corner)[0][0] for corner in corners_indices]
    corner_boundary_indices.sort()
    
    # Строим матрицу Лапласа
    W = sparse.lil_matrix((n, n))
    
    for i in range(3):
        i2 = (i+1) % 3
        i3 = (i+2) % 3
        
        u = X[:, F[i2, :]] - X[:, F[i, :]]
        v = X[:, F[i3, :]] - X[:, F[i, :]]
        
        # Нормализация векторов
        u_norm = np.sqrt(np.sum(u**2, 0))
        v_norm = np.sqrt(np.sum(v**2, 0))
        u = u / np.maximum(u_norm, 1e-8)
        v = v / np.maximum(v_norm, 1e-8)
        
        # Вычисление углов
        cos_angles = np.sum(u * v, 0)
        cos_angles = np.clip(cos_angles, -0.999, 0.999)
        alpha = 1.0 / np.tan(np.arccos(cos_angles))
        alpha = np.maximum(alpha, 1e-2)
        
        for j in range(F.shape[1]):
            W[F[i2, j], F[i3, j]] += alpha[j]
            W[F[i3, j], F[i2, j]] += alpha[j]
    
    # Диагональная матрица и Лапласиан
    d = W.sum(axis=1).A.ravel()
    D = sparse.diags(d, 0)
    L = D - W
    
    # Подготовка системы с граничными условиями
    L1 = L.tolil()
    L1[SB, :] = 0
    for i in SB:
        L1[i, i] = 1
    
    # Граничные условия: отображаем на единичный квадрат
    R = np.zeros((2, n))
    
    # Равномерно распределяем точки границы на стороны квадрата
    t = np.arange(len(SB)) / len(SB)
    Z_boundary = np.zeros((2, len(SB)))
    
    # Распределяем точки по сторонам квадрата
    side_lengths = []
    for i in range(4):
        start_idx = corner_boundary_indices[i]
        if i == 3:
            end_idx = len(SB)
        else:
            end_idx = corner_boundary_indices[i+1]
        side_lengths.append(end_idx - start_idx)
    
    total_sides = sum(side_lengths)
    
    # Задаем координаты для каждой стороны
    current_idx = 0
    for i in range(4):
        side_len = side_lengths[i]
        if side_len == 0:
            continue
            
        t_side = np.linspace(0, 1, side_len, endpoint=False)
        
        if i == 0:  # нижняя сторона
            Z_boundary[0, current_idx:current_idx+side_len] = t_side
            Z_boundary[1, current_idx:current_idx+side_len] = 0
        elif i == 1:  # правая сторона
            Z_boundary[0, current_idx:current_idx+side_len] = 1
            Z_boundary[1, current_idx:current_idx+side_len] = t_side
        elif i == 2:  # верхняя сторона
            Z_boundary[0, current_idx:current_idx+side_len] = 1 - t_side
            Z_boundary[1, current_idx:current_idx+side_len] = 1
        elif i == 3:  # левая сторона
            Z_boundary[0, current_idx:current_idx+side_len] = 0
            Z_boundary[1, current_idx:current_idx+side_len] = 1 - t_side
        
        current_idx += side_len
    
    R[:, SB] = Z_boundary
    
    # Решение системы
    Y = np.zeros((2, n))
    L1_csr = L1.tocsr()
    Y[0, :] = spsolve(L1_csr, R[0, :])
    Y[1, :] = spsolve(L1_csr, R[1, :])
    
    # Вычисление энергии отображения
    E = 0.5 * np.sum((L.dot(Y.T))**2)
    
    # Интерполяция на регулярную сетку
    q = sampling_size
    M = np.zeros((q, q, 3))
    for i in range(3):
        M[:, :, i] = compute_triang_interp(F, Y, X[i, :], q)
    
    # Нормализация для визуализации
    M_image = M.copy()
    for i in range(3):
        M_min = M_image[:, :, i].min()
        M_max = M_image[:, :, i].max()
        if M_max > M_min:
            M_image[:, :, i] = (M_image[:, :, i] - M_min) / (M_max - M_min)
    
    # Визуализация
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.triplot(Y[0, :], Y[1, :], F.T, 'b-', alpha=0.5, linewidth=0.5)
    plt.plot(Y[0, SB], Y[1, SB], 'r-', linewidth=2)
    plt.title('Отображение в квадрат')
    plt.axis('equal')
    
    plt.subplot(1, 3, 2)
    plt.imshow(M_image)
    plt.title('Интерполированное изображение')
    
    plt.subplot(1, 3, 3)
    plt.scatter(Y[0, :], Y[1, :], s=1, alpha=0.6)
    plt.title('Точки отображения')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_mapping.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Сохранение данных
    x_data = M[:, :, 0].flatten()
    y_data = M[:, :, 1].flatten()
    
    np.savetxt('x_data.csv', x_data, delimiter=',')
    np.savetxt('y_data.csv', y_data, delimiter=',')
    
    print(f"Диффеоморфизм завершен:")
    print(f"- Энергия отображения: {E:.6f}")
    print(f"- Размер выборки: {sampling_size}×{sampling_size}")
    print(f"- Файлы сохранены: x_data.csv, y_data.csv, {output_prefix}_mapping.png")
    
    return Y, M, E

###############################################################################
# ИСПОЛЬЗОВАНИЕ
###############################################################################

if __name__ == "__main__":
    # Пример использования
    obj_filename = "darcy_pentagon.obj"  # или "darcy_heptagon.obj"
    
    try:
        Y, M, E = create_harmonic_map(obj_filename, sampling_size=128, output_prefix="pentagon")
        print("Диффеоморфизм успешно построен!")
    except Exception as e:
        print(f"Ошибка: {e}")
        print("Убедитесь, что файл OBJ существует и имеет правильный формат")
