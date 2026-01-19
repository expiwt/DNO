# -*- coding: utf-8 -*-
import numpy as np
import scipy as scp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import csv
import os
from scipy.sparse import linalg
import warnings
warnings.filterwarnings('ignore')

# Импортируем модули напрямую, избегая циклических импортов
from map_utils.read_obj import read_obj
from map_utils.compute_triang_interp import compute_triang_interp

# Импортируем напрямую из файлов
from nt_toolbox.compute_boundary import compute_boundary
def calculate_angle(point1, point2):
    """Вычисляет угол между двумя точками"""
    x1, y1 = point1
    x2, y2 = point2
    return np.arctan2(y2 - y1, x2 - x1)

def find_corner_points(B):
    """Находит угловые точки на границе"""
    corner_points = []
    corner = []
    for i in range(len(B)):
        angle1 = calculate_angle(B[i - 1], B[i])
        if i == len(B)-1:
            angle2 = calculate_angle(B[i], B[0])
        else:
            angle2 = calculate_angle(B[i], B[i + 1])
        angle_difference = np.abs(angle2 - angle1)
        if angle_difference > np.pi / 6: 
            corner_points.append(B[i])
            corner.append(i)
    return corner_points, corner

def create_image_quad(obj_name, sampling_size, objfilename, z=0, geo_data=None):

    """
    Основная функция с использованием данных из geo_data_quad.csv
    """
    [X, F] = read_obj(obj_name)
    print(f"  Загружено: {X.shape[1]} вершин, {F.shape[1]} треугольников")
    
    F = F - 1
    F = np.unique(F, axis=1)
    
    # Находим границы области
    x_min = X[0].min()
    x_max = X[0].max() 
    y_min = X[1].min()
    y_max = X[1].max()
    
    print(f"  Границы: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")
    
    # ОПРЕДЕЛЕНИЕ УГЛОВЫХ ТОЧЕК
    if geo_data is not None:
        # Используем точные координаты из geo_data_quad.csv
        
        # Ищем ближайшие вершины к заданным координатам
        def find_closest_point(target_x, target_y):
            distances = np.sqrt((X[0] - target_x)**2 + (X[1] - target_y)**2)
            return np.argmin(distances)
        
        # Находим точки по координатам из CSV
        point_1 = find_closest_point(geo_data['x_left_down'], y_min)  # левый нижний
        point_2 = find_closest_point(geo_data['x_right_down'], y_min) # правый нижний
        point_3 = find_closest_point(geo_data['x_right_up'], geo_data['y_right_up']) # правый верхний
        point_4 = find_closest_point(geo_data['x_left_up'], geo_data['y_left_up'])   # левый верхний
        
    # Проверяем, что точки разные
    corners = [point_1, point_2, point_3, point_4]


    if len(set(corners)) < 4:
        print(f"    Предупреждение: найдены повторяющиеся угловые точки: {corners}")
        # Используем резервную логику для поиска уникальных точек
        all_points = set(range(X.shape[1]))
        remaining = list(all_points - set(corners))
        if len(remaining) > 0:
            point_4 = remaining[0]  # берем первую доступную точку
  
    
    
    n = X.shape[1]  # количество вершин
    
    # Находим граничные точки с помощью функции из статьи
    SB = compute_boundary(F, True, point_1)
    print(f"  Граничных точек: {len(SB)}")
    
    # Получаем координаты граничных точек
    B_points = np.transpose(X[0:2, SB])
    
    # ... остальная часть функции без изменений ...
    
    B = SB
    p = len(B)
    
    print("✓ Подготовка границы завершена")
    
    # Строим матрицу весов (cotangent Laplacian) как в статье
    W = sparse.coo_matrix((n, n))
    
    print("  Строим матрицу Лапласиана...")
    
    for i in range(3):  # для каждого угла треугольника
        i2 = (i + 1) % 3
        i3 = (i + 2) % 3
        F = F.astype('int64')
        
        # Векторы сторон треугольника
        u = X[:, F[i2, :]] - X[:, F[i, :]]
        v = X[:, F[i3, :]] - X[:, F[i, :]]
        
        # Нормализация векторов
        u_norm = np.sqrt(np.sum(u**2, axis=0))
        v_norm = np.sqrt(np.sum(v**2, axis=0))
        
        u = u / np.tile(u_norm, (3, 1))
        v = v / np.tile(v_norm, (3, 1))
        
        # Вычисляем косинус угла
        cos_angle = np.sum(u * v, axis=0)
        cos_angle = np.clip(cos_angle, -0.9999, 0.9999)
        
        # Вычисляем котангенс угла
        angle = np.arccos(cos_angle)
        alpha = 1.0 / np.tan(angle)
        alpha = np.maximum(alpha, 1e-2 * np.ones(len(alpha)))
        
        # Добавляем веса в матрицу
        W = W + sparse.coo_matrix((alpha, (F[i2, :], F[i3, :])), shape=(n, n))
        W = W + sparse.coo_matrix((alpha, (F[i3, :], F[i2, :])), shape=(n, n))
    
    # Строим матрицу Лапласиана L = D - W
    d = np.array(W.sum(axis=0)).flatten()
    D = sparse.diags(d, 0)
    L = D - W
    
    print("  Матрица Лапласиана построена")
    
    # Подготовка модифицированного Лапласиана для граничных условий
    L1 = L.toarray().copy()
    
    # Обнуляем строки граничных точек и ставим 1 на диагонали
    L1[B, :] = 0
    for i in range(len(B)):
        L1[B[i], B[i]] = 1
    
    # Защита от сингулярности
    for i in range(L1.shape[0]):
        if L1[i, i] == 0:
            L1[i, i] = 1e-6
    
    print("  Граничные условия установлены")
    
    # Создаем точки на единичном квадрате для граничных условий
    sample_boundary_num = sampling_size * 4
    
    # Равномерное распределение точек от 0 до 1
    tq_1 = (np.arange(1, sampling_size + 1) - 1) / (sampling_size - 1)
    
    # Создаем точки на сторонах единичного квадрата
    Z_sample = np.vstack((
        np.hstack((tq_1, np.ones(sampling_size), 1 - tq_1, np.zeros(sampling_size))),  # X-координаты
        np.hstack((np.zeros(sampling_size), tq_1, np.ones(sampling_size), 1 - tq_1))   # Y-координаты
    ))
    
    # Сопоставляем граничные точки физической области с единичным квадратом
    Z_B_sample_list = []
    for i in range(len(B)):
        idx = int(i / len(B) * sampling_size * 4)
        idx = min(idx, Z_sample.shape[1] - 1)  # Защита от выхода за границы
        Z_B_sample_list.append(Z_sample[:, idx])
    
    Z = np.array(Z_B_sample_list).T
    
    print("  Граничные условия на единичном квадрате созданы")
    
    # Создаем правую часть уравнения
    R = np.zeros([2, n])
    R[:, B] = Z  # для граничных точек задаем позиции на единичном квадрате
    
    # Решаем уравнение Лапласа для x и y координат отдельно
    Y = np.zeros([2, n])
    
    print("  Решаем уравнение Лапласа...")
    
    try:
        Y[0, :] = linalg.spsolve(sparse.csr_matrix(L1), R[0, :])
        Y[1, :] = linalg.spsolve(sparse.csr_matrix(L1), R[1, :])
        print("   Уравнение решено успешно")
    except Exception as e:
        print(f"   Ошибка решения: {e}")
        return False
    
    # Проверяем качество решения
    E = np.linalg.norm(np.dot(L1, Y.T)) / 2
    print(f"  Норма невязки: {E:.6f}")
    
    # Интерполяция на регулярную сетку с использованием функции из статьи
    q = sampling_size
    M = np.zeros([q, q, 3])
    
    print("  Начинаем интерполяцию...")
    
    # Интерполируем физические координаты на регулярную сетку унитарного пространства
    for i in range(3):  # для x, y, z координат
        M[:, :, i] = compute_triang_interp(F, Y, X[i, :], q)
    
    print("   Интерполяция завершена")
    
    # Сохраняем координаты
    x_data = M[:, :, 0].flatten().tolist()
    y_data = M[:, :, 1].flatten().tolist()
    
    # Создаем изображение для визуализации
    M_image = M.copy()
    for i in range(3):
        if M_image[:, :, i].max() != M_image[:, :, i].min():
            M_image[:, :, i] = (M_image[:, :, i] - M_image[:, :, i].min()) / \
                              (M_image[:, :, i].max() - M_image[:, :, i].min())
    
    # Сохраняем scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(x_data, y_data, s=1, alpha=0.6)
    plt.title(f"Fourугольник {objfilename}")
    plt.axis('equal')
    plt.savefig(f"./part_img/{objfilename}_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Сохраняем изображение
    plt.imsave(f"./part_img/{objfilename}_img.png", M_image)
    
    # Сохраняем данные в CSV
    with open('./data/x_data.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(x_data)
    
    with open('./data/y_data.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(y_data)
    
    with open('./data/E.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([str(E)])
    
    print(f"  ✓ Данные сохранены для fourугольника {objfilename}")
    print(f"  Размер регулярной сетки: {q}x{q} = {len(x_data)} точек")
    
    return True

def save_data(filename, data):
    """Сохранение данных в CSV"""
    with open(filename, 'w') as f:
        np.savetxt(f, data, delimiter=',')

if __name__ == "__main__":

    # Создаем необходимые папки
    os.makedirs('data', exist_ok=True)
    os.makedirs('part_img', exist_ok=True)
    
    # Очищаем выходные файлы
    for file in ['./data/x_data.csv', './data/y_data.csv', './data/E.csv']:
        if os.path.exists(file):
            os.remove(file)
    
    # Читаем geo_data_quad.csv
    csvfile = open('./geo_data_quad.csv')
    new_geo = []
    
    for num, row in enumerate(csvfile):
        if num >= 0:
            line = row.replace("\n", "")
            line = line.split(',')
            objfilename = int(float(line[0]))
            
            # Извлекаем координаты углов из geo_data_quad.csv
            # Формат: индекс, x_left_up, y_left_up, x_right_up, y_right_up, x_left_down, x_right_down, scaled
            geo_data = {
                'x_left_up': float(line[1]),
                'y_left_up': float(line[2]), 
                'x_right_up': float(line[3]),
                'y_right_up': float(line[4]),
                'x_left_down': float(line[5]),
                'x_right_down': float(line[6]),
                'scaled': float(line[7])
            }

            print(f"\n--- Обработка 4угольника {objfilename} ---")
            objname = f'./part_obj/{objfilename}.obj'
            
            if os.path.exists(objname):
                # Передаем geo_data в функцию
                success = create_image_quad(objname, 128, str(objfilename), 0, geo_data)
                if success:
                    new_line = [float(i) for i in line]
                    new_geo.append(new_line)
                    save_data('geo_new.csv', np.array(new_geo))
            else:
                print(f"  ✗ Файл {objname} не найден")
        
    print("\n Все fourугольники обработаны!")
