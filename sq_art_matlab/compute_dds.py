# -*- coding: utf-8 -*-
import numpy as np
import csv
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def load_coordinates_from_csv(filepath, num_samples=None):
    """
    Загружает координаты из CSV файла
    
    Parameters:
    -----------
    filepath : str
        Путь к файлу (x_data.csv или y_data.csv)
    num_samples : int, optional
        Количество образцов для загрузки (если None - загружаем все)
    
    Returns:
    --------
    coords : ndarray, shape [num_samples, 128*128]
        Координаты для каждого образца
    """
    coords = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if num_samples is not None and i >= num_samples:
                break
            coords.append([float(x) for x in row])
    
    return np.array(coords)


def compute_DDS(domain_coords_1, domain_coords_2):
    """
    Вычисляет DDS (Domain Diffeomorphism Similarity) между двумя доменами
    
    Parameters:
    -----------
    domain_coords_1 : ndarray, shape [128*128, 2] или [2, 128*128]
        X и Y координаты точек сетки домена 1 в генерическом пространстве
    domain_coords_2 : ndarray, shape [128*128, 2] или [2, 128*128]
        X и Y координаты точек сетки домена 2 в генерическом пространстве
    
    Returns:
    --------
    dds : float
        Значение DDS от -1 до 1 (обычно 0.9-1.0 для похожих доменов)
    """
    # Уплощаем координаты в один вектор (все X и Y координаты вместе)
    coords_1_flat = domain_coords_1.flatten()
    coords_2_flat = domain_coords_2.flatten()
    
    # Вычисляем нормированную кросс-корреляцию (NCC) = коэффициент Пирсона
    dds, p_value = pearsonr(coords_1_flat, coords_2_flat)
    
    return dds


def compute_DDS_matrix(x_coords, y_coords):
    """
    Вычисляет матрицу DDS между всеми парами образцов
    
    Parameters:
    -----------
    x_coords : ndarray, shape [num_samples, 128*128]
        X координаты для всех образцов
    y_coords : ndarray, shape [num_samples, 128*128]
        Y координаты для всех образцов
    
    Returns:
    --------
    dds_matrix : ndarray, shape [num_samples, num_samples]
        Матрица DDS между всеми парами
    """
    num_samples = x_coords.shape[0]
    dds_matrix = np.zeros((num_samples, num_samples))
    
    print(f"Вычисляем матрицу DDS для {num_samples} образцов...")
    
    for i in range(num_samples):
        for j in range(i, num_samples):
            # Объединяем X и Y координаты
            coords_i = np.stack([x_coords[i], y_coords[i]], axis=0)
            coords_j = np.stack([x_coords[j], y_coords[j]], axis=0)
            
            dds = compute_DDS(coords_i, coords_j)
            dds_matrix[i, j] = dds
            dds_matrix[j, i] = dds  # Симметричная матрица
            
        if (i + 1) % 10 == 0:
            print(f"  Обработано {i + 1}/{num_samples} образцов")
    
    return dds_matrix


def compute_DDS_between_datasets(x_coords_train, y_coords_train, 
                                   x_coords_test, y_coords_test):
    """
    Вычисляет DDS между тренировочным (пятиугольники) и тестовым (четырехугольники) датасетами
    
    Returns:
    --------
    dds_scores : ndarray, shape [num_test_samples]
        Средний DDS для каждого тестового образца относительно тренировочных
    """
    num_train = x_coords_train.shape[0]
    num_test = x_coords_test.shape[0]
    
    dds_scores = np.zeros(num_test)
    
    print(f"Вычисляем DDS между {num_test} тестовыми и {num_train} тренировочными образцами...")
    
    for i in range(num_test):
        coords_test = np.stack([x_coords_test[i], y_coords_test[i]], axis=0)
        
        dds_values = []
        for j in range(num_train):
            coords_train = np.stack([x_coords_train[j], y_coords_train[j]], axis=0)
            dds = compute_DDS(coords_test, coords_train)
            dds_values.append(dds)
        
        # Средний DDS относительно всех тренировочных образцов
        dds_scores[i] = np.mean(dds_values)
        
        print(f"  Тестовый образец {i+1}: DDS = {dds_scores[i]:.4f} "
              f"(min={np.min(dds_values):.4f}, max={np.max(dds_values):.4f})")
    
    return dds_scores


def visualize_DDS_matrix(dds_matrix, save_path='dds_matrix.png'):
    """Визуализирует матрицу DDS"""
    plt.figure(figsize=(10, 8))
    plt.imshow(dds_matrix, cmap='RdYlGn', vmin=0.85, vmax=1.0)
    plt.colorbar(label='DDS')
    plt.title('Матрица Domain Diffeomorphism Similarity (DDS)')
    plt.xlabel('Индекс образца')
    plt.ylabel('Индекс образца')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Матрица DDS сохранена: {save_path}")


if __name__ == "__main__":

    # СЦЕНАРИЙ 1: DDS внутри одного датасета (четырехугольники)
    print("\n--- Сценарий 1: DDS между четырехугольниками ---")
    
    try:
        x_coords_quad = load_coordinates_from_csv('./data/x_data.csv')
        y_coords_quad = load_coordinates_from_csv('./data/y_data.csv')
        
        print(f"Загружено четырехугольников: {x_coords_quad.shape[0]}")
        print(f"Размер сетки: {x_coords_quad.shape[1]} точек (должно быть 128*128=16384)")
        
        # Вычисляем матрицу DDS
        dds_matrix_quad = compute_DDS_matrix(x_coords_quad, y_coords_quad)
        
        # Статистика (исключая диагональ, где DDS=1)
        mask = ~np.eye(dds_matrix_quad.shape[0], dtype=bool)
        dds_values = dds_matrix_quad[mask]
        
        print(f"\n Статистика DDS между четырехугольниками:")
        print(f"  Средний DDS: {np.mean(dds_values):.4f}")
        print(f"  Медианный DDS: {np.median(dds_values):.4f}")
        print(f"  Мин DDS: {np.min(dds_values):.4f}")
        print(f"  Макс DDS: {np.max(dds_values):.4f}")
        print(f"  Std DDS: {np.std(dds_values):.4f}")
        
        # Визуализация
        visualize_DDS_matrix(dds_matrix_quad, 'dds_matrix_quadrilaterals.png')
        
    except FileNotFoundError:
        print("  Файлы ./data/x_data.csv и ./data/y_data.csv не найдены")
        print("Сначала запустите create_map.py для генерации данных")
    

