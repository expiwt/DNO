# -*- coding: utf-8 -*-
import numpy as np
import csv
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


def load_coordinates_from_csv(filepath_x, filepath_y, num_samples=None):
    """
    Загружает координаты из CSV файлов
    
    Returns:
    --------
    x_coords, y_coords : ndarray
        Массивы координат [num_samples, 128*128]
    """
    x_coords = []
    y_coords = []
    
    with open(filepath_x, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if num_samples is not None and i >= num_samples:
                break
            x_coords.append([float(x) for x in row])
    
    with open(filepath_y, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if num_samples is not None and i >= num_samples:
                break
            y_coords.append([float(y) for y in row])
    
    return np.array(x_coords), np.array(y_coords)


def compute_DDS(domain_coords_1, domain_coords_2):
    """
    Вычисляет DDS между двумя доменами
    
    Parameters:
    -----------
    domain_coords_1, domain_coords_2 : ndarray, shape [2, N]
        X и Y координаты точек сетки
    
    Returns:
    --------
    dds : float
    """
    coords_1_flat = domain_coords_1.flatten()
    coords_2_flat = domain_coords_2.flatten()
    dds, _ = pearsonr(coords_1_flat, coords_2_flat)
    return dds


def compute_cross_DDS(x_coords_A, y_coords_A, x_coords_B, y_coords_B):
    """
    Вычисляет DDS между всеми парами образцов из двух датасетов A и B
    
    Returns:
    --------
    cross_dds_matrix : ndarray, shape [num_A, num_B]
        Матрица DDS между датасетами
    dds_scores_A : ndarray, shape [num_A]
        Средний/максимальный DDS для каждого образца из A относительно B
    """
    num_A = x_coords_A.shape[0]
    num_B = x_coords_B.shape[0]
    
    cross_dds_matrix = np.zeros((num_A, num_B))
    
    print(f"Вычисляем DDS между {num_A} образцами A и {num_B} образцами B...")
    
    for i in range(num_A):
        coords_A = np.stack([x_coords_A[i], y_coords_A[i]], axis=0)
        
        for j in range(num_B):
            coords_B = np.stack([x_coords_B[j], y_coords_B[j]], axis=0)
            dds = compute_DDS(coords_A, coords_B)
            cross_dds_matrix[i, j] = dds
        
        if (i + 1) % 5 == 0 or i == num_A - 1:
            print(f"  Обработано {i + 1}/{num_A} образцов A")
    
    # Для каждого образца A: средний и максимальный DDS относительно всех B
    dds_scores_mean = np.mean(cross_dds_matrix, axis=1)
    dds_scores_max = np.max(cross_dds_matrix, axis=1)
    
    return cross_dds_matrix, dds_scores_mean, dds_scores_max


def visualize_cross_DDS(cross_dds_matrix, label_A='Четырехугольники', 
                         label_B='Пятиугольники', save_path='cross_dds_matrix.png'):
    """Визуализирует матрицу DDS между двумя датасетами"""
    plt.figure(figsize=(12, 8))
    
    # Используем seaborn для красивой тепловой карты
    sns.heatmap(cross_dds_matrix, cmap='RdYlGn', vmin=0.85, vmax=1.0, 
                cbar_kws={'label': 'DDS'}, annot=False)
    
    plt.title(f'DDS между {label_A} и {label_B}', fontsize=14)
    plt.xlabel(f'Индекс образца ({label_B})', fontsize=12)
    plt.ylabel(f'Индекс образца ({label_A})', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Матрица cross-DDS сохранена: {save_path}")


def plot_dds_distribution(dds_scores, label='Четырехугольники vs Пятиугольники',
                          save_path='dds_distribution.png'):
    """Визуализирует распределение DDS scores"""
    plt.figure(figsize=(10, 6))
    
    plt.hist(dds_scores, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Добавляем пороговую линию из статьи
    plt.axvline(0.97, color='red', linestyle='--', linewidth=2, label='Порог DNO = 0.97')
    
    # Статистика
    mean_dds = np.mean(dds_scores)
    median_dds = np.median(dds_scores)
    plt.axvline(mean_dds, color='green', linestyle='-', linewidth=2, label=f'Средний = {mean_dds:.4f}')
    
    plt.title(f'Распределение DDS: {label}', fontsize=14)
    plt.xlabel('DDS', fontsize=12)
    plt.ylabel('Количество образцов', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Распределение DDS сохранено: {save_path}")
def normalize_coordinates(x_coords, y_coords):
    """
    Нормализует координаты в диапазон [0, 1] для каждого образца отдельно
    """
    x_coords_norm = np.zeros_like(x_coords)
    y_coords_norm = np.zeros_like(y_coords)
    
    for i in range(x_coords.shape[0]):
        # Нормализация X координат для образца i
        x_min, x_max = x_coords[i].min(), x_coords[i].max()
        if x_max - x_min > 0:
            x_coords_norm[i] = (x_coords[i] - x_min) / (x_max - x_min)
        else:
            x_coords_norm[i] = x_coords[i]
        
        # Нормализация Y координат для образца i
        y_min, y_max = y_coords[i].min(), y_coords[i].max()
        if y_max - y_min > 0:
            y_coords_norm[i] = (y_coords[i] - y_min) / (y_max - y_min)
        else:
            y_coords_norm[i] = y_coords[i]
    
    return x_coords_norm, y_coords_norm

if __name__ == "__main__":
    print("ВЫЧИСЛЕНИЕ CROSS-DDS (Четырехугольники vs Пятиугольники)")
    
    # Загружаем данные четырехугольников
    print("\n Загрузка данных четырехугольников...")
    x_coords_quad, y_coords_quad = load_coordinates_from_csv(
        './data/x_data.csv', 
        './data/y_data.csv'
    )
    print(f"  Загружено четырехугольников: {x_coords_quad.shape[0]}")
    
    # Загружаем данные пятиугольников
    print("\n Загрузка данных пятиугольников...")
    # ВАЖНО: Укажите правильные пути к файлам пятиугольников!
    x_coords_pent, y_coords_pent = load_coordinates_from_csv(
        '../../Diffeomorphism-Neural-Operator/data_geo5_r128/train_x_data.csv',  # ← ИЗМЕНИТЕ на ваш путь!
        '../../Diffeomorphism-Neural-Operator/data_geo5_r128/train_y_data.csv'   # ← ИЗМЕНИТЕ на ваш путь!
    )
    print(f"  Загружено пятиугольников: {x_coords_pent.shape[0]}")
    print("\n Нормализация координат...")
    x_coords_quad_norm, y_coords_quad_norm = normalize_coordinates(x_coords_quad, y_coords_quad)
    x_coords_pent_norm, y_coords_pent_norm = normalize_coordinates(x_coords_pent, y_coords_pent)

    # Вычисляем cross-DDS
    cross_dds_matrix, dds_scores_mean, dds_scores_max = compute_cross_DDS(
        x_coords_quad_norm, y_coords_quad_norm,
        x_coords_pent_norm, y_coords_pent_norm
    )
    
    # Статистика
    print(" РЕЗУЛЬТАТЫ: DDS между четырехугольниками и пятиугольниками")
    
    print("\n🔹 Средний DDS для каждого четырехугольника относительно всех пятиугольников:")
    print(f"  Средний: {np.mean(dds_scores_mean):.4f}")
    print(f"  Медианный: {np.median(dds_scores_mean):.4f}")
    print(f"  Мин: {np.min(dds_scores_mean):.4f}")
    print(f"  Макс: {np.max(dds_scores_mean):.4f}")
    print(f"  Std: {np.std(dds_scores_mean):.4f}")
    
    print("\n🔹 Максимальный DDS для каждого четырехугольника (лучшее совпадение с пятиугольником):")
    print(f"  Средний: {np.mean(dds_scores_max):.4f}")
    print(f"  Медианный: {np.median(dds_scores_max):.4f}")
    print(f"  Мин: {np.min(dds_scores_max):.4f}")
    print(f"  Макс: {np.max(dds_scores_max):.4f}")
    
    # Анализ относительно порога 0.97
    print("\n Анализ относительно порога DNO = 0.97:")
    good_samples_mean = np.sum(dds_scores_mean > 0.97)
    bad_samples_mean = np.sum(dds_scores_mean < 0.97)
    
    print(f"\n  По среднему DDS:")
    print(f" Четырехугольников с DDS > 0.97: {good_samples_mean}/{len(dds_scores_mean)} ({100*good_samples_mean/len(dds_scores_mean):.1f}%)")
    print(f"  Четырехугольников с DDS < 0.97: {bad_samples_mean}/{len(dds_scores_mean)} ({100*bad_samples_mean/len(dds_scores_mean):.1f}%)")
    
    good_samples_max = np.sum(dds_scores_max > 0.97)
    bad_samples_max = np.sum(dds_scores_max < 0.97)
    
    print(f"\n  По максимальному DDS (лучшее совпадение):")
    print(f" Четырехугольников с DDS > 0.97: {good_samples_max}/{len(dds_scores_max)} ({100*good_samples_max/len(dds_scores_max):.1f}%)")
    print(f"  Четырехугольников с DDS < 0.97: {bad_samples_max}/{len(dds_scores_max)} ({100*bad_samples_max/len(dds_scores_max):.1f}%)")
    
    # Выводы

    if np.mean(dds_scores_mean) < 0.95:
        print("\n  КРИТИЧЕСКАЯ ПРОБЛЕМА: Средний DDS < 0.95")
        print("   Четырехугольники геометрически СИЛЬНО отличаются от пятиугольников!")
        print("\n   Рекомендации:")
        print("   1. ПЕРЕОБУЧИТЬ модель на смешанном датасете (пятиугольники + четырехугольники)")
        print("   2. Использовать transfer learning: fine-tune на четырехугольниках")
        print("   3. Рассмотреть другой генерический домен (круг вместо квадрата)")
    
    elif np.mean(dds_scores_mean) < 0.97:
        print("\n  УМЕРЕННАЯ ПРОБЛЕМА: Средний DDS = 0.95-0.97")
        print("   Четырехугольники на границе порога обобщения DNO.")
        print("\n   Рекомендации:")
        print("   1. Fine-tuning модели на небольшом количестве четырехугольников")
        print("   2. Увеличить разрешение генерического домена (256×256)")
        print("   3. Улучшить качество триангуляции mesh")
    
    else:
        print("\n DDS > 0.97: Геометрия не является основной проблемой!")
        print("   Проверьте другие факторы:")
        print("   1. Распределение функции параметров a(x,y)")
        print("   2. Граничные условия")
        print("   3. Качество интерполяции и решения уравнения Лапласа")
    
    # Визуализация
    print("\n Генерация визуализаций...")
    visualize_cross_DDS(cross_dds_matrix, save_path='cross_dds_quad_vs_pent.png')
    plot_dds_distribution(dds_scores_mean, save_path='dds_distribution_mean.png')
    plot_dds_distribution(dds_scores_max, label='Четырехугольники vs Пятиугольники (макс)', 
                          save_path='dds_distribution_max.png')
    
    # Сохранение результатов
    np.savetxt('./cross_dds_scores_mean.csv', dds_scores_mean, delimiter=',')
    np.savetxt('./cross_dds_scores_max.csv', dds_scores_max, delimiter=',')
    np.savetxt('./cross_dds_matrix.csv', cross_dds_matrix, delimiter=',')
    
    print("\n Результаты сохранены в:")
    print("   - cross_dds_scores_mean.csv")
    print("   - cross_dds_scores_max.csv")
    print("   - cross_dds_matrix.csv")
    
