import numpy as np
import csv
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Загрузите данные
x_coords_quad = []
y_coords_quad = []

with open('./data/x_data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in enumerate(reader):
        x_coords_quad.append([float(x) for x in row[1]])

with open('./data/y_data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in enumerate(reader):
        y_coords_quad.append([float(y) for y in row[1]])

x_coords_quad = np.array(x_coords_quad)
y_coords_quad = np.array(y_coords_quad)

x_coords_pent = []
y_coords_pent = []

with open('../../Diffeomorphism-Neural-Operator/data_geo5_r128/train_x_data.csv', 'r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i < 10:  # первые 10
            x_coords_pent.append([float(x) for x in row])

with open('../../Diffeomorphism-Neural-Operator/data_geo5_r128/train_y_data.csv', 'r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i < 10:
            y_coords_pent.append([float(y) for y in row])

x_coords_pent = np.array(x_coords_pent)
y_coords_pent = np.array(y_coords_pent)



# 1. Проверяем нормализацию


q_x_min, q_x_max = x_coords_quad[0].min(), x_coords_quad[0].max()
q_y_min, q_y_max = y_coords_quad[0].min(), y_coords_quad[0].max()
p_x_min, p_x_max = x_coords_pent[0].min(), x_coords_pent[0].max()
p_y_min, p_y_max = y_coords_pent[0].min(), y_coords_pent[0].max()

print(f"Четырехугольник [0]:")
print(f"  X: [{q_x_min:.4f}, {q_x_max:.4f}] (диапазон: {q_x_max - q_x_min:.4f})")
print(f"  Y: [{q_y_min:.4f}, {q_y_max:.4f}] (диапазон: {q_y_max - q_y_min:.4f})")

print(f"\nПятиугольник [0]:")
print(f"  X: [{p_x_min:.4f}, {p_x_max:.4f}] (диапазон: {p_x_max - p_x_min:.4f})")
print(f"  Y: [{p_y_min:.4f}, {p_y_max:.4f}] (диапазон: {p_y_max - p_y_min:.4f})")


print(f"\nЧетырехугольник [0]:")
print(f"  X mean={x_coords_quad[0].mean():.4f}, std={x_coords_quad[0].std():.4f}")
print(f"  Y mean={y_coords_quad[0].mean():.4f}, std={y_coords_quad[0].std():.4f}")
print(f"  X values [0:5]: {x_coords_quad[0, :5]}")
print(f"  Y values [0:5]: {y_coords_quad[0, :5]}")

print(f"\nПятиугольник [0]:")
print(f"  X mean={x_coords_pent[0].mean():.4f}, std={x_coords_pent[0].std():.4f}")
print(f"  Y mean={y_coords_pent[0].mean():.4f}, std={y_coords_pent[0].std():.4f}")
print(f"  X values [0:5]: {x_coords_pent[0, :5]}")
print(f"  Y values [0:5]: {y_coords_pent[0, :5]}")

# 3. Проверяем корреляцию ВНУТРИ каждого датасета


corr_quad = pearsonr(x_coords_quad[0], y_coords_quad[0])[0]
corr_pent = pearsonr(x_coords_pent[0], y_coords_pent[0])[0]

print(f"Четырехугольник: corr(X, Y) = {corr_quad:.4f}")
print(f"Пятиугольник: corr(X, Y) = {corr_pent:.4f}")

# 4. Проверяем DDS ПОСЛЕ нормализации


def normalize_coords(x, y):
    x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x
    y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() != y.min() else y
    return x_norm, y_norm

x_quad_norm, y_quad_norm = normalize_coords(x_coords_quad[0], y_coords_quad[0])
x_pent_norm, y_pent_norm = normalize_coords(x_coords_pent[0], y_coords_pent[0])

coords_quad_norm = np.concatenate([x_quad_norm, y_quad_norm])
coords_pent_norm = np.concatenate([x_pent_norm, y_pent_norm])

dds_norm = pearsonr(coords_quad_norm, coords_pent_norm)[0]
print(f"DDS ПОСЛЕ нормализации каждого образца: {dds_norm:.4f}")

# 5. Визуализация

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Четырехугольник
axes[0, 0].scatter(x_coords_quad[0], y_coords_quad[0], s=1, alpha=0.5)
axes[0, 0].set_title(f"Четырехугольник (физ. домен)\nX:[{q_x_min:.2f},{q_x_max:.2f}] Y:[{q_y_min:.2f},{q_y_max:.2f}]")
axes[0, 0].set_aspect('equal')

axes[0, 1].hist(x_coords_quad[0], bins=50, alpha=0.7, label='X')
axes[0, 1].hist(y_coords_quad[0], bins=50, alpha=0.7, label='Y')
axes[0, 1].set_title("Четырехугольник: распределение X и Y")
axes[0, 1].legend()

axes[0, 2].scatter(x_quad_norm, y_quad_norm, s=1, alpha=0.5)
axes[0, 2].set_title("Четырехугольник (нормализован)")
axes[0, 2].set_xlim(-0.1, 1.1)
axes[0, 2].set_ylim(-0.1, 1.1)

# Пятиугольник
axes[1, 0].scatter(x_coords_pent[0], y_coords_pent[0], s=1, alpha=0.5, color='orange')
axes[1, 0].set_title(f"Пятиугольник (физ. домен)\nX:[{p_x_min:.2f},{p_x_max:.2f}] Y:[{p_y_min:.2f},{p_y_max:.2f}]")
axes[1, 0].set_aspect('equal')

axes[1, 1].hist(x_coords_pent[0], bins=50, alpha=0.7, label='X', color='orange')
axes[1, 1].hist(y_coords_pent[0], bins=50, alpha=0.7, label='Y', color='green')
axes[1, 1].set_title("Пятиугольник: распределение X и Y")
axes[1, 1].legend()

axes[1, 2].scatter(x_pent_norm, y_pent_norm, s=1, alpha=0.5, color='orange')
axes[1, 2].set_title("Пятиугольник (нормализован)")
axes[1, 2].set_xlim(-0.1, 1.1)
axes[1, 2].set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.savefig('deep_diagnose.png', dpi=150)
plt.close()


