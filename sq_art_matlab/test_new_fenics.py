# test_new_fenics_fixed.py
import dolfinx as fe
from dolfinx import mesh, fem, io
import numpy as np
from mpi4py import MPI

print("=== FEniCSx Structure ===")
print("Версия:", fe.__version__)

# Тест создания mesh через правильный способ
print("\n=== Тест создания mesh ===")

# Способ 1: Прямоугольник (рабочий)
domain = mesh.create_rectangle(
    MPI.COMM_WORLD, 
    [np.array([0, 0]), np.array([1, 1])], 
    [32, 32]
)
print("✓ Прямоугольный mesh создан")

# Способ 2: Проверяем gmsh для сложных геометрий
try:
    import gmsh
    print(" gmsh доступен - можно создавать сложные геометрии")
except ImportError as e:
    print(" gmsh проблема:", e)

print("Mesh информация:")
print("  Вершин:", domain.geometry.x.shape[0])
print("  Ячеек:", domain.topology.index_map(2).size_local)

# Способ 3: Простой прямоугольник для теста
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])], [32, 32])
print("Прямоугольный mesh создан:", domain.geometry.x.shape)
