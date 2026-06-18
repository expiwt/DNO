"""
Ручная проверка одного ребра: cotangent vs QH(I) weight.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from read_msh import read_msh_file
from scipy.sparse import coo_matrix

test_dir = os.path.join(os.path.dirname(__file__), "../test_domains")
files = sorted([f for f in os.listdir(test_dir) if f.endswith(".msh")])
X, F, bnds = read_msh_file(os.path.join(test_dir, files[0]))

from quasi_harmonic import QuasiHarmonicMapper
mapper = QuasiHarmonicMapper(X, F, bnds, n_iter=0)
Y = mapper.build_mapping(verbose=False)

# Выбираем ребро (746, 1484) из debug вывода
# Находим треугольники, содержащие это ребро
edge = (746, 1484)
tri_found = []
for t in range(len(F)):
    if edge[0] in F[t] and edge[1] in F[t]:
        tri_found.append(t)

print(f"Ребро {edge}: {len(tri_found)} треугольников")
for t in tri_found:
    a, b, c = F[t]
    print(f"\n--- Треугольник #{t}: ({a},{b},{c}) ---")
    
    # Все вершины
    phys = X[[a, b, c]]
    log = Y[[a, b, c]]
    
    print(f"Физические: {phys}")
    print(f"Логические: {log}")
    
    # Площади
    A_phys_s = 0.5 * (log[1,0]*(log[2,1]-log[0,1]) + log[2,0]*(log[0,1]-log[1,1]) + log[0,0]*(log[1,1]-log[2,1]))
    # Но нам нужна площадь в логическом пространстве
    A_log_s = 0.5 * (log[1,0]*(log[2,1]-log[0,1]) + log[2,0]*(log[0,1]-log[1,1]) + log[0,0]*(log[1,1]-log[2,1]))
    A_log_abs = abs(A_log_s)
    
    print(f"A_log_signed = {A_log_s:.10f}")
    print(f"A_log_abs = {A_log_abs:.10f}")
    
    # Какая вершина в треугольнике — edge[0] и edge[1]?
    # Определяем индексы внутри треугольника
    vidx = {a:0, b:1, c:2}
    i_idx = vidx[edge[0]]
    j_idx = vidx[edge[1]]
    k_idx = 3 - i_idx - j_idx  # третья вершина
    k = [a, b, c][k_idx]
    
    print(f"Ребро между вершинами {edge[0]}(idx={i_idx}) и {edge[1]}(idx={j_idx})")
    print(f"Третья вершина: {k}(idx={k_idx})")
    
    # Perpendiculars в логическом пространстве
    perps = [
        np.array([log[1,1]-log[2,1], log[2,0]-log[1,0]]),  # e_a: ребро (b→c)⊥
        np.array([log[2,1]-log[0,1], log[0,0]-log[2,0]]),  # e_b: ребро (c→a)⊥
        np.array([log[0,1]-log[1,1], log[1,0]-log[0,0]])   # e_c: ребро (a→b)⊥
    ]
    
    ei = perps[i_idx]
    ej = perps[j_idx]
    
    print(f"perp[{i_idx}] = {ei}")
    print(f"perp[{j_idx}] = {ej}")
    
    # my QH contribution (без минуса)
    K_contrib = ei @ ej / (4.0 * A_log_abs)
    # w = -K
    w_qh = -K_contrib
    
    print(f"perps[i]·perps[j] = {ei @ ej:.10f}")
    print(f"K_ij = {K_contrib:.10f}")
    print(f"w_qh = {w_qh:.10f}")
    
    # Cotangent вес для этого ребра
    # w_cot = 0.5 * cot(angle at vertex k)
    pk_phys = X[k]
    pi_phys = X[edge[0]]
    pj_phys = X[edge[1]]
    
    u = pi_phys - pk_phys  # from k to i
    v = pj_phys - pk_phys  # from k to j
    
    cos_gamma = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-15)
    cross = abs(u[0]*v[1] - u[1]*v[0])
    cot = cos_gamma / (cross / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-15) + 1e-15)
    w_cot = 0.5 * cot
    
    print(f"                    !")  
    print(f"Физические углы: k={pk_phys}, i={pi_phys}, j={pj_phys}")
    print(f"u = i-k = {u}, v = j-k = {v}")
    print(f"cos γ = {cos_gamma:.10f}")
    print(f"cot γ = {cot:.10f}")
    print(f"w_cot = 0.5·cot = {w_cot:.10f}")
    print(f"\n  >>> w_qh = {w_qh:.10f}, w_cot = {w_cot:.10f}, diff = {abs(w_qh - w_cot):.10f}")
    
    # Проверка: если использовать SIGNED площадь вместо ABS
    w_qh_signed = -ei @ ej / (4.0 * A_log_s)
    print(f"  >>> w_qh_signed = {w_qh_signed:.10f}")
