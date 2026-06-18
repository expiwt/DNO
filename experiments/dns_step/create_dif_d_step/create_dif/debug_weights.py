"""
Debug: сравниваем quasi-harmonic веса (C=I) с котангенсными.
Должны совпадать.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from read_msh import read_msh_file
from scipy.sparse import coo_matrix, diags

# Загрузка
test_dir = os.path.join(os.path.dirname(__file__), "../test_domains")
files = sorted([f for f in os.listdir(test_dir) if f.endswith(".msh")])
X, F, bnds = read_msh_file(os.path.join(test_dir, files[0]))
n_nodes, n_tri = X.shape[0], F.shape[0]
print(f"Узлов: {n_nodes}, треугольников: {n_tri}")

# Строим гармоническое отображение (эталон)
from quasi_harmonic import QuasiHarmonicMapper
mapper = QuasiHarmonicMapper(X, F, bnds, n_iter=0)
Y = mapper.build_mapping(verbose=False)

# 1. Котангенсные веса (эталон)
def cotangent_weights(X, F, n_nodes):
    v0 = X[F[:, 0]]; v1 = X[F[:, 1]]; v2 = X[F[:, 2]]
    def get_cotan(u, v):
        dot = np.sum(u * v, axis=1)
        cross = np.abs(u[:, 0]*v[:, 1] - u[:, 1]*v[:, 0])
        cross = np.maximum(cross, 1e-12)
        return dot / cross
    cot0 = get_cotan(v1 - v0, v2 - v0)
    cot1 = get_cotan(v2 - v1, v0 - v1)
    cot2 = get_cotan(v0 - v2, v1 - v2)
    rows, cols, data = [], [], []
    for k, cot in enumerate([cot0, cot1, cot2]):
        i = F[:, (k + 1) % 3]; j = F[:, (k + 2) % 3]
        rows.extend(i); cols.extend(j); data.extend(cot * 0.5)
        rows.extend(j); cols.extend(i); data.extend(cot * 0.5)
    return coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()

W_cot = cotangent_weights(X, F, n_nodes)

# 2. QH веса с C=I
# Блокируем _compute_tensor_C, чтобы всегда возвращать I
def build_qh_weights_c_eq_I(Y, X, F, n_nodes, n_tri):
    """QH weights with C=I — должны совпадать с cotangent."""
    rows, cols, data = [], [], []
    for t in range(n_tri):
        a, b, c = F[t]
        log = Y[[a, b, c]]
        phys = X[[a, b, c]]
        
        A_log = 0.5 * abs(
            log[1, 0]*(log[2,1]-log[0,1]) + log[2,0]*(log[0,1]-log[1,1]) + log[0,0]*(log[1,1]-log[2,1])
        )
        if A_log < 1e-15:
            continue
        
        C_I = np.eye(2)
        
        e_a = np.array([log[1,1]-log[2,1], log[2,0]-log[1,0]])
        e_b = np.array([log[2,1]-log[0,1], log[0,0]-log[2,0]])
        e_c = np.array([log[0,1]-log[1,1], log[1,0]-log[0,0]])
        perps = [e_a, e_b, e_c]
        
        for idx1, idx2 in [(0,1), (1,2), (2,0)]:
            vi, vj = F[t][idx1], F[t][idx2]
            # K_ij = element stiffness matrix entry (negative for off-diag)
            # w_ij = -K_ij = weight (positive, matching cotangent)
            K_ij = (perps[idx1] @ (C_I @ perps[idx2])) / (4.0 * A_log)
            w_ij = -K_ij
            if abs(w_ij) < 1e-15: continue
            rows.extend([vi, vj]); cols.extend([vj, vi]); data.extend([w_ij, w_ij])
    
    return coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()

W_qh_I = build_qh_weights_c_eq_I(Y, X, F, n_nodes, n_tri)

# 3. Сравнение
# Веса для нескольких случайных рёбер
np.random.seed(42)
test_edges = []
edges_set = set()
for t in range(min(100, n_tri)):
    for idx1, idx2 in [(0,1), (1,2), (2,0)]:
        e = tuple(sorted([F[t][idx1], F[t][idx2]]))
        if e not in edges_set:
            edges_set.add(e)
            test_edges.append(e)

print(f"\nСравнение весов на {len(test_edges)} рёбрах:")
print(f"{'ребро':>10} {'cot(w_ij)':>12} {'qh_I(w_ij)':>12} {'разница':>12}")
print("-" * 50)

max_diff = 0
for i, j in test_edges[:20]:
    w_c = W_cot[i, j]
    w_q = W_qh_I[i, j]
    diff = abs(w_c - w_q)
    max_diff = max(max_diff, diff)
    print(f"({i:>4},{j:>4}) {w_c:>12.6f} {w_q:>12.6f} {diff:>12.8f}")

print(f"\nMax diff на выборке: {max_diff:.10f}")
print(f"Всего ненулевых: cot={W_cot.nnz}, qh_I={W_qh_I.nnz}")

# Глобальное сравнение
W_cot_coo = W_cot.tocoo()
W_qh_coo = W_qh_I.tocoo()
# Собираем все пары (i,j) из обеих матриц
all_pairs = set()
for ij in zip(W_cot_coo.row, W_cot_coo.col):
    all_pairs.add(ij)
for ij in zip(W_qh_coo.row, W_qh_coo.col):
    all_pairs.add(ij)

gmax = 0
for i, j in all_pairs:
    gmax = max(gmax, abs(W_cot[i,j] - W_qh_I[i,j]))

print(f"Global max diff: {gmax:.10f}")
