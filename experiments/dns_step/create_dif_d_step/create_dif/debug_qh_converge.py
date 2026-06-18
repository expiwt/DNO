"""
Debug: проверяем сходимость QH итерации за итерацией.
Сравниваем результаты после каждой итерации с гармоническим решением.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from read_msh import read_msh_file
from quasi_harmonic import QuasiHarmonicMapper, compute_distortions, summarize_distortions
from scipy.sparse import diags

test_dir = os.path.join(os.path.dirname(__file__), "../test_domains")
files = sorted([f for f in os.listdir(test_dir) if f.endswith(".msh")])
X, F, bnds = read_msh_file(os.path.join(test_dir, files[0]))

# Гармоническое отображение
mapper = QuasiHarmonicMapper(X, F, bnds, n_iter=0)
Y0 = mapper.build_mapping(verbose=False)
angle0, area0 = compute_distortions(X, Y0, F)
s0 = summarize_distortions(angle0, area0)
print(f"Harmonic:  angle_mean={s0['angle_mean']:.4f}, area_max={s0['area_max']:.4f}, area_median={s0['area_median']:.4f}")

# QH с mode='sqrt', без гомотопии, без relax
Y = Y0.copy()
for it in range(5):
    # Чисто QH веса (без смешивания с cot)
    mapper_qh = QuasiHarmonicMapper(X, F, bnds, n_iter=0)
    W_qh = mapper_qh._build_weights(Y)
    
    # Laplacian
    degrees = np.array(W_qh.sum(axis=1)).flatten()
    L = diags(degrees) - W_qh
    
    # Тот же солвер
    xi = mapper_qh._solve_dirichlet(
        L, bnds['inlet'] + bnds['outlet'],
        [0.0]*len(bnds['inlet']) + [1.0]*len(bnds['outlet'])
    )
    eta = mapper_qh._solve_dirichlet(
        L, bnds['bottom'] + bnds['top'],
        [0.0]*len(bnds['bottom']) + [1.0]*len(bnds['top'])
    )
    Y_new = np.column_stack((xi, eta))
    
    diff = np.max(np.abs(Y_new - Y))
    Y = Y_new.copy()
    
    angle_q, area_q = compute_distortions(X, Y, F)
    s = summarize_distortions(angle_q, area_q)
    n_neg = (L.diagonal() < 0).sum()
    print(f"Iter {it+1}: Δ={diff:.6e}, angle_mean={s['angle_mean']:.4f}, "
          f"area_max={s['area_max']:.4f}, area_median={s['area_median']:.4f}, "
          f"neg_diag={n_neg}")
