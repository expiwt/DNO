import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import os
import sys
import csv

# Импортируем твои модули
try:
    from read_msh import read_msh_file
    from interpolate import GridInterpolator
except ImportError:
    print("Ошибка: Не найдены файлы read_msh.py или interpolate.py")
    sys.exit(1)

class CotangentMapper:
    def __init__(self, X, F, boundaries):
        self.X = X
        self.F = F
        self.boundaries = boundaries
        self.n_nodes = X.shape[0]
        self.L = None
        # Y - это координаты в квадрате (Xi, Eta), которые мы найдем
        self.Y = None 

    def build_cotangent_laplacian(self):
        """Строит матрицу Лапласа с котангенсными весами."""
        v0 = self.X[self.F[:, 0]]
        v1 = self.X[self.F[:, 1]]
        v2 = self.X[self.F[:, 2]]

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
            i = self.F[:, (k + 1) % 3]
            j = self.F[:, (k + 2) % 3]
            rows.extend(i); cols.extend(j); data.extend(cot * 0.5)
            rows.extend(j); cols.extend(i); data.extend(cot * 0.5)

        W = coo_matrix((data, (rows, cols)), shape=(self.n_nodes, self.n_nodes))
        degrees = np.array(W.sum(axis=1)).flatten()
        D = diags(degrees)
        self.L = (D - W).tocsr()
        return self.L

    def solve_dirichlet(self, fixed_nodes, fixed_values):
        if self.L is None: self.build_cotangent_laplacian()
        A = self.L.copy()
        b = np.zeros(self.n_nodes)
        penalty = 1e15
        
        idx = np.array(fixed_nodes, dtype=int)
        val = np.array(fixed_values, dtype=float)
        
        diag = A.diagonal()
        diag[idx] += penalty
        A.setdiag(diag)
        b[idx] = val * penalty
        return spsolve(A, b)

    def build_mapping(self):
        self.build_cotangent_laplacian()
        
        # 1. Xi (Горизонталь): Вход -> 0, Выход -> 1
        xi_nodes = self.boundaries['inlet'] + self.boundaries['outlet']
        xi_vals  = [0.0]*len(self.boundaries['inlet']) + [1.0]*len(self.boundaries['outlet'])
        self.xi = self.solve_dirichlet(xi_nodes, xi_vals)
        
        # 2. Eta (Вертикаль): Пол -> 0, Потолок -> 1
        eta_nodes = self.boundaries['bottom'] + self.boundaries['top']
        eta_vals  = [0.0]*len(self.boundaries['bottom']) + [1.0]*len(self.boundaries['top'])
        self.eta = self.solve_dirichlet(eta_nodes, eta_vals)
        
        # Собираем Y (координаты в квадрате) для интерполятора
        self.Y = np.column_stack((self.xi, self.eta))
        return self.Y

def visualize_mapping(msh_path, M_x, M_y):
    """
    Рисует две панели:
    1. Идеальная регулярная сетка в квадрате.
    2. Та же сетка, перенесенная в физическое пространство.
    """
    debug_dir = os.path.dirname(msh_path)
    base = os.path.basename(msh_path).replace('.msh', '')
    save_path = os.path.join(debug_dir, f"{base}_final_check.png")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Панель 1: Универсальный домен (Квадрат) ---
    ax = axes[0]
    ax.set_title("1. Universal Domain (Regular Grid)")
    
    # Создаем идеальную сетку для визуализации
    resolution = M_x.shape[0]
    grid_1d = np.linspace(0, 1, resolution)
    U, V = np.meshgrid(grid_1d, grid_1d)
    
    # Рисуем линии сетки (каждую 8-ю для чистоты)
    step = 1
    ax.plot(U[:, ::step], V[:, ::step], 'k-', lw=0.5, alpha=0.3) # Вертикальные
    ax.plot(U[::step, :].T, V[::step, :].T, 'k-', lw=0.5, alpha=0.3) # Горизонтальные
    
    # Подписи
    ax.text(0.5, -0.05, "Bottom (Eta=0)", ha='center', color='red')
    ax.text(0.5, 1.05, "Top (Eta=1)", ha='center', color='blue')
    ax.text(-0.05, 0.5, "Inlet", va='center', rotation=90)
    ax.text(1.05, 0.5, "Outlet", va='center', rotation=-90)
    
    ax.set_aspect('equal')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(False) # Сетка уже нарисована нами

    # --- Панель 2: Физический домен (Ступенька) ---
    ax = axes[1]
    ax.set_title("2. Physical Domain (Mapped Grid)")
    
    # Рисуем перенесенную сетку (те же индексы, что и слева!)
    ax.plot(M_x[:, ::step], M_y[:, ::step], 'k-', lw=0.8, alpha=0.6)
    ax.plot(M_x[::step, :].T, M_y[::step, :].T, 'k-', lw=0.8, alpha=0.6)
    
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def process_mesh(msh_path, resolution=128, debug=True):
    # 1. Читаем (read_msh.py)
    X, F, bnds = read_msh_file(msh_path)
    if X is None: return None, None
    if not bnds['bottom'] or not bnds['top']:
        print(f"Skip {msh_path}: bad boundaries")
        return None, None

    # 2. Маппинг (Находим Y = (Xi, Eta) для каждого узла)
    mapper = CotangentMapper(X, F, bnds)
    mapper.build_mapping()
    
    # 3. Интерполяция (interpolate.py)
    # Вот здесь происходит то, что ты сказал:
    # Берем регулярную сетку в квадрате -> Барицентрики -> Физические координаты
    interpolator = GridInterpolator(mapper)
    M_x, M_y = interpolator.interpolate(sampling_size=resolution)
    
    # Лечение NaN на краях
    if np.isnan(M_x).any():
        from scipy.interpolate import NearestNDInterpolator
        mask = ~np.isnan(M_x)
        coords = np.argwhere(mask)
        fill_x = NearestNDInterpolator(coords, M_x[mask])
        fill_y = NearestNDInterpolator(coords, M_y[mask])
        
        nan_coords = np.argwhere(np.isnan(M_x))
        for r, c in nan_coords:
            M_x[r, c] = fill_x(r, c)
            M_y[r, c] = fill_y(r, c)

    # 4. Визуализация
    if debug:
        visualize_mapping(msh_path, M_x, M_y)
        
    return M_x, M_y

def main():
    INPUT_DIR = "../test_double_steps"
    OUTPUT_X = "../test_data2/x_data.csv"
    OUTPUT_Y = "../test_data2/y_data.csv"
    
    if not os.path.exists(INPUT_DIR):
        print(f"Directory {INPUT_DIR} not found.")
        return

    open(OUTPUT_X, 'w').close()
    open(OUTPUT_Y, 'w').close()
    
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".msh") and "step" in f],
                   key=lambda x: int(x.split('_')[1].split('.')[0]) if '_' in x else x)
    
    print(f"Processing {len(files)} files...")
    
    for fname in files:
        path = os.path.join(INPUT_DIR, fname)
        print(f"Processing {fname}...", end=" ")
        try:
            gx, gy = process_mesh(path, debug=True)
            if gx is not None:
                with open(OUTPUT_X, 'a') as fx, open(OUTPUT_Y, 'a') as fy:
                    np.savetxt(fx, gx.reshape(1, -1), delimiter=',')
                    np.savetxt(fy, gy.reshape(1, -1), delimiter=',')
                print("OK")
            else:
                print("FAIL")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()