# test_quad_corrected.py - ИСПРАВЛЕННАЯ ВЕРСИЯ ДЛЯ ЧЕТЫРЕХУГОЛЬНИКОВ
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata
import matplotlib.tri as tri
import open3d as o3d
from skimage.metrics import structural_similarity as ssim
from matplotlib.colors import Normalize, ListedColormap
import matplotlib.colors as colors

# ПОЛНОЕ ОТКЛЮЧЕНИЕ CUDA
torch.cuda.is_available = lambda: False
device = torch.device('cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("Running on CPU (CUDA disabled)")

# Импортируем архитектуру модели
print("Importing model architecture...")

class SpectralConv2d_fast(torch.nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = torch.nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = torch.nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(torch.nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = torch.nn.Linear(4, self.width)
        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv5 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)

        self.w0 = torch.nn.Conv2d(self.width, self.width, 1)
        self.w1 = torch.nn.Conv2d(self.width, self.width, 1)
        self.w2 = torch.nn.Conv2d(self.width, self.width, 1)
        self.w3 = torch.nn.Conv2d(self.width, self.width, 1)
        self.w4 = torch.nn.Conv2d(self.width, self.width, 1)
        self.w5 = torch.nn.Conv2d(self.width, self.width, 1)
        
        self.b0 = torch.nn.Conv2d(2, self.width, 1)
        self.b1 = torch.nn.Conv2d(2, self.width, 1)
        self.b2 = torch.nn.Conv2d(2, self.width, 1)
        self.b3 = torch.nn.Conv2d(2, self.width, 1)
        self.b4 = torch.nn.Conv2d(2, self.width, 1)
        self.b5 = torch.nn.Conv2d(2, self.width, 1)
        
        self.c0 = torch.nn.Conv2d(3, self.width, 1)
        self.c1 = torch.nn.Conv2d(3, self.width, 1)
        self.c2 = torch.nn.Conv2d(3, self.width, 1)
        self.c3 = torch.nn.Conv2d(3, self.width, 1)
        self.c4 = torch.nn.Conv2d(3, self.width, 1)
        self.c5 = torch.nn.Conv2d(3, self.width, 1)

        self.fc1 = torch.nn.Linear(self.width, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 128)
        self.fc5 = torch.nn.Linear(128, 1)

    def forward(self, x):
        grid_mesh = x[:,:,:,1:4]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        grid_mesh = grid_mesh.permute(0, 3, 1, 2)
        grid = self.get_grid([x.shape[0], x.shape[-2], x.shape[-1]], x.device).permute(0, 3, 1, 2)
       
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x3 = self.b0(grid)
        x4 = self.c0(grid_mesh)
        x = x1 + x2 + x3 + x4
        x = torch.nn.functional.gelu(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x3 = self.b1(grid)
        x4 = self.c1(grid_mesh)
        x = x1 + x2 + x3 + x4
        x = torch.nn.functional.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x3 = self.b2(grid) 
        x4 = self.c2(grid_mesh)
        x = x1 + x2 + x3 + x4
        x = torch.nn.functional.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x3 = self.b3(grid) 
        x4 = self.c3(grid_mesh)
        x = x1 + x2 + x3 + x4
        x = torch.nn.functional.gelu(x)
        
        x1 = self.conv4(x)
        x2 = self.w4(x)
        x3 = self.b4(grid) 
        x4 = self.c4(grid_mesh)
        x = x1 + x2 + x3 + x4
        x = torch.nn.functional.gelu(x)
        
        x1 = self.conv5(x)
        x2 = self.w5(x)
        x3 = self.b5(grid)
        x4 = self.c5(grid_mesh)
        x = x1 + x2 + x3 + x4
        x = torch.nn.functional.gelu(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc3(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class LpLoss(object):
    def __init__(self, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        self.size_average = size_average
        self.reduction = reduction

    def __call__(self, pred, target, type=False):
        # Защита от NaN и бесконечных значений
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if type:
            loss = torch.mean(torch.abs(pred - target))
        else:
            # Добавляем маленькое значение чтобы избежать деления на ноль
            denominator = torch.abs(target) + 1e-8
            loss = torch.mean(torch.abs(pred - target) / denominator)
        return loss

def load_quad_data():
    """Загружаем данные четырехугольников из папки quad_results"""
    if key == 7: 
        print("Loading sev test data from sev_data_w_hole...")
        data_path = 'sev_data_w_hole/'
    else:
        print("Loading quad test data from sq_data_w_hole...")
        data_path = 'sq_data_w_hole/'
    
    try:
        # Проверяем существование файлов
        required_files = ['test_C.csv', 'test_U.csv', 'test_x_data.csv', 'test_y_data.csv']
        for file in required_files:
            if not os.path.exists(data_path + file):
                print(f"File not found: {data_path + file}")
                return None, None, 0

        # --- ШАГ 1: ЗАГРУЖАЕМ СЫРЫЕ ДАННЫЕ С NaN ---
        # Мы предполагаем, что в test_C.csv (и в других) есть NaN в дырках
        raw_C = np.loadtxt(data_path + 'test_C.csv', delimiter=',')
        
        # --- ШАГ 2: СОЗДАЕМ МАСКУ ИЗ NaN (КАК В TRAIN) ---
        # Если в C есть NaN, используем их. Если нет, пробуем по C=0.
        if np.isnan(raw_C).any():
            print("Маска создается по NaN в поле C (ПРАВИЛЬНО).")
            boundary_masks = (~np.isnan(raw_C)).astype(np.float32)
        else:
            print("ВНИМАНИЕ: NaN не найдены в C. Маска создается по C != 0.")
            boundary_masks = (raw_C != 0).astype(np.float32)

        # --- ШАГ 3: ТЕПЕРЬ ОЧИЩАЕМ ВСЕ ДАННЫЕ ОТ NaN ---
        test_C = np.nan_to_num(raw_C, nan=0.0)
        test_U = np.nan_to_num(np.loadtxt(data_path + 'test_U.csv', delimiter=','), nan=0.0)
        test_x = np.nan_to_num(np.loadtxt(data_path + 'test_x_data.csv', delimiter=','), nan=0.0)
        test_y = np.nan_to_num(np.loadtxt(data_path + 'test_y_data.csv', delimiter=','), nan=0.0)
        
        # --- ШАГ 4: RESHAPE ДАННЫХ ---
        n_samples = test_C.shape[0]
        S = 128
        test_C = test_C.reshape(n_samples, S, S)
        test_U = test_U.reshape(n_samples, S, S)
        test_x = test_x.reshape(n_samples, S, S)
        test_y = test_y.reshape(n_samples, S, S)
        boundary_masks = boundary_masks.reshape(n_samples, S, S)
        
        # --- ШАГ 5: СОБОРКА ВХОДНОГО ТЕНЗОРА ---
        test_a = np.stack([test_C, test_x, test_y, boundary_masks], axis=-1)
        test_u = test_U[..., np.newaxis] * 10
        
        print(f"Final shapes: test_a{test_a.shape}, test_u{test_u.shape}")
        
        return torch.FloatTensor(test_a), torch.FloatTensor(test_u), n_samples

    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0


def test_model_on_quads():
    """Тестируем модель на четырехугольниках"""
    
    # Загружаем модель
    base_name = 'model_best'
    
    possible_paths = [
        f'model/{base_name}.pth',   # Приоритет 1: с расширением в папке model
        f'model/{base_name}',       # Приоритет 2: без расширения в папке model
        f'{base_name}.pth',         # Приоритет 3: в корне
        f'{base_name}'              # Приоритет 4: в корне без расширения
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            print(f"[OK] Found model file: {path}")
            break
            
    if model_path is None:
        print(f"[FAIL] Model '{base_name}' not found anywhere.")
        print("Check if training finished and where it saved the file.")
        return None, None, None, None, None, None

    # 2. Загружаем модель
    try:
        # ВАЖНО: параметры должны совпадать с train_hole_impr.py
        modes = 16      # или то значение, с которым обучалась ЛУЧШАЯ модель
        width = 32      # то же самое
        model = FNO2d(modes, modes, width).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"[ERROR] Failed to load model architecture: {e}")
        print("Make sure classes 'FNO2d' and 'SpectralConv2d_fast' are defined in THIS script exactly as in training!")
        return None, None, None, None, None, None


    print("Loading test data...")
    test_a, test_u, n_samples = load_quad_data()
    
    if test_a is None:
        print("[FAIL] Could not load data. Aborting test.")
        return None, None, None, None, None, None
    # --------------------------

    
    # Перемещаем на device
    test_a = test_a.to(device)
    test_u = test_u.to(device)
    
    # Тестируем
    print(f"Testing on {n_samples} samples...")
    
    predictions = []
    L2_errors = []
    MSE_errors = []
    SSIM_scores = []
    
    myloss = LpLoss(size_average=True)
    MSEloss = torch.nn.MSELoss(reduction='mean')
    
    with torch.no_grad():
        for i in range(min(n_samples, 30)):
            sample_a = test_a[i:i+1]
            sample_u = test_u[i:i+1]
            
            # Проверяем данные перед подачей в модель
            if torch.isnan(sample_a).any() or torch.isinf(sample_a).any():
                print(f"Sample {i+1}: Input contains NaN or Inf, skipping")
                continue
                
            mask = sample_a[..., 3:4]         # [1, S, S, 1]
            sample_a_masked = sample_a.clone()
            sample_a_masked[..., :3] *= mask  # обнуляем C, x, y в дырке

            pred = model(sample_a_masked)
            
            # Проверяем предсказание на NaN
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                print(f"Sample {i+1}: Prediction contains NaN or Inf")
                pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)
            
            pred = pred * mask 
            # Вычисляем ошибки как в первом коде
            #L2_error = myloss(sample_u.reshape(1,-1), pred.reshape(1,-1), type=False)
            diff_norm = torch.norm(sample_u.reshape(1, -1) - pred.reshape(1, -1), p=2)
            true_norm = torch.norm(sample_u.reshape(1, -1), p=2)

            # Добавляем 1e-8, чтобы не делить на чистый ноль, если сэмпл пустой
            L2_error = diff_norm / (true_norm + 1e-8)
            MSE_error = MSEloss(sample_u.reshape(1,-1), pred.reshape(1,-1))
            
            # Вычисляем SSIM
            truth_u_np = sample_u[0,:,:,0].cpu().numpy()
            pred_u_np = pred[0,:,:,0].cpu().numpy()
            
            # Очищаем от NaN для SSIM
            truth_u_np = np.nan_to_num(truth_u_np, nan=0.0)
            pred_u_np = np.nan_to_num(pred_u_np, nan=0.0)
            
            # Проверяем диапазон данных для SSIM
            data_range = truth_u_np.max() - truth_u_np.min()
            if data_range == 0:
                data_range = 1.0  # избегаем деления на ноль
                
            try:
                SSIM_score = ssim(truth_u_np, pred_u_np, data_range=data_range)
            except:
                SSIM_score = 0.0
            
            L2_errors.append(L2_error.item())
            MSE_errors.append(MSE_error.item())
            SSIM_scores.append(SSIM_score)
            predictions.append(pred.cpu().numpy())
            
            print(f" Sample {i+1}: L2_error = {L2_error.item():.6f}, MSE = {MSE_error.item():.6f}, SSIM = {SSIM_score:.4f}")
    
    # Статистика
    if L2_errors:
        # Очищаем от возможных NaN в ошибках
        L2_errors = [x if not np.isnan(x) else 0.0 for x in L2_errors]
        MSE_errors = [x if not np.isnan(x) else 0.0 for x in MSE_errors]
        SSIM_scores = [x if not np.isnan(x) else 0.0 for x in SSIM_scores]
        
        avg_L2 = np.mean(L2_errors)
        avg_MSE = np.mean(MSE_errors)
        avg_SSIM = np.mean(SSIM_scores)
        
        print(f"\nTest Results on Quads:")
        print(f"  Average L2 Error: {avg_L2:.6f}")
        print(f"  Average MSE: {avg_MSE:.6f}")
        print(f"  Average SSIM: {avg_SSIM:.4f}")
        print(f"  Min L2 Error: {np.min(L2_errors):.6f}")
        print(f"  Max L2 Error: {np.max(L2_errors):.6f}")
    else:
        print("No valid samples tested")
        return None, None, None, None, None, None
    
    return predictions, L2_errors, MSE_errors, SSIM_scores, test_a.cpu().numpy(), test_u.cpu().numpy()

def visualize_results(predictions, L2_errors, MSE_errors, SSIM_scores, test_a, test_u, num_samples=5):
    """
    3x2:
      Row1: Matrix True  | Physical True  (scatter по узлам)
      Row2: Matrix Pred  | Physical Pred  (scatter по узлам)
      Row3: Matrix Error | Physical Error (scatter по узлам)

    Маска:
      - Mask из test_a[...,3] (1 = физдомен, 0 = дырка)
      - В физике рисуем только точки Mask==1 (через valid_idx)
    """
    n_vis = min(num_samples, len(predictions))
    print(f"Visualizing {n_vis} samples...")

    save_folder = "test_sev_pic_results" if key == 7 else "test_sq_pic_results"
    os.makedirs(save_folder, exist_ok=True)

    SUPTITLE_FS = 20
    CBAR_TICK_FS = 12
    TICK_FS = 12
    AX_TITLE_FS = 23
    TITLE_PAD = 18  


    for i in range(n_vis):
        # --- unpack true/pred ---
        sample_u = test_u[i]
        if hasattr(sample_u, "detach"):
            sample_u = sample_u.detach().cpu().numpy()
        U_true = np.squeeze(sample_u)

        sample_pred = predictions[i]
        if hasattr(sample_pred, "detach"):
            sample_pred = sample_pred.detach().cpu().numpy()
        U_pred = np.squeeze(sample_pred)

        # --- unpack coords/mask ---
        sample_a = test_a[i]  # [S, S, 4]
        if hasattr(sample_a, "detach"):
            sample_a = sample_a.detach().cpu().numpy()

        X = np.squeeze(sample_a[:, :, 1])
        Y = np.squeeze(sample_a[:, :, 2])
        Mask = np.squeeze(sample_a[:, :, 3])

        # --- matrix views (no NaNs) ---
        U_true_mat = np.nan_to_num(U_true)
        U_pred_mat = np.nan_to_num(U_pred)
        Error_mat = np.abs(U_true_mat - U_pred_mat)

        # --- physical views (keep NaNs for stats if needed, but plot via valid_idx) ---
        U_true_phys = U_true.copy()
        U_pred_phys = U_pred.copy()
        U_true_phys[Mask < 0.5] = np.nan
        U_pred_phys[Mask < 0.5] = np.nan
        Error_phys = np.abs(U_true_phys - U_pred_phys)

        # --- flatten + filter for scatter ---
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Mask_flat = Mask.flatten()
        valid_idx = (Mask_flat > 0.5)

        X_valid = X_flat[valid_idx]
        Y_valid = Y_flat[valid_idx]

        U_true_valid = U_true.flatten()[valid_idx]
        U_pred_valid = U_pred.flatten()[valid_idx]
        Error_valid = np.abs(U_true - U_pred).flatten()[valid_idx]

        # --- shared color limits (so matrix/physical comparable) ---
        # Handle edge case: if everything masked -> skip plotting safely
        if X_valid.size == 0:
            print(f"[WARN] Sample {i}: no valid points (Mask is empty). Skipping.")
            continue

        # True/Pred use same scale
        u_min = np.nanmin([np.nanmin(U_true_phys), np.nanmin(U_pred_phys)])
        u_max = np.nanmax([np.nanmax(U_true_phys), np.nanmax(U_pred_phys)])
        if not np.isfinite(u_min) or not np.isfinite(u_max) or (u_max == u_min):
            u_min, u_max = 0.0, 1.0

        # Error scale
        e_max = np.nanmax(Error_phys)
        if not np.isfinite(e_max) or (e_max == 0):
            e_max = 1.0

        # --- plot 3x2 ---
        fig, axs = plt.subplots(3, 2, figsize=(14, 18))
        fig.suptitle(
            f"Образец {i} | L2_relative={L2_errors[i]:.4f} | MSE={MSE_errors[i]:.4e} | SSIM={SSIM_scores[i]:.4f}",
            fontsize=SUPTITLE_FS
        )

        # общие настройки осей (делаем одинаковую рамку и одинаковые пределы)
        for r in range(3):
            # Одинаковая визуальная рамка (высота/ширина оси) слева и справа
            axs[r, 0].set_box_aspect(1)
            axs[r, 1].set_box_aspect(1)

            # Левую колонку НЕ трогаем по координатам (imshow сам разрулит как раньше)
            # Можно явно, но не обязательно:
            axs[r, 0].set_aspect("equal", adjustable="box")

            # Правая колонка: сохраняем "физические" единицы (1:1), но НЕ фиксируем xlim/ylim
            # и НЕ даем matplotlib сжимать рамку -> пусть меняет datalim
            axs[r, 1].set_aspect("equal", adjustable="datalim")

        # Row 1: True
        axs[0, 0].set_title("Универсальное пр-во: истина",  fontsize=AX_TITLE_FS, pad=TITLE_PAD)
        im1 = axs[0, 0].imshow(
            U_true_mat, cmap="jet", origin="lower",
            vmin=u_min, vmax=u_max,
            extent=(0.0, 1.0, 0.0, 1.0)   # <<<<<< ключевое: оси 0..1 вместо 0..S-1
        )
        cb1 = fig.colorbar(im1, ax=axs[0, 0])
        cb1.ax.tick_params(labelsize=CBAR_TICK_FS)

        axs[0, 1].set_title("Физическое пр-во: истина",  fontsize=AX_TITLE_FS, pad=TITLE_PAD)
        axs[0, 1].set_facecolor("white")
        sc1 = axs[0, 1].scatter(X_valid, Y_valid, c=U_true_valid, cmap="jet", s=2, vmin=u_min, vmax=u_max)
        cb2 = fig.colorbar(sc1, ax=axs[0, 1])
        cb2.ax.tick_params(labelsize=CBAR_TICK_FS)

        # Row 2: Pred
        axs[1, 0].set_title("Универсальное пр-во: предсказанное",  fontsize=AX_TITLE_FS, pad=TITLE_PAD)
        im2 = axs[1, 0].imshow(
            U_pred_mat, cmap="jet", origin="lower",
            vmin=u_min, vmax=u_max,
            extent=(0.0, 1.0, 0.0, 1.0)
        )
        cb3 = fig.colorbar(im2, ax=axs[1, 0])
        cb3.ax.tick_params(labelsize=CBAR_TICK_FS)

        axs[1, 1].set_title("Физическое пр-во: предсказанное",  fontsize=AX_TITLE_FS, pad=TITLE_PAD)
        axs[1, 1].set_facecolor("white")
        sc2 = axs[1, 1].scatter(X_valid, Y_valid, c=U_pred_valid, cmap="jet", s=2, vmin=u_min, vmax=u_max)
        cb4 = fig.colorbar(sc2, ax=axs[1, 1])
        cb4.ax.tick_params(labelsize=CBAR_TICK_FS)

        # Row 3: Error
        axs[2, 0].set_title("Универсальное пр-во: ошибка",  fontsize=AX_TITLE_FS, pad=TITLE_PAD)
        im3 = axs[2, 0].imshow(
            Error_mat, cmap="plasma", origin="lower",
            vmin=0.0, vmax=e_max,
            extent=(0.0, 1.0, 0.0, 1.0)
        )
        cb5 = fig.colorbar(im3, ax=axs[2, 0])
        cb5.ax.tick_params(labelsize=CBAR_TICK_FS)

        axs[2, 1].set_title("Физическое пр-во: ошибка",  fontsize=AX_TITLE_FS, pad=TITLE_PAD)
        axs[2, 1].set_facecolor("white")
        sc3 = axs[2, 1].scatter(X_valid, Y_valid, c=Error_valid, cmap="plasma", s=2, vmin=0.0, vmax=e_max)
        cb6 = fig.colorbar(sc3, ax=axs[2, 1])
        cb6.ax.tick_params(labelsize=CBAR_TICK_FS)


        for ax in axs.ravel():
            ax.tick_params(labelsize=TICK_FS)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        path = os.path.join(save_folder, f"clean_result_{i}.png")
        plt.savefig(path, bbox_inches="tight", dpi=200)
        plt.close()
        print(f"Saved clean plot: {path}")



def save_error_stats(L2_errors, MSE_errors, SSIM_scores):
    """Сохраняем статистику ошибок как в первом коде"""
    if not L2_errors:
        return
    
    # Вычисляем средние значения
    avg_L2 = np.mean(L2_errors)
    avg_MSE = np.mean(MSE_errors)
    avg_SSIM = np.mean(SSIM_scores)
    
    # Добавляем средние значения в конец массивов
    L2_errors_with_avg = np.append(L2_errors, avg_L2)
    MSE_errors_with_avg = np.append(MSE_errors, avg_MSE)
    SSIM_scores_with_avg = np.append(SSIM_scores, avg_SSIM)
    
    # Сохраняем в файлы
    if key == 7:
        np.savetxt('hole_sev_test_results/test_error_L2_error.csv', L2_errors_with_avg, delimiter=',')
        np.savetxt('hole_sev_test_results/test_error_MSE.csv', MSE_errors_with_avg, delimiter=',')
        np.savetxt('hole_sev_test_results/test_error_SSIM.csv', SSIM_scores_with_avg, delimiter=',')
    
    else:
        np.savetxt('hole_sq_test_results/test_error_L2_error.csv', L2_errors_with_avg, delimiter=',')
        np.savetxt('hole_sq_test_results/test_error_MSE.csv', MSE_errors_with_avg, delimiter=',')
        np.savetxt('hole_sq_test_results/test_error_SSIM.csv', SSIM_scores_with_avg, delimiter=',')
    
    print(f"\nError statistics saved:")
    print(f" MRAE: {avg_L2:.6f}")
    print(f"  SSIM: {avg_SSIM:.4f}")
    print(f"  MSE: {avg_MSE:.6f}")

def main():
   
    """Основная функция тестирования"""
    print("Testing FNO Model on Quad Darcy Flow")
    print("=" * 50)
    
    # Тестируем модель
    results = test_model_on_quads()
    if results[0] is None:
        print("Testing failed. Exiting.")
        return
        
    predictions, L2_errors, MSE_errors, SSIM_scores, test_a, test_u = results
    
    # Визуализируем результаты
    visualize_results(predictions, L2_errors, MSE_errors, SSIM_scores, test_a, test_u, 
                     num_samples= len(predictions))
    
    # Сохраняем статистику ошибок
    save_error_stats(L2_errors, MSE_errors, SSIM_scores)
    
    print(f"\nTesting completed!")
    if key == 7:
        print(f"Results saved in: hole_sev_test_results/")
    else:
        print(f"Results saved in: hole_sq_test_results/")
    print(f"Tested {len(predictions)} samples")

if __name__ == "__main__":
    key = 7
    main()
