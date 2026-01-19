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
from Loss_function import LpLoss


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



def load_quad_data():
    """Загружаем данные четырехугольников из папки quad_results"""
    print("Loading quad test data from quad_results...")
    
    data_path = 'quad_results/'
    
    try:
        # Проверяем существование файлов
        required_files = ['test_C.csv', 'test_U.csv', 'test_x_data.csv', 'test_y_data.csv']
        for file in required_files:
            if not os.path.exists(data_path + file):
                print(f"File not found: {data_path + file}")
                return None, None, 0
        
        # Загружаем данные
        test_C = np.loadtxt(data_path + 'test_C.csv', delimiter=',')
        test_U = np.loadtxt(data_path + 'test_U.csv', delimiter=',')
        test_x = np.loadtxt(data_path + 'test_x_data.csv', delimiter=',')
        test_y = np.loadtxt(data_path + 'test_y_data.csv', delimiter=',')
        
        print(f"Data shapes: C{test_C.shape}, U{test_U.shape}")
        
        # Очищаем данные от NaN и бесконечных значений
        test_C = np.nan_to_num(test_C, nan=0.0, posinf=1.0, neginf=-1.0)
        test_U = np.nan_to_num(test_U, nan=0.0, posinf=1.0, neginf=-1.0)
        test_x = np.nan_to_num(test_x, nan=0.0, posinf=1.0, neginf=-1.0)
        test_y = np.nan_to_num(test_y, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Reshape данных
        n_samples = test_C.shape[0]
        S = 128
        
        # Reshape из (n_samples, 16384) в (n_samples, 128, 128)
        test_C = test_C.reshape(n_samples, S, S)
        test_U = test_U.reshape(n_samples, S, S)
        test_x = test_x.reshape(n_samples, S, S)
        test_y = test_y.reshape(n_samples, S, S)

        domain_masks = np.ones((n_samples, S, S), dtype=np.float32)

        print(f"Reshaped to 128x128: C{test_C.shape}, U{test_U.shape}")
        
        # Создаем boundary mask (область где координаты не нулевые)
        # boundary mask как в train_final.py: 1 по границе, 0 внутри
        b = np.zeros((S-2, S-2), dtype=np.float32)
        b = np.pad(b, pad_width=1, mode='constant', constant_values=1)  # (S,S)
        boundary_masks = np.repeat(b[None, :, :], n_samples, axis=0)     # (n_samples,S,S)

        
        # Подготавливаем входные данные
        test_a = np.stack([test_C, test_x, test_y, boundary_masks], axis=-1)
        test_u = test_U[..., np.newaxis] * 10
        
        print(f"Final shapes: test_a{test_a.shape}, test_u{test_u.shape}")
        
        return torch.FloatTensor(test_a), torch.FloatTensor(test_u), n_samples, domain_masks


    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0

def test_model_on_quads():
    """Тестируем модель на четырехугольниках"""
    
    # Загружаем модель
    print("Loading trained model...")
    try:
        modes = 16
        width = 32

        model_path = 'model/model_data_geo5_r128_to_ep0140.pth'  # пример: выбери нужную эпоху
        model = FNO2d(modes, modes, width).to(device)

        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print(f"Loaded checkpoint: {model_path}")


        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None
    
    # Загружаем данные
    test_a, test_u, n_samples, domain_masks = load_quad_data()
    if test_a is None:
        return None, None, None, None, None, None
    
    # Перемещаем на device
    test_a = test_a.to(device)
    test_u = test_u.to(device)
    
    # Тестируем
    print(f"Testing on {n_samples} quad samples...")
    
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
                
            pred = model(sample_a)
            
            # Проверяем предсказание на NaN
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                print(f"Sample {i+1}: Prediction contains NaN or Inf")
                pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Вычисляем ошибки как в первом коде
            L2_error = myloss(sample_u.reshape(1,-1), pred.reshape(1,-1), type=False)
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
    
    return predictions, L2_errors, MSE_errors, SSIM_scores, test_a.cpu().numpy(), test_u.cpu().numpy(), domain_masks

def visualize_results(predictions, L2_errors, MSE_errors, SSIM_scores, test_a, test_u, domain_masks, num_samples=5):
    """Визуализируем результаты для четырехугольников в стиле первого кода"""
    S = test_a.shape[1]
    
    print(f"Visualizing {num_samples} samples...")
    
    os.makedirs('quad_test_results', exist_ok=True)
    
    for i in range(min(num_samples, len(predictions))):
        fig = plt.figure(num=1, figsize=(5, 12), dpi=100)
        
        plt.subplots_adjust(left=0.17, bottom=0.10, right=0.95, top=0.95, wspace=0.05, hspace=0.3)
        
        # Исходные данные
        x_coords = test_a[i, :, :, 1]
        y_coords = test_a[i, :, :, 2]
        #boundary_mask = test_a[i, :, :, 3]
        domain_mask = domain_masks[i]
        domain_mask = np.nan_to_num(domain_mask, nan=0.0)

        U_true = test_u[i, :, :, 0]
        
        # Получаем предсказание
        if predictions[i].ndim == 4:
            U_pred = predictions[i][0, :, :, 0]
        else:
            U_pred = predictions[i][:, :, 0]
        
        # Очищаем данные от NaN
        U_true = np.nan_to_num(U_true, nan=0.0)
        U_pred = np.nan_to_num(U_pred, nan=0.0)
        #boundary_mask = np.nan_to_num(boundary_mask, nan=0.0)
        
        # ПРИМЕНЯЕМ МАСКУ К ДАННЫМ
        mask = domain_mask > 0.5
        
        # Маскируем данные - устанавливаем значения вне области в NaN
        U_true_masked = np.where(mask, U_true, np.nan)
        U_pred_masked = np.where(mask, U_pred, np.nan)
        
        # Создаем сетку координат для триангуляции
        XY = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)
        
        try:
            # Триангуляция (как в основном коде)
            Z = np.random.random((XY.shape[0], 1)) * 0.5
            XYZ = np.hstack([XY, Z])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(XYZ)
            pcd.estimate_normals()
            
            alpha = 0.80
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            triang = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
            use_triangulation = True
            
        except Exception as e:
            print(f"Triangulation failed for sample {i}, using regular grid: {e}")
            use_triangulation = False
        
        # Настройки цветовых диапазонов как в первом коде
        # Определяем диапазон на основе реальных данных
        vmin_true = np.nanmin(U_true_masked) if not np.isnan(np.nanmin(U_true_masked)) else -1
        vmax_true = np.nanmax(U_true_masked) if not np.isnan(np.nanmax(U_true_masked)) else 8
        
        low = min(vmin_true, -1)
        up = max(vmax_true, 8)
        
        # 1. Истинное давление U (как в первом коде - subplot 3,1,1)
        ax1 = fig.add_subplot(3, 1, 1)
        plt.xlabel('x', labelpad=-5)
        plt.ylabel('y')
        
        if use_triangulation:
            sampled_true = griddata((XY[:, 0], XY[:, 1]), U_true_masked.flatten(), 
                                  (vertices[:, 0], vertices[:, 1]), method='cubic', fill_value=np.nan)
            tri1 = ax1.tripcolor(triang, sampled_true, shading='gouraud', 
                               cmap=plt.cm.rainbow, vmin=low, vmax=up)
        else:
            # Если триангуляция не удалась, используем обычный plot
            im1 = ax1.imshow(U_true_masked, cmap=plt.cm.rainbow, origin='lower', 
                           vmin=low, vmax=up)
        
        cbar1 = plt.colorbar(tri1 if use_triangulation else im1, ax=ax1, 
                           orientation='vertical', ticks=[0, 4, 8])
        ax1.axis('off')
        ax1.set_title('True Pressure', pad=20)
        
        # 2. Предсказанное давление U (как в первом коде - subplot 3,1,2)
        ax2 = fig.add_subplot(3, 1, 2)
        plt.xlabel('x', labelpad=-10)
        plt.ylabel('y')
        
        if use_triangulation:
            sampled_pred = griddata((XY[:, 0], XY[:, 1]), U_pred_masked.flatten(), 
                                  (vertices[:, 0], vertices[:, 1]), method='cubic', fill_value=np.nan)
            tri2 = ax2.tripcolor(triang, sampled_pred, shading='gouraud', 
                               cmap=plt.cm.rainbow, vmin=low, vmax=up)
        else:
            im2 = ax2.imshow(U_pred_masked, cmap=plt.cm.rainbow, origin='lower', 
                           vmin=low, vmax=up)
        
        cbar2 = plt.colorbar(tri2 if use_triangulation else im2, ax=ax2, 
                           orientation='vertical', ticks=[0, 4, 8])
        ax2.axis('off')
        ax2.set_title('Predicted Pressure', pad=20)
        
        # 3. Абсолютная разница (как в первом коде - subplot 3,1,3)
        ax3 = fig.add_subplot(3, 1, 3)
        plt.xlabel('x')
        plt.ylabel('y')
        
        diff = np.abs(U_true - U_pred)
        diff_masked = np.where(mask, diff, np.nan)
        
        # Настройки как в первом коде: low=0, up=1
        low_diff = 0
        up_diff = 1
        
        if use_triangulation:
            sampled_diff = griddata((XY[:, 0], XY[:, 1]), diff_masked.flatten(),
                                  (vertices[:, 0], vertices[:, 1]), method='cubic', fill_value=np.nan)
            tri3 = ax3.tripcolor(triang, sampled_diff, shading='gouraud', 
                               cmap=plt.cm.Reds, vmin=low_diff, vmax=up_diff)
        else:
            im3 = ax3.imshow(diff_masked, cmap=plt.cm.Reds, origin='lower', 
                           vmin=low_diff, vmax=up_diff)
        
        norm = Normalize(vmin=0, vmax=1)
        cbar3 = plt.colorbar(tri3 if use_triangulation else im3, ax=ax3, 
                           orientation='vertical', norm=norm, ticks=[0, 0.5, 1])
        ax3.axis('off')
        ax3.set_title('Absolute Error', pad=20)
        
        # Добавляем информацию об ошибках как в первом коде
        plt.figtext(0.02, 0.98, f"Sample {i+1}", fontsize=12, ha='left')
        plt.figtext(0.02, 0.95, f"L2 Error: {L2_errors[i]:.6f}", fontsize=10, ha='left')
        plt.figtext(0.02, 0.92, f"MSE: {MSE_errors[i]:.6f}", fontsize=10, ha='left')
        plt.figtext(0.02, 0.89, f"SSIM: {SSIM_scores[i]:.4f}", fontsize=10, ha='left')
        
        plt.savefig(f'quad_test_results/quad_result_{i+1:02d}.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization: quad_result_{i+1:02d}.png")
    
    print(f"All visualizations saved to quad_test_results/")

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
    np.savetxt('quad_test_results/test_error_L2_error.csv', L2_errors_with_avg, delimiter=',')
    np.savetxt('quad_test_results/test_error_MSE.csv', MSE_errors_with_avg, delimiter=',')
    np.savetxt('quad_test_results/test_error_SSIM.csv', SSIM_scores_with_avg, delimiter=',')
    
    print(f"\nError statistics saved:")
    print(f"  L2_error: {avg_L2:.6f}")
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
        
    predictions, L2_errors, MSE_errors, SSIM_scores, test_a, test_u, domain_masks = results
    visualize_results(predictions, L2_errors, MSE_errors, SSIM_scores, test_a, test_u, domain_masks,
                    num_samples=len(predictions))

    # Сохраняем статистику ошибок
    save_error_stats(L2_errors, MSE_errors, SSIM_scores)
    
    print(f"\nTesting completed!")
    print(f"Results saved in: quad_test_results/")
    print(f"Tested {len(predictions)} samples")

if __name__ == "__main__":
    main()
