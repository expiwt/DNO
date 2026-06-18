# -*- coding: utf-8 -*-
"""
FNO2d for Navier-Stokes (Backward-facing step) with FiLM conditioning on Re.
- Inputs: X, Y (deformed coordinates)
- Condition: Re (scalar, via FiLM)
- Outputs: U, V, P (Velocity and Pressure)
"""

import os
import csv
import json
import itertools
import random
from timeit import default_timer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from Loss_function import LpLoss, count_params

# Utils & Normalization
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_csv(path, array_1d):
    arr = np.array(array_1d, dtype=np.float64)
    np.savetxt(path, arr, delimiter=",")

def plot_losses(train_losses, test_losses, path_png):
    plt.figure(figsize=(12, 8), dpi=120)
    plt.plot(range(len(train_losses)), train_losses, label="train loss")
    plt.plot(range(len(test_losses)), test_losses, label="test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss value (Relative Lp)")
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()

class StandardScaler:
    """Z-score нормализатор. Выравнивает данные к среднему 0 и отклонению 1."""
    def __init__(self, data):
        # Игнорируем нули (бывшие NaN) при подсчете статистики
        valid_data = data[data != 0]
        if len(valid_data) == 0:
            valid_data = data
        self.mean = np.mean(valid_data)
        self.std = np.std(valid_data)
        if self.std == 0:
            self.std = 1.0

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean

# Data IO
def load_csv_fast(filepath):
    print(f"Loading {filepath}...")
    return np.loadtxt(filepath, delimiter=",")

def load_fluid_data(data_dir, S, train_ratio=0.9):
    """
    Загружает датасет Навье-Стокса.
    Re НЕ входит в каналы входа, а возвращается отдельно для FiLM-conditioning.
    Возвращает: train_a, train_u, test_a, test_u, train_re, test_re, scalers
    """
    
    raw_x  = load_csv_fast(os.path.join(data_dir, "x_data.csv"))
    raw_y  = load_csv_fast(os.path.join(data_dir, "y_data.csv"))
    raw_u  = load_csv_fast(os.path.join(data_dir, "u_data.csv"))
    raw_v  = load_csv_fast(os.path.join(data_dir, "v_data.csv"))
    raw_p  = load_csv_fast(os.path.join(data_dir, "p_data.csv"))
    
    raw_re = load_csv_fast(os.path.join(data_dir, "re_data.csv"))
    if raw_re.ndim == 1:
        raw_re = raw_re.reshape(-1, 1)  # [N, 1]
    raw_re_spatial = np.repeat(raw_re, S*S, axis=1)  # [N, S*S] — для нормализации

    num_samples = raw_u.shape[0]
    print(f"Loaded {num_samples} samples")

    # Обрезаем под длину физ. данных (страховка)
    raw_x = raw_x[:num_samples]
    raw_y = raw_y[:num_samples]
    raw_re = raw_re[:num_samples]
    raw_re_spatial = raw_re_spatial[:num_samples]

    # Заменяем NaN на нули
    X  = np.nan_to_num(raw_x, nan=0.0)
    Y  = np.nan_to_num(raw_y, nan=0.0)
    Re_s = np.nan_to_num(raw_re_spatial, nan=0.0)
    U  = np.nan_to_num(raw_u, nan=0.0)
    V  = np.nan_to_num(raw_v, nan=0.0)
    P  = np.nan_to_num(raw_p, nan=0.0)

    # Нормализация
    scalers = {
        'x': StandardScaler(X),
        'y': StandardScaler(Y),
        're': StandardScaler(Re_s),
        'u': StandardScaler(U),
        'v': StandardScaler(V),
        'p': StandardScaler(P)
    }

    X = scalers['x'].encode(X)
    Y = scalers['y'].encode(Y)
    U = scalers['u'].encode(U)
    V = scalers['v'].encode(V)
    P = scalers['p'].encode(P)

    # Re нормализуем, но оставляем скаляром [N, 1]
    re_scalar = scalers['re'].encode(raw_re)  # [N, 1]

    # Reshape
    X  = X.reshape(num_samples, S, S)
    Y  = Y.reshape(num_samples, S, S)
    U  = U.reshape(num_samples, S, S)
    V  = V.reshape(num_samples, S, S)
    P  = P.reshape(num_samples, S, S)

    # Вход: [X, Y] — 2 канала (без Re!)
    inputs = np.stack([X, Y], axis=3)
    # Выход: [U, V, P] — 3 канала
    targets = np.stack([U, V, P], axis=3)

    ntrain = int(num_samples * train_ratio)
    
    train_a = inputs[:ntrain]
    train_u = targets[:ntrain]
    train_re = re_scalar[:ntrain]
    test_a  = inputs[ntrain:]
    test_u  = targets[ntrain:]
    test_re  = re_scalar[ntrain:]

    print(f"Data shape: Train inputs {train_a.shape}, Train targets {train_u.shape}")
    return train_a, train_u, test_a, test_u, train_re, test_re, scalers

# FNO Model Architecture
class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation conditioning on Re.
    Преобразует скаляр Re в scale и bias для каждого канала.
    """
    def __init__(self, width, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * width)
        )

    def forward(self, x, re_scalar):
        """
        x: [N, C, H, W]
        re_scalar: [N, 1]
        """
        params = self.net(re_scalar)  # [N, 2*C]
        gamma, beta = params.chunk(2, dim=-1)  # [N, C] each
        gamma = gamma[:, :, None, None]  # [N, C, 1, 1]
        beta = beta[:, :, None, None]    # [N, C, 1, 1]
        return gamma * x + beta


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # ВХОД: 2 канала (X, Y) — Re идёт через FiLM
        self.fc0 = nn.Linear(2, self.width)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv5 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.w4 = nn.Conv2d(self.width, self.width, 1)
        self.w5 = nn.Conv2d(self.width, self.width, 1)

        self.b0 = nn.Conv2d(2, self.width, 1)
        self.b1 = nn.Conv2d(2, self.width, 1)
        self.b2 = nn.Conv2d(2, self.width, 1)
        self.b3 = nn.Conv2d(2, self.width, 1)
        self.b4 = nn.Conv2d(2, self.width, 1)
        self.b5 = nn.Conv2d(2, self.width, 1)

        self.c0 = nn.Conv2d(2, self.width, 1)
        self.c1 = nn.Conv2d(2, self.width, 1)
        self.c2 = nn.Conv2d(2, self.width, 1)
        self.c3 = nn.Conv2d(2, self.width, 1)
        self.c4 = nn.Conv2d(2, self.width, 1)
        self.c5 = nn.Conv2d(2, self.width, 1)

        # FiLM-слои: scale+beta для каждого spectral block
        self.film0 = FiLM(width)
        self.film1 = FiLM(width)
        self.film2 = FiLM(width)
        self.film3 = FiLM(width)
        self.film4 = FiLM(width)
        self.film5 = FiLM(width)

        self.fc1 = nn.Linear(self.width, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        
        # ВЫХОД: 3 канала (U, V, P)
        self.fc5 = nn.Linear(128, 3)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float32)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float32)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def forward(self, x, re_scalar):
        """
        x: [N, H, W, 2] — (X, Y)
        re_scalar: [N, 1] — скаляр Re для FiLM
        """
        grid_mesh = x[:, :, :, 0:2]  # исходные X, Y

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        grid_mesh = grid_mesh.permute(0, 3, 1, 2)
        grid = self.get_grid([x.shape[0], x.shape[-2], x.shape[-1]], x.device).permute(0, 3, 1, 2)

        for conv, w, b, c, film in [
            (self.conv0, self.w0, self.b0, self.c0, self.film0),
            (self.conv1, self.w1, self.b1, self.c1, self.film1),
            (self.conv2, self.w2, self.b2, self.c2, self.film2),
            (self.conv3, self.w3, self.b3, self.c3, self.film3),
            (self.conv4, self.w4, self.b4, self.c4, self.film4),
            (self.conv5, self.w5, self.b5, self.c5, self.film5),
        ]:
            x = conv(x) + w(x) + b(grid) + c(grid_mesh)
            x = film(x, re_scalar)  # FiLM-модуляция по Re
            x = F.gelu(x)

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc3(x))
        x = self.fc4(x)
        x = self.fc5(x)
        return x

# Training Loop
def plot_test_predictions(model, test_a, test_u, test_re, scalers, save_dir, device, myloss, n_samples=10):
    """
    Генерирует n_samples тестовых примеров в физическом пространстве.
    6 панелей: True |v|, True p; Pred |v|, Pred p; Error |v|, Error p.
    Геометрия (ступенька) видна через pcolormesh по деформированным X, Y.
    """
    graphs_dir = os.path.join(save_dir, "graphs")
    ensure_dir(graphs_dir)

    model.eval()
    n_show = min(n_samples, test_a.shape[0])

    u_scaler = scalers['u']
    v_scaler = scalers['v']
    p_scaler = scalers['p']
    x_scaler = scalers['x']
    y_scaler = scalers['y']

    with torch.no_grad():
        for i in range(n_show):
            xx = test_a[i:i+1].to(device)
            yy = test_u[i:i+1].to(device)
            re_val = test_re[i:i+1].to(device)

            pred = model(xx, re_val)

            # Физические координаты сетки (декодированные X, Y)
            x_phys = x_scaler.decode(xx[0, :, :, 0].cpu().numpy())
            y_phys = y_scaler.decode(xx[0, :, :, 1].cpu().numpy())

            # Определяем геометрию: точки внутри ступеньки/твёрдого тела
            # Там, где X, Y ≈ 0 (были NaN → nan_to_num → 0)
            eps = 1e-3
            geo_mask = (np.abs(x_phys) < eps) & (np.abs(y_phys) < eps)

            # Физические поля скорости и давления
            u_true = u_scaler.decode(yy[0, :, :, 0].cpu().numpy())
            v_true = v_scaler.decode(yy[0, :, :, 1].cpu().numpy())
            p_true = p_scaler.decode(yy[0, :, :, 2].cpu().numpy())

            u_pred = u_scaler.decode(pred[0, :, :, 0].cpu().numpy())
            v_pred = v_scaler.decode(pred[0, :, :, 1].cpu().numpy())
            p_pred = p_scaler.decode(pred[0, :, :, 2].cpu().numpy())

            vm_true = np.sqrt(u_true**2 + v_true**2)
            vm_pred = np.sqrt(u_pred**2 + v_pred**2)
            err_vm = np.abs(vm_true - vm_pred)
            err_p = np.abs(p_true - p_pred)

            # Маскируем точки внутри геометрии
            vm_true_m = np.ma.masked_where(geo_mask, vm_true)
            vm_pred_m = np.ma.masked_where(geo_mask, vm_pred)
            p_true_m  = np.ma.masked_where(geo_mask, p_true)
            p_pred_m  = np.ma.masked_where(geo_mask, p_pred)
            err_vm_m  = np.ma.masked_where(geo_mask, err_vm)
            err_p_m   = np.ma.masked_where(geo_mask, err_p)

            # Loss на этот сэмпл
            loss_u = float(myloss(pred[0:1, :, :, 0], yy[0:1, :, :, 0], type=False).cpu())
            loss_v = float(myloss(pred[0:1, :, :, 1], yy[0:1, :, :, 1], type=False).cpu())
            loss_p_val = float(myloss(pred[0:1, :, :, 2], yy[0:1, :, :, 2], type=False).cpu())
            total_loss = loss_u + loss_v + loss_p_val

            re_phys = scalers['re'].decode(np.array([test_re[i].cpu().numpy()]))[0, 0]

            # --- Построение 3×2 в физическом пространстве ---
            fig, axes = plt.subplots(3, 2, figsize=(13, 15))

            plots_data = [
                # (row, col, field, cmap, title)
                (0, 0, vm_true_m, 'inferno', f"True |v| • loss={total_loss:.4f}"),
                (0, 1, p_true_m,  'viridis', "True p"),
                (1, 0, vm_pred_m, 'inferno', "Pred |v|"),
                (1, 1, p_pred_m,  'viridis', "Pred p"),
                (2, 0, err_vm_m,  'Reds',    f"Error |v| • (L2_u={loss_u:.4f}, L2_v={loss_v:.4f})"),
                (2, 1, err_p_m,   'Reds',    f"Error p • L2_p={loss_p_val:.4f}"),
            ]

            for row, col, field, cmap, title in plots_data:
                ax = axes[row, col]
                c = ax.pcolormesh(x_phys, y_phys, field, cmap=cmap, shading='auto')
                ax.set_title(title, fontsize=11)
                ax.set_aspect('equal')
                plt.colorbar(c, ax=ax)

            for ax in axes.flat:
                ax.set_xlabel('x')
                ax.set_ylabel('y')

            plt.suptitle(f"Test sample {i+1}/{n_show} | Re={re_phys:.0f} | Total loss={total_loss:.4f}",
                         fontsize=13, y=0.98)
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_dir, f"sample_{i+1:04d}_re{re_phys:.0f}_loss{total_loss:.4f}.png"),
                        dpi=120, bbox_inches='tight')
            plt.close()

            print(f"  Graph {i+1}/{n_show}: loss={total_loss:.4f} (u={loss_u:.4f}, v={loss_v:.4f}, p={loss_p_val:.4f})")

    print(f"  Graphs saved to {graphs_dir}")


def train_one_run(
    run_dir, device, train_a, train_u, test_a, test_u, train_re, test_re, S,
    batch_size, modes, width, epochs, learning_rate, scheduler_step, scheduler_gamma,
    scalers, weight_decay=1e-4, seed=1234, log_every=10
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "models"))
    ensure_dir(os.path.join(run_dir, "plots"))
    ensure_dir(os.path.join(run_dir, "logs"))

    # Даталоадеры: теперь (вход, выход, Re)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u, train_re),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u, test_re),
        batch_size=batch_size, shuffle=False
    )

    model = FNO2d(modes, modes, width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    
    myloss = LpLoss(size_average=True)

    train_losses = []
    test_losses = []
    best_test = float("inf")

    path_last = os.path.join(run_dir, "models", "model_last.pth")
    path_best = os.path.join(run_dir, "models", "model_best.pth")

    t0 = default_timer()
    for ep in range(epochs):
        # ---- train ----
        model.train()
        sum_train = 0.0
        n_train_batches = 0

        for xx, yy, re_val in train_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            re_val = re_val.to(device)

            pred = model(xx, re_val)

            # Считаем относительную ошибку для каждого поля отдельно
            loss_u = myloss(pred[..., 0], yy[..., 0], type=False)
            loss_v = myloss(pred[..., 1], yy[..., 1], type=False)
            loss_p = myloss(pred[..., 2], yy[..., 2], type=False)

            loss = loss_u + loss_v + loss_p

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_train += float(loss.item())
            n_train_batches += 1

        train_loss_ep = sum_train / max(1, n_train_batches)
        train_losses.append(train_loss_ep)

        # ---- eval ----
        model.eval()
        sum_test = 0.0
        n_test_batches = 0
        with torch.no_grad():
            for xx, yy, re_val in test_loader:
                xx = xx.to(device)
                yy = yy.to(device)
                re_val = re_val.to(device)
                pred = model(xx, re_val)

                loss_u = myloss(pred[..., 0], yy[..., 0], type=False)
                loss_v = myloss(pred[..., 1], yy[..., 1], type=False)
                loss_p = myloss(pred[..., 2], yy[..., 2], type=False)

                loss = loss_u + loss_v + loss_p

                sum_test += float(loss.item())
                n_test_batches += 1

        test_loss_ep = sum_test / max(1, n_test_batches)
        test_losses.append(test_loss_ep)

        scheduler.step()

        if (ep + 1) % log_every == 0 or ep == 0:
            dt = default_timer() - t0
            print(f"  [{ep+1:>3d}/{epochs}] train={train_loss_ep:.6f}  test={test_loss_ep:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  time={dt:.0f}s")

        if test_loss_ep < best_test:
            best_test = test_loss_ep
            torch.save(model.state_dict(), path_best)

    torch.save(model.state_dict(), path_last)

    dt = default_timer() - t0
    print(f"\nDone in {dt:.1f}s. Best test loss: {best_test:.6f}")

    # ---- Визуализация тестовых примеров ----
    print("Generating test sample visualizations...")
    plot_test_predictions(
        model=model,
        test_a=test_a,
        test_u=test_u,
        test_re=test_re,
        scalers=scalers,
        save_dir=run_dir,
        device=device,
        myloss=myloss,
        n_samples=10
    )

    save_csv(os.path.join(run_dir, "logs", "train_loss.csv"), train_losses)
    save_csv(os.path.join(run_dir, "logs", "test_loss.csv"), test_losses)
    plot_losses(train_losses, test_losses, os.path.join(run_dir, "plots", "loss.png"))

    summary = {
        "best_test_loss": best_test,
        "last_test_loss": float(test_losses[-1]) if test_losses else None,
        "epochs": epochs,
        "time_sec": dt,
        "params_count": int(sum(p.numel() for p in model.parameters())),
    }
    with open(os.path.join(run_dir, "logs", "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary

# Main sweep
def main():
    DATA_DIR = "dns_averaged_dataset"
    S = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Загружаем и нормализуем данные (Re отдельно для FiLM)
    train_a, train_u, test_a, test_u, train_re, test_re, scalers = load_fluid_data(DATA_DIR, S, train_ratio=0.9)
    
    root_runs = "./runs_navier_stokes_FiLM"
    ensure_dir(root_runs)
    
    # Скейлеры — для визуализации и инференса
    scalers_dict = {}
    for key, scaler in scalers.items():
        scalers_dict[key] = {'mean': float(scaler.mean), 'std': float(scaler.std)}
    with open(os.path.join(root_runs, "scalers_info.json"), "w") as f:
        json.dump(scalers_dict, f, indent=2)
    print("Scalers info saved to scalers_info.json!")

    train_a = torch.FloatTensor(train_a)
    train_u = torch.FloatTensor(train_u)
    train_re = torch.FloatTensor(train_re)
    test_a = torch.FloatTensor(test_a)
    test_u = torch.FloatTensor(test_u)
    test_re = torch.FloatTensor(test_re)

    # Настройка сетки гиперпараметров
    sweep = {
        "batch_size": [32, 16],
        "modes": [24, 32],
        "width": [32, 48],
        "learning_rate": [1e-3, 5e-4],
        "scheduler_step": [50],
        "scheduler_gamma": [0.5],
    }
    epochs = 120
    weight_decay = 1e-4
    seed = 42

    def fmt_float(x: float) -> str:
        return f"{x:.10f}".rstrip("0").rstrip(".")

    existing_runs = {d.name for d in os.scandir(root_runs) if d.is_dir()}

    keys = list(sweep.keys())
    values = [sweep[k] for k in keys]
    all_results = []

    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        lr_str = fmt_float(cfg["learning_rate"])

        run_name = f"bs{cfg['batch_size']}_m{cfg['modes']}_w{cfg['width']}_lr{lr_str}_st{cfg['scheduler_step']}"

        if run_name in existing_runs:
            print(f"SKIP (already exists): {run_name}")
            continue

        run_dir = os.path.join(root_runs, run_name)
        existing_runs.add(run_name)

        print(f"\n=== RUN: {run_name} ===")
        summary = train_one_run(
            run_dir=run_dir, device=device,
            train_a=train_a, train_u=train_u,
            test_a=test_a, test_u=test_u,
            train_re=train_re, test_re=test_re,
            S=S, batch_size=cfg["batch_size"], modes=cfg["modes"], width=cfg["width"],
            epochs=epochs, learning_rate=cfg["learning_rate"],
            scheduler_step=cfg["scheduler_step"], scheduler_gamma=cfg["scheduler_gamma"],
            scalers=scalers,
            weight_decay=weight_decay, seed=seed, log_every=10
        )
        all_results.append({"run": run_name, **cfg, **summary})

    all_results_sorted = sorted(all_results, key=lambda x: x["best_test_loss"])
    with open(os.path.join(root_runs, "leaderboard.json"), "w", encoding="utf-8") as f:
        json.dump(all_results_sorted, f, ensure_ascii=False, indent=2)

    if len(all_results_sorted) > 0:
        print("\nDone. Best run:", all_results_sorted[0]["run"],
              "best_test_loss=", all_results_sorted[0]["best_test_loss"])

if __name__ == "__main__":
    main()
