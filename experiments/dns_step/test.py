# -*- coding: utf-8 -*-
"""
test.py — Inference & visualization for FiLM and LpL models.

Loads a trained model checkpoint, runs on 7 test Re values,
plots True | Pred | Abs Error for U, V, P in physical space,
and reports L2 & MSE errors.

Usage:
    python test.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# 0. Paths

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "dns_averaged_dataset")

FILM_RUN = os.path.join(ROOT, "runs_navier_stokes_FiLM", "bs16_m32_w48_lr0.001_st50")
LPL_RUN  = os.path.join(ROOT, "runs_navier_stokes_LpL",  "bs32_m24_w32_lr0.001_st50")

FILM_OUT = os.path.join(FILM_RUN, "test_inference")
LPL_OUT  = os.path.join(LPL_RUN,  "test_inference")

S = 128        # spatial resolution
TRAIN_RATIO = 0.9
N_SAMPLES = 7  # how many test samples to visualize


# 1. Data loading & normalization (shared)

class StandardScaler:
    def __init__(self, data):
        valid = data[data != 0]
        if len(valid) == 0:
            valid = data
        self.mean = float(np.mean(valid))
        self.std  = float(np.std(valid))
        if self.std == 0:
            self.std = 1.0
    def encode(self, x):
        return (x - self.mean) / self.std
    def decode(self, x):
        return x * self.std + self.mean

def load_test_data():
    """Load data, normalise, split 90/10, return test splits + scalers."""
    raw_x  = np.loadtxt(os.path.join(DATA_DIR, "x_data.csv"),  delimiter=",")
    raw_y  = np.loadtxt(os.path.join(DATA_DIR, "y_data.csv"),  delimiter=",")
    raw_u  = np.loadtxt(os.path.join(DATA_DIR, "u_data.csv"),  delimiter=",")
    raw_v  = np.loadtxt(os.path.join(DATA_DIR, "v_data.csv"),  delimiter=",")
    raw_p  = np.loadtxt(os.path.join(DATA_DIR, "p_data.csv"),  delimiter=",")
    raw_re = np.loadtxt(os.path.join(DATA_DIR, "re_data.csv"), delimiter=",")

    N = raw_u.shape[0]
    if raw_re.ndim == 1:
        raw_re = raw_re.reshape(-1, 1)

    raw_re_spatial = np.repeat(raw_re, S*S, axis=1)

    # crop guard
    raw_x = raw_x[:N]; raw_y = raw_y[:N]; raw_re = raw_re[:N]
    raw_re_spatial = raw_re_spatial[:N]

    X    = np.nan_to_num(raw_x, nan=0.0)
    Y    = np.nan_to_num(raw_y, nan=0.0)
    Re_s = np.nan_to_num(raw_re_spatial, nan=0.0)
    U    = np.nan_to_num(raw_u, nan=0.0)
    V    = np.nan_to_num(raw_v, nan=0.0)
    P    = np.nan_to_num(raw_p, nan=0.0)

    scalers = {k: StandardScaler(d) for k, d in
               zip(['x','y','re','u','v','p'], [X, Y, Re_s, U, V, P])}

    X  = scalers['x'].encode(X).reshape(N, S, S)
    Y  = scalers['y'].encode(Y).reshape(N, S, S)
    U  = scalers['u'].encode(U).reshape(N, S, S)
    V  = scalers['v'].encode(V).reshape(N, S, S)
    P  = scalers['p'].encode(P).reshape(N, S, S)

    # Re stays scalar [N, 1]
    re_scalar = scalers['re'].encode(raw_re)

    # Split
    ntrain = int(N * TRAIN_RATIO)
    # Inputs for FiLM: [N, S, S, 2] (X, Y)  — no Re
    # Inputs for LpL:  [N, S, S, 3] (X, Y, Re)
    inputs_film = np.stack([X, Y], axis=-1)           # [N, S, S, 2]
    inputs_lpl  = np.stack([X, Y,                     # [N, S, S, 3]
        scalers['re'].encode(raw_re_spatial).reshape(N, S, S)], axis=-1)
    targets = np.stack([U, V, P], axis=-1)            # [N, S, S, 3]

    test_a_film = torch.FloatTensor(inputs_film[ntrain:])
    test_a_lpl  = torch.FloatTensor(inputs_lpl[ntrain:])
    test_u      = torch.FloatTensor(targets[ntrain:])
    test_re     = torch.FloatTensor(re_scalar[ntrain:])

    return test_a_film, test_a_lpl, test_u, test_re, scalers


# 2. Model architectures (reproduced from train.py & dno.py)


# ---- 2a. FiLM model (from train.py) ----
class SpectralConv2d_fast(nn.Module):
    """Uses weights1/weights2 to match checkpoint naming."""
    def __init__(self, in_c, out_c, m1, m2):
        super().__init__()
        self.in_c = in_c; self.out_c = out_c
        self.m1 = m1; self.m2 = m2
        s = 1.0 / (in_c * out_c)
        self.weights1 = nn.Parameter(s * torch.rand(in_c, out_c, m1, m2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(s * torch.rand(in_c, out_c, m1, m2, dtype=torch.cfloat))
    def compl_mul2d(self, x, w):
        return torch.einsum("bixy,ioxy->boxy", x, w)
    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_c, x.size(-2), x.size(-1)//2+1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.m1, :self.m2] = self.compl_mul2d(
            x_ft[:, :, :self.m1, :self.m2], self.weights1)
        out_ft[:, :, -self.m1:, :self.m2] = self.compl_mul2d(
            x_ft[:, :, -self.m1:, :self.m2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

class FiLM(nn.Module):
    def __init__(self, width, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, 2 * width))
    def forward(self, x, re_scalar):
        gamma, beta = self.net(re_scalar).chunk(2, dim=-1)  # [N, C]
        return gamma[:, :, None, None] * x + beta[:, :, None, None]

class FNO2d_FiLM(nn.Module):
    """Exact match of train.py: individual attr names, no ModuleList."""
    def __init__(self, modes1, modes2, width):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

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

        self.film0 = FiLM(width)
        self.film1 = FiLM(width)
        self.film2 = FiLM(width)
        self.film3 = FiLM(width)
        self.film4 = FiLM(width)
        self.film5 = FiLM(width)

        self.fc1 = nn.Linear(self.width, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 3)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float32)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float32)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def forward(self, x, re_scalar):
        grid_mesh = x[:, :, :, 0:2]

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
            x = film(x, re_scalar)
            x = F.gelu(x)

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc3(x))
        x = self.fc4(x)
        x = self.fc5(x)
        return x

# ---- 2b. DNO model (from dno.py without external imports) ----
class GridEmbeddingND(nn.Module):
    def __init__(self, dim=2, bounds=[[0,1],[0,1]]):
        super().__init__()
        self.dim = dim
        self.bounds = bounds
        self.register_buffer('grid', torch.empty(1, 0, 0, dim))

    def forward(self, x):
        B, H, W = x.shape[0], x.shape[1], x.shape[2]
        if self.grid.shape[1] != H or self.grid.shape[2] != W:
            gx = torch.linspace(self.bounds[0][0], self.bounds[0][1], H, device=x.device)
            gy = torch.linspace(self.bounds[1][0], self.bounds[1][1], W, device=x.device)
            gx, gy = torch.meshgrid(gx, gy, indexing='ij')
            self.grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).to(x.device)
        return torch.cat([x, self.grid.expand(B, -1, -1, -1)], dim=-1)

class FNO2d_LpL(nn.Module):
    """Same as FNO2d_FiLM but without FiLM layers and with fc0: Linear(3, width)."""
    def __init__(self, modes, width):
        super().__init__()
        self.modes1 = modes
        self.modes2 = modes
        self.width = width

        # fc0 takes 3 channels (X, Y, Re) — no FiLM
        self.fc0 = nn.Linear(3, self.width)

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

        self.fc1 = nn.Linear(self.width, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 3)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float32)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float32)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def forward(self, x):
        # x: [B, H, W, 3] — (X, Y, Re)
        grid_mesh = x[:, :, :, 0:2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        grid_mesh = grid_mesh.permute(0, 3, 1, 2)
        grid = self.get_grid([x.shape[0], x.shape[-2], x.shape[-1]], x.device).permute(0, 3, 1, 2)

        for conv, w, b, c in [
            (self.conv0, self.w0, self.b0, self.c0),
            (self.conv1, self.w1, self.b1, self.c1),
            (self.conv2, self.w2, self.b2, self.c2),
            (self.conv3, self.w3, self.b3, self.c3),
            (self.conv4, self.w4, self.b4, self.c4),
            (self.conv5, self.w5, self.b5, self.c5),
        ]:
            x = conv(x) + w(x) + b(grid) + c(grid_mesh)
            x = F.gelu(x)

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc3(x))
        x = self.fc4(x)
        x = self.fc5(x)
        return x


# 3. Inference & plotting

def run_inference(model, test_inputs, test_re, test_targets, scalers,
                  device, out_dir, is_film=True):
    """
    Run model on 7 evenly spaced test samples and save plots.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    n_test = test_inputs.shape[0]
    indices = np.linspace(0, n_test-1, N_SAMPLES, dtype=int)

    u_scaler = scalers['u']; v_scaler = scalers['v']; p_scaler = scalers['p']
    x_scaler = scalers['x']; y_scaler = scalers['y']; re_scaler = scalers['re']

    geo_mask = None  # computed per sample

    with torch.no_grad():
        for idx_in_test in indices:
            # Prepare sample
            xx = test_inputs[idx_in_test:idx_in_test+1].to(device)
            yy = test_targets[idx_in_test:idx_in_test+1].to(device)

            if is_film:
                re_val = test_re[idx_in_test:idx_in_test+1].to(device)
                pred = model(xx, re_val)
            else:
                pred = model(xx)

            # Decode to physical space
            x_phys = x_scaler.decode(xx[0, :, :, 0].cpu().numpy())
            y_phys = y_scaler.decode(xx[0, :, :, 1].cpu().numpy())

            # Geometry mask (where X,Y ≈ 0 — was NaN in data)
            eps = 1e-3
            geo_mask = (np.abs(x_phys) < eps) & (np.abs(y_phys) < eps)

            # Decode fields
            fields = ['U', 'V', 'P']
            cmaps = ['inferno', 'inferno', 'viridis']
            err_cmaps = ['Reds', 'Reds', 'Reds']

            true_fields = {}
            pred_fields = {}
            err_fields  = {}
            l2_errors = {}
            mse_errors = {}

            for fi, field in enumerate(fields):
                t = yy[0, :, :, fi].cpu().numpy()
                p = pred[0, :, :, fi].cpu().numpy()

                # Decode
                if field == 'U':
                    t_dec = u_scaler.decode(t)
                    p_dec = u_scaler.decode(p)
                elif field == 'V':
                    t_dec = v_scaler.decode(t)
                    p_dec = v_scaler.decode(p)
                else:  # P
                    t_dec = p_scaler.decode(t)
                    p_dec = p_scaler.decode(p)

                err = np.abs(t_dec - p_dec)

                # Mask geometry
                t_m = np.ma.masked_where(geo_mask, t_dec)
                p_m = np.ma.masked_where(geo_mask, p_dec)
                e_m = np.ma.masked_where(geo_mask, err)

                true_fields[field] = t_m
                pred_fields[field] = p_m
                err_fields[field]  = e_m

                # L2 relative and MSE
                diff = t_dec - p_dec
                mse_err = float(np.mean(diff**2))
                l2_err = float(np.sqrt(np.sum(diff**2)) / np.sqrt(np.sum(t_dec**2)))
                l2_errors[field] = l2_err
                mse_errors[field] = mse_err

            # Get Re value in physical units
            if is_film:
                re_phys = float(re_scaler.decode(test_re[idx_in_test].cpu().numpy().reshape(1,-1))[0,0])
            else:
                # Re is the 3rd channel of xx (normalized)
                re_norm = xx[0, 0, 0, 2].cpu().numpy()
                re_phys = float(re_scaler.decode(np.array([[re_norm]]))[0,0])

            # Build 3×3 plot
            fig, axes = plt.subplots(3, 3, figsize=(18, 16))

            row_titles = ['True', 'Pred', 'Abs Error']
            col_titles = fields

            for fi, field in enumerate(fields):
                # Row 0: True
                c = axes[0, fi].pcolormesh(x_phys, y_phys, true_fields[field],
                                           cmap=cmaps[fi], shading='auto')
                axes[0, fi].set_title(f'True {field}', fontsize=12)
                axes[0, fi].set_aspect('equal')
                plt.colorbar(c, ax=axes[0, fi])

                # Row 1: Pred
                c = axes[1, fi].pcolormesh(x_phys, y_phys, pred_fields[field],
                                           cmap=cmaps[fi], shading='auto')
                axes[1, fi].set_title(f'Pred {field}', fontsize=12)
                axes[1, fi].set_aspect('equal')
                plt.colorbar(c, ax=axes[1, fi])

                # Row 2: Error
                c = axes[2, fi].pcolormesh(x_phys, y_phys, err_fields[field],
                                           cmap=err_cmaps[fi], shading='auto')
                axes[2, fi].set_title(f'Error {field}', fontsize=12)
                axes[2, fi].set_aspect('equal')
                plt.colorbar(c, ax=axes[2, fi])

            # Annotate errors
            err_str = (
                f"Re = {re_phys:.0f}  |  "
                f"U: L2={l2_errors['U']:.4f} MSE={mse_errors['U']:.6f}  |  "
                f"V: L2={l2_errors['V']:.4f} MSE={mse_errors['V']:.6f}  |  "
                f"P: L2={l2_errors['P']:.4f} MSE={mse_errors['P']:.6f}"
            )
            fig.suptitle(err_str, fontsize=13, y=0.99)

            for ax in axes.flat:
                ax.set_xlabel('x')
                ax.set_ylabel('y')

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            fname = f"test_re{re_phys:.0f}_idx{idx_in_test}.png"
            plt.savefig(os.path.join(out_dir, fname), dpi=120, bbox_inches='tight')
            plt.close()
            print(f"  Saved {fname}  |  {err_str}")

    # Also save a summary CSV
    print(f"  Done → {out_dir}/")


# 4. Main

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data once
    print("Loading data...")
    test_a_film, test_a_lpl, test_targets, test_re, scalers = load_test_data()
    print(f"  Test samples: {test_a_film.shape[0]}")

    # FiLM model
    print("\n===== FiLM Model =====")
    film_ckpt = torch.load(os.path.join(FILM_RUN, "models", "model_best.pth"),
                           map_location=device, weights_only=True)
    model_film = FNO2d_FiLM(modes1=32, modes2=32, width=48).to(device)
    model_film.load_state_dict(film_ckpt)
    print(f"  Loaded {FILM_RUN} (params={sum(p.numel() for p in model_film.parameters())})")

    run_inference(model_film, test_a_film, test_re, test_targets, scalers,
                  device, FILM_OUT, is_film=True)

    # LpL model
    print("\n===== LpL Model =====")
    lpl_ckpt = torch.load(os.path.join(LPL_RUN, "models", "model_best.pth"),
                          map_location=device, weights_only=True)
    model_lpl = FNO2d_LpL(modes=24, width=32).to(device)
    model_lpl.load_state_dict(lpl_ckpt)
    print(f"  Loaded {LPL_RUN} (params={sum(p.numel() for p in model_lpl.parameters())})")

    run_inference(model_lpl, test_a_lpl, test_re, test_targets, scalers,
                  device, LPL_OUT, is_film=False)

    print("\nDone! Check the output directories:")
    print(f"  {FILM_OUT}/")
    print(f"  {LPL_OUT}/")

if __name__ == "__main__":
    main()
