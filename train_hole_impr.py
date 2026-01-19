# -*- coding: utf-8 -*-
"""
Grid training for Darcy Flow (hole) with FNO2d.
- Hyperparameter sweep (grid search)
- Each run saves into its own folder named by params
- Saves: config.json, train/test loss csv, loss plot png, model_last.pth, model_best.pth
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

from Loss_function import LpLoss, count_params  # keep your import
# from Adam import Adam  # not used (you use AdamW)


# -----------------------
# Utils
# -----------------------
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
    plt.ylabel("Loss value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()


def mem(tag=""):
    if not torch.cuda.is_available():
        return
    a = torch.cuda.memory_allocated() / 1024**2
    r = torch.cuda.memory_reserved() / 1024**2
    m = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[{tag}] alloc={a:.0f}MB reserv={r:.0f}MB peak={m:.0f}MB")


# -----------------------
# Data IO (same logic as yours)
# -----------------------
def readfile(path):
    with open(path, "r") as f:
        rows = list(csv.reader(f))
    return rows


def openfile(filename, dataset):
    K = readfile("./" + dataset + "/" + filename + ".csv")
    for t in range(len(K)):
        K[t] = [float(v) for v in K[t]]
    return K


def load_data(train_dataset_list, test_dataset_list, S):
    print("Loading data with DYNAMIC Masking (Moving Holes)...")

    def safe_load(filename, dataset_list):
        raw_data = np.array(openfile(filename, dataset_list[0]))
        if len(dataset_list) > 1:
            for i in range(1, len(dataset_list)):
                next_chunk = np.array(openfile(filename, dataset_list[i]))
                raw_data = np.vstack((raw_data, next_chunk))
        return raw_data

    raw_train_C = safe_load("train_C", train_dataset_list)
    raw_test_C = safe_load("test_C", test_dataset_list)

    # mask: 1 = valid, 0 = hole
    train_mask = (~np.isnan(raw_train_C)).astype(np.float32)
    test_mask = (~np.isnan(raw_test_C)).astype(np.float32)

    train_C = np.nan_to_num(raw_train_C, nan=0.0, posinf=0.0, neginf=0.0)
    train_x = np.nan_to_num(safe_load("train_x_data", train_dataset_list), nan=0.0, posinf=0.0, neginf=0.0)
    train_y = np.nan_to_num(safe_load("train_y_data", train_dataset_list), nan=0.0, posinf=0.0, neginf=0.0)
    train_U = np.nan_to_num(safe_load("train_U", train_dataset_list), nan=0.0, posinf=0.0, neginf=0.0)

    test_C = np.nan_to_num(raw_test_C, nan=0.0, posinf=0.0, neginf=0.0)
    test_x = np.nan_to_num(safe_load("test_x_data", test_dataset_list), nan=0.0, posinf=0.0, neginf=0.0)
    test_y = np.nan_to_num(safe_load("test_y_data", test_dataset_list), nan=0.0, posinf=0.0, neginf=0.0)
    test_U = np.nan_to_num(safe_load("test_U", test_dataset_list), nan=0.0, posinf=0.0, neginf=0.0)

    ntrain = train_C.shape[0]
    ntest = test_C.shape[0]

    # reshape to SxS images
    train_C = train_C.reshape(ntrain, S, S)
    train_x = train_x.reshape(ntrain, S, S)
    train_y = train_y.reshape(ntrain, S, S)
    train_mask = train_mask.reshape(ntrain, S, S)
    train_U = train_U.reshape(ntrain, S, S)

    test_C = test_C.reshape(ntest, S, S)
    test_x = test_x.reshape(ntest, S, S)
    test_y = test_y.reshape(ntest, S, S)
    test_mask = test_mask.reshape(ntest, S, S)
    test_U = test_U.reshape(ntest, S, S)

    # stack channels: [C, x, y, mask]
    train_a = np.stack([train_C, train_x, train_y, train_mask], axis=3)
    test_a = np.stack([test_C, test_x, test_y, test_mask], axis=3)

    # scale output
    train_u = np.expand_dims(train_U, axis=-1) * 10.0
    test_u = np.expand_dims(test_U, axis=-1) * 10.0

    return train_a, train_u, test_a, test_u


# -----------------------
# Model (same as yours, but kept intact)
# -----------------------
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


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.fc0 = nn.Linear(4, self.width)

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

        self.c0 = nn.Conv2d(3, self.width, 1)
        self.c1 = nn.Conv2d(3, self.width, 1)
        self.c2 = nn.Conv2d(3, self.width, 1)
        self.c3 = nn.Conv2d(3, self.width, 1)
        self.c4 = nn.Conv2d(3, self.width, 1)
        self.c5 = nn.Conv2d(3, self.width, 1)

        self.fc1 = nn.Linear(self.width, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 1)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float32)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float32)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def forward(self, x):
        grid_mesh = x[:, :, :, 1:4]  # (x, y, mask)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # [B,C,H,W]

        grid_mesh = grid_mesh.permute(0, 3, 1, 2)  # [B,3,H,W]
        grid = self.get_grid([x.shape[0], x.shape[-2], x.shape[-1]], x.device).permute(0, 3, 1, 2)  # [B,2,H,W]

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

        x = x.permute(0, 2, 3, 1)  # [B,H,W,C]
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc3(x))
        x = self.fc4(x)
        x = self.fc5(x)
        return x


# -----------------------
# Training / evaluation
# -----------------------
def train_one_run(
    run_dir,
    device,
    train_a, train_u, test_a, test_u,
    S,
    batch_size,
    modes,
    width,
    epochs,
    learning_rate,
    scheduler_step,
    scheduler_gamma,
    weight_decay=1e-4,
    seed=1234,
    log_every=10,
    use_mem_log=False,
):
    # seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "models"))
    ensure_dir(os.path.join(run_dir, "plots"))
    ensure_dir(os.path.join(run_dir, "logs"))

    # dataloaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u),
        batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u),
        batch_size=batch_size, shuffle=False, drop_last=False
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
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # ---- train ----
        model.train()
        sum_train = 0.0
        n_train_batches = 0

        for batch_id, (xx, yy) in enumerate(train_loader):
            xx = xx.to(device)
            yy = yy.to(device)

            # random coordinate shift (as in your code)
            for i in range(xx.shape[0]):
                x_bias = (np.random.random() - 0.5) * 20.0
                y_bias = (np.random.random() - 0.5) * 20.0
                xx[i][:, :, 1] = xx[i][:, :, 1] + x_bias
                xx[i][:, :, 2] = xx[i][:, :, 2] + y_bias

            mask = xx[..., 3:4]               # [B,H,W,1]
            xx_masked = xx.clone()
            xx_masked[..., :3] *= mask        # zero C,x,y in hole

            if use_mem_log and batch_id == 0:
                mem(f"ep{ep} start")

            pred = model(xx_masked)

            pred_masked = pred * mask
            yy_masked = yy * mask

            bsz = xx.shape[0]
            loss = myloss(
                pred_masked.reshape(bsz, -1),
                yy_masked.reshape(bsz, -1),
                type=False
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_train += float(loss.item())
            n_train_batches += 1

            if use_mem_log and batch_id == 0:
                mem(f"ep{ep} after_step")

        train_loss_ep = sum_train / max(1, n_train_batches)
        train_losses.append(train_loss_ep)

        # ---- eval ----
        model.eval()
        sum_test = 0.0
        n_test_batches = 0
        with torch.no_grad():
            for xx, yy in test_loader:
                xx = xx.to(device)
                yy = yy.to(device)

                mask = xx[..., 3:4]
                xx_masked = xx.clone()
                xx_masked[..., :3] *= mask

                pred = model(xx_masked)
                pred_masked = pred * mask
                yy_masked = yy * mask

                bsz = xx.shape[0]
                loss = myloss(
                    pred_masked.reshape(bsz, -1),
                    yy_masked.reshape(bsz, -1),
                    type=False
                )

                sum_test += float(loss.item())
                n_test_batches += 1

        test_loss_ep = sum_test / max(1, n_test_batches)
        test_losses.append(test_loss_ep)

        # save best
        if test_loss_ep < best_test:
            best_test = test_loss_ep
            torch.save(model.state_dict(), path_best)

        # save last each epoch (можно реже, если хочешь)
        torch.save(model.state_dict(), path_last)

        # scheduler step per epoch
        scheduler.step()

        if ep % log_every == 0:
            lr_curr = optimizer.state_dict()["param_groups"][0]["lr"]
            print(f"[{os.path.basename(run_dir)}] ep={ep:03d} lr={lr_curr:.6f} train={train_loss_ep:.6f} test={test_loss_ep:.6f}")

    dt = default_timer() - t0

    # save logs
    save_csv(os.path.join(run_dir, "logs", "train_loss.csv"), train_losses)
    save_csv(os.path.join(run_dir, "logs", "test_loss.csv"), test_losses)
    plot_losses(train_losses, test_losses, os.path.join(run_dir, "plots", "loss.png"))

    # save summary
    summary = {
        "best_test_loss": best_test,
        "last_test_loss": float(test_losses[-1]) if test_losses else None,
        "train_samples": int(train_a.shape[0]),
        "test_samples": int(test_a.shape[0]),
        "epochs": epochs,
        "time_sec": dt,
        "params_count": int(sum(p.numel() for p in model.parameters())),
    }
    with open(os.path.join(run_dir, "logs", "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


# -----------------------
# Main sweep
# -----------------------
def main():
    # ====== DATASET CONFIG ======
    train_dataset_list = ["sq_data_w_hole"]
    test_dataset_list = ["sq_data_w_hole"]
    S = 128

    # ====== DEVICE ======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # ====== LOAD DATA ONCE ======
    train_a, train_u, test_a, test_u = load_data(train_dataset_list, test_dataset_list, S)
    train_a = torch.FloatTensor(train_a)
    train_u = torch.FloatTensor(train_u)
    test_a = torch.FloatTensor(test_a)
    test_u = torch.FloatTensor(test_u)

    # ====== SWEEP GRID ======
    sweep = {
        "batch_size": [8,16],                # под тебя
        "modes": [20, 24],
        "width": [24, 32, 48],
        "learning_rate": [1e-3, 3e-4],
        "scheduler_step": [20, 30],
        "scheduler_gamma": [0.9, 0.7],
    }
    epochs = 150
    weight_decay = 1e-4
    seed = 1234

    root_runs = "./runs"
    ensure_dir(root_runs)

    # --- ДОБАВЛЕНО: функция “красиво” печатать float в имени папки ---
    def fmt_float(x: float) -> str:
        # 0.001 -> "0.001", 3e-4 -> "0.0003", 0.9 -> "0.9"
        return f"{x:.10f}".rstrip("0").rstrip(".")

    # --- ДОБАВЛЕНО: сканируем уже существующие папки ---
    existing_runs = {
        d.name for d in os.scandir(root_runs)
        if d.is_dir()
    }

    keys = list(sweep.keys())
    values = [sweep[k] for k in keys]

    all_results = []

    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))

        lr_str = fmt_float(cfg["learning_rate"])
        gm_str = fmt_float(cfg["scheduler_gamma"])

        run_name = (
            f"bs{cfg['batch_size']}_m{cfg['modes']}_w{cfg['width']}_"
            f"lr{lr_str}_st{cfg['scheduler_step']}_gm{gm_str}"
        )

        # --- ДОБАВЛЕНО: пропускаем уже существующие ---
        if run_name in existing_runs:
            print(f"SKIP (already exists): {run_name}")
            continue

        run_dir = os.path.join(root_runs, run_name)
        ensure_dir(run_dir)

        # (опционально) чтобы в рамках одного запуска тоже учитывалось
        existing_runs.add(run_name)

        # save config
        run_cfg = {
            **cfg,
            "epochs": epochs,
            "S": S,
            "weight_decay": weight_decay,
            "seed": seed,
            "train_dataset_list": train_dataset_list,
            "test_dataset_list": test_dataset_list,
        }
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(run_cfg, f, ensure_ascii=False, indent=2)

        print(f"\n=== RUN: {run_name} ===")
        summary = train_one_run(
            run_dir=run_dir,
            device=device,
            train_a=train_a, train_u=train_u,
            test_a=test_a, test_u=test_u,
            S=S,
            batch_size=cfg["batch_size"],
            modes=cfg["modes"],
            width=cfg["width"],
            epochs=epochs,
            learning_rate=cfg["learning_rate"],
            scheduler_step=cfg["scheduler_step"],
            scheduler_gamma=cfg["scheduler_gamma"],
            weight_decay=weight_decay,
            seed=seed,
            log_every=10,
            use_mem_log=False,   # включи True, если хочешь печатать VRAM
        )
        all_results.append({"run": run_name, **run_cfg, **summary})

    # save global leaderboard
    all_results_sorted = sorted(all_results, key=lambda x: x["best_test_loss"])
    with open(os.path.join(root_runs, "leaderboard.json"), "w", encoding="utf-8") as f:
        json.dump(all_results_sorted, f, ensure_ascii=False, indent=2)

    if len(all_results_sorted) > 0:
        print("\nDone. Best run:", all_results_sorted[0]["run"],
              "best_test_loss=", all_results_sorted[0]["best_test_loss"])
    else:
        print("\nDone. No new runs were launched (all existed).")


if __name__ == "__main__":
    main()
