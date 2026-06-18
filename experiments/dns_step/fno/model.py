#!/usr/bin/env python3
"""
model.py — FNO2d with FiLM conditioning for stationary NS.

Архитектура:
  Input:  (3, N, N): [mask, ξ, η]
  → Lift (Conv2d 3→width)
  → 4× FourierLayer (width channels, modes)
  → FiLM (Re conditioning)
  → Project (Conv2d width→3)
  Output: (3, N, N): [u, v, p]
"""
import torch
import torch.nn as nn
from timeit import default_timer


# Spectral Convolution (Fourier Layer)
class SpectralConv2d(nn.Module):
    """2D Fourier layer: FFT → linear in freq → IFFT + residual."""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels,
                                     modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels,
                                     modes1, modes2, dtype=torch.cfloat)
        )

    def _mul_complex(self, input, weights):
        """Complex batch matrix multiply: (b, i, x, y) × (i, o, x, y) → (b, o, x, y)."""
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels,
                              x.size(-2), x.size(-1) // 2 + 1,
                              dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self._mul_complex(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self._mul_complex(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


# FiLM: Feature-wise Linear Modulation (Re conditioning)
class FiLM(nn.Module):
    """Скаляр Re → scale и bias для каждого канала."""
    def __init__(self, width, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * width),
        )

    def forward(self, x, re_scalar):
        """
        x : (N, C, H, W)
        re_scalar : (N, 1)
        Returns : x modulated (N, C, H, W)
        """
        gamma_beta = self.net(re_scalar)           # (N, 2*C)
        gamma, beta = gamma_beta.chunk(2, dim=1)   # each (N, C)
        gamma = gamma[:, :, None, None]            # (N, C, 1, 1)
        beta = beta[:, :, None, None]
        return gamma * x + beta


# FNO2d Model
class FNO2d(nn.Module):
    """
    FNO2d с FiLM-conditioning на Re.

    Parameters
    modes1, modes2 : int — число мод Фурье по x и y
    width : int — число каналов в скрытых слоях
    n_layers : int — число Fourier слоёв
    """
    def __init__(self, modes1=12, modes2=12, width=32, n_layers=4):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers

        # Lift: проекция входа в скрытое пространство
        self.fc0 = nn.Linear(3, self.width)  # 3 входа: mask, ξ, η

        # Fourier слои
        self.spectral_convs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.spectral_convs.append(
                SpectralConv2d(width, width, modes1, modes2)
            )
            self.convs.append(
                nn.Conv2d(width, width, 1)  # pointwise residual
            )

        # FiLM (применяется после Fourier слоёв)
        self.film = FiLM(width)

        # Project: обратно в 3 канала
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x, re_scalar=None):
        """
        x : (N, 3, H, W) — [mask, ξ, η]
        re_scalar : (N, 1) or None
        Returns : (N, 3, H, W) — [u, v, p]
        """
        grid = x.permute(0, 2, 3, 1)              # (N, H, W, 3)
        x = self.fc0(grid)                         # (N, H, W, width)
        x = x.permute(0, 3, 1, 2)                 # (N, width, H, W)

        # Fourier слои
        for k in range(self.n_layers):
            x1 = self.spectral_convs[k](x)
            x2 = self.convs[k](x)
            x = x1 + x2
            x = torch.tanh(x)

        # FiLM (Re conditioning)
        if re_scalar is not None:
            x = self.film(x, re_scalar)

        # Project
        x = x.permute(0, 2, 3, 1)                 # (N, H, W, width)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)                            # (N, H, W, 3)
        x = x.permute(0, 3, 1, 2)                 # (N, 3, H, W)

        return x


# Подсчёт параметров
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Тест
if __name__ == "__main__":
    model = FNO2d(modes1=12, modes2=12, width=32, n_layers=4)
    print(f"Model params: {count_params(model):,}")

    x = torch.randn(4, 3, 128, 128)
    re = torch.randn(4, 1)
    y = model(x, re)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"  min={y.min():.4f}, max={y.max():.4f}, "
          f"mean={y.mean():.4f}, std={y.std():.4f}")
