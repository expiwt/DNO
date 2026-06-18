"""Loss functions for DNO training."""

import torch


def count_params(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LpLoss:
    """Relative Lp loss."""
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        assert p > 0
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num = x.size()[0]
        diff_norm = torch.norm(x.reshape(num, -1) - y.reshape(num, -1), self.p, 1)
        y_norm = torch.norm(y.reshape(num, -1), self.p, 1)
        if self.reduction:
            return torch.mean(diff_norm / y_norm) if self.size_average else torch.sum(diff_norm / y_norm)
        return diff_norm / y_norm

    def __call__(self, x, y):
        return self.rel(x, y)


class MaskedLpLoss:
    """LpLoss with mask — ignores holes (mask=0)."""
    def __init__(self, p=2):
        self.p = p

    def rel(self, x, y, mask=None):
        if mask is not None:
            x, y = x * mask, y * mask
        diff = torch.norm(x.view(x.shape[0], -1) - y.view(y.shape[0], -1), p=self.p)
        norm_y = torch.norm(y.view(y.shape[0], -1), p=self.p)
        return diff / (norm_y + 1e-8)
