import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################
# Implement trainable activation functions
# Define your trainable activation functions here
class ShiLU(nn.Module):
    def __init__(self):
        super(ShiLU, self).__init__()
        self.alpha = nn.Parameter(torch.empty(1))
        self.beta = nn.Parameter(torch.empty(1))

        nn.init.constant_(self.alpha, 1.0)
        nn.init.constant_(self.beta, 1.0)

    def forward(self, x):
        return self.alpha * F.relu(x) + self.beta

class ScaledSoftSign(nn.Module):
    def __init__(self):
        super(ScaledSoftSign, self).__init__()
        self.alpha = nn.Parameter(torch.empty(1))
        self.beta = nn.Parameter(torch.empty(1))
        self.eps = 1e-8

        nn.init.constant_(self.alpha, 1.0)
        nn.init.constant_(self.beta, 1.0)

    def forward(self, x):
        return (self.alpha * x) / (self.beta + torch.abs(x) + self.eps)

class ReLUN(nn.Module):
    def __init__(self):
        super(ReLUN, self).__init__()
        self.n = nn.Parameter(torch.empty(1))
        nn.init.constant_(self.n, 1.0)

    def forward(self, x):
        maxv = torch.max(torch.zeros_like(x), x)
        minv = torch.min(maxv, torch.ones_like(x) * self.n)
        return minv

class HELU(nn.Module):
    def __init__(self):
        super(HELU, self).__init__()

    def forward(self, x):
        left = (x <= 0.0) * F.silu(x)
        right = (x > 0.0) * (0.5 * x + torch.abs(torch.exp(-x) - 1.0))
        return left + right

class DELU(nn.Module):
    def __init__(self):
        super(DELU, self).__init__()
        self.n = nn.Parameter(torch.empty(1))
        nn.init.constant_(self.n, 0.0)

    def forward(self, x):
        left = (x <= 0.0) * F.silu(x)
        delu = (self.n + 0.5) * x + torch.abs(torch.exp(-x) - 1.0)
        right = (x > 0.0) * delu
        return left + right

class SinLU(nn.Module):
    def __init__(self):
        super(SinLU, self).__init__()
        self.a = nn.Parameter(torch.empty(1))
        self.b = nn.Parameter(torch.empty(1))

        nn.init.constant_(self.a, 1.0)
        nn.init.constant_(self.b, 1.0)

    def forward(self, x):
        return F.sigmoid(x) * (x + self.a * torch.sin(self.b * x))

class CosLU(nn.Module):
    def __init__(self):
        super(CosLU, self).__init__()
        self.a = nn.Parameter(torch.empty(1))
        self.b = nn.Parameter(torch.empty(1))

        nn.init.constant_(self.a, 1.0)
        nn.init.constant_(self.b, 1.0)

    def forward(self, x):
        return F.sigmoid(x) * (x + self.a * torch.cos(self.b * x))
