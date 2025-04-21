from .linear import Linear
from .nonlinear import ReLU
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, h_dim, intermediate_dim):
        super().__init__()
        self.h_dim = h_dim
        self.intermediate_dim = intermediate_dim

        self.up_proj = Linear(h_dim, intermediate_dim)
        self.relu = ReLU()
        self.down_proj = Linear(intermediate_dim, h_dim)

    def forward(self, x):
        return self.down_proj(self.relu(self.up_proj(x)))

    def man_backward(self, d_output):
        temp = self.down_proj.man_backward(d_output)
        temp = self.relu.man_backward(temp)
        return self.up_proj.man_backward(temp)