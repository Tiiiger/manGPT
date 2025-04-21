import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.ndim = ndim
        self.scale = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.ones(ndim))

        # backward buffer
        self.cache_input = None

    def forward(self, input):
        self.cache_input = input
        centered = (input - input.mean(dim=-1, keepdim=True)) / (input.var(dim=-1, keepdim=True) + 1e-6).sqrt()
        return centered * self.scale + self.bias

    def man_backward(self, d_output):
        # recompute centered and std
        std = (self.cache_input.var(dim=-1, keepdim=True) + 1e-6).sqrt()
        centered = (self.cache_input - self.cache_input.mean(dim=-1, keepdim=True)) / std

        d_scale = (d_output * centered).view(-1, self.ndim).sum(dim=0)

        if self.scale.grad is None:
            self.scale.grad = d_scale
        else:
            self.scale.grad += d_scale

        d_bias = d_output.view(-1, self.ndim).sum(dim=0)
        if self.bias.grad is None:
            self.bias.grad = d_bias
        else:
            self.bias.grad += d_bias

        d_centered = d_output * self.scale

        # d_center / d_input = 
        d_input = std * (
            d_centered - 
            d_centered.sum(dim=-1, keepdim=True) / self.ndim - 
            centered * (d_centered * centered).sum(dim=-1, keepdim=True) / self.ndim
        )

        return d_input

if __name__ == "__main__":
    x = torch.randn(10, 128, 64).to("cuda:0")
    norm = LayerNorm(64).to("cuda:0")
    norm_x = norm(x)
    norm.man_backward(torch.ones_like(norm_x))
        

