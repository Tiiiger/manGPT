import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.ones(in_features, out_features))
        self.bias = nn.Parameter(torch.ones(out_features))

        # init
        self.init_weight()

        # backward buffer
        self.cache_input = None

    def init_weight(self):
        self.weight.data.normal_(0, 0.02)
        self.bias.data.normal_(0, 0.02)

    def forward(self, input):
        self.cache_input = input
        return input @ self.weight + self.bias

    def man_backward(self, d_output):
        d_input = d_output @ self.weight.transpose(0, 1)
        d_weight = self.cache_input.transpose(-2, -1) @ d_output
        d_bias = d_output.view(-1, self.out_features).sum(dim=0)

        # reset or accumulate
        if d_weight.ndim > 2:
            d_weight = d_weight.view(-1, self.in_features, self.out_features).sum(dim=0)
        if self.weight.grad is None:
            self.weight.grad = d_weight
        else:
            self.weight.grad += d_weight

        if d_bias.ndim > 1:
            d_bias = d_bias.view(-1, self.out_features).sum(dim=0)
        if self.bias.grad is None:
            self.bias.grad = d_bias
        else:
            self.bias.grad += d_bias

        return d_input
