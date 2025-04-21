import torch
import torch.nn as nn

class ReLU(nn.Module):

    def __init__(self):
        super().__init__()

        # backward buffer
        self.cache_mask = None

    def forward(self, input):
        self.cache_mask = input > 0
        return input * self.cache_mask

    def man_backward(self, d_output):
        return d_output * self.cache_mask