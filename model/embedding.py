import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, n_words, h_dim):
        super().__init__()
        self.embeddings = nn.Parameter(torch.ones(n_words, h_dim))

        # init
        self.init_weight()

        # backward buffer
        self.cache_input = None

    def init_weight(self):
        self.embeddings.data.normal_(0, 0.02)

    def forward(self, input):
        # input: BxT
        self.cache_input = input
        return self.embeddings[input]

    def man_backward(self, d_output):
        # Initialize gradient tensor
        d_embeddings = torch.zeros_like(self.embeddings)
        
        # Flatten the input indices and output gradients
        flat_indices = self.cache_input.reshape(-1)  # shape: [batch_size * seq_len]
        flat_d_output = d_output.reshape(-1, d_output.size(-1))  # shape: [batch_size * seq_len, h_dim]
        
        # Use index_add_ to accumulate gradients at the appropriate indices
        d_embeddings.index_add_(0, flat_indices, flat_d_output)
        
        if self.embeddings.grad is None:
            self.embeddings.grad = d_embeddings
        else:
            self.embeddings.grad += d_embeddings

        # no need to propagate gradients to tokens
        return None