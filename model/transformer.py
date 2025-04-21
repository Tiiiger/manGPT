import torch
import torch.nn as nn
from .norm import LayerNorm
from .attn import CausalAttn
from .mlp import MLP
from .embedding import Embedding
from .linear import Linear

class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_head,
        h_dim,
        intermediate_dim,
        max_len=128
    ):
        super().__init__()

        self.h_dim = h_dim
        self.intermediate_dim = intermediate_dim
        self.n_head = n_head

        self.norm1 = LayerNorm(h_dim)
        self.norm2 = LayerNorm(h_dim)
        self.attn = CausalAttn(self.n_head, self.h_dim, max_len=max_len)
        self.mlp = MLP(self.h_dim, self.intermediate_dim)

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))

        return x

    def man_backward(self, d_output):
        d_x = d_output + self.norm2.man_backward(self.mlp.man_backward(d_output))
        d_x = d_x + self.norm2.man_backward(self.attn.man_backward(d_x))

        return d_x

class Transformer(nn.Module):
    def __init__(
        self,
        n_layer,
        n_head,
        h_dim,
        intermediate_dim,
        n_words,
        max_len=128
    ):
        super().__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.h_dim = h_dim
        self.intermediate_dim = intermediate_dim
        self.max_len = max_len

        self.token_emb = Embedding(n_words=n_words, h_dim=h_dim)
        self.pos_emb = Embedding(n_words=max_len, h_dim=h_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(
                n_head=n_head,
                h_dim=h_dim,
                intermediate_dim=intermediate_dim,
                max_len=max_len
            ) for _ in range(n_layer)
        ])

        self.norm_final = LayerNorm(h_dim)
        self.lm_head = Linear(h_dim, n_words)

    def forward(self, input_ids, mask):
        token_emb = self.token_emb(input_ids)
        positions = torch.arange(0, input_ids.size(-1), device=input_ids.device).unsqueeze(0).repeat(input_ids.size(0), 1)
        pos_emb = self.pos_emb(positions)

        x = token_emb + pos_emb
        for block in self.layers:
            x = block(x, mask)

        x = self.norm_final(x)
        logits = self.lm_head(x)

        return logits

    def man_backward(self, d_logits):
        d_x = self.lm_head.man_backward(d_logits)
        d_x = self.norm_final.man_backward(d_x)

        for block in self.layers[::-1]:
            d_x = block.man_backward(d_x)

        self.token_emb.man_backward(d_x)
        self.pos_emb.man_backward(d_x)