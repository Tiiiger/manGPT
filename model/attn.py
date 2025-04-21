import torch
import torch.nn as nn
from .linear import Linear
import math

def softmax_backward(dL_dS, S):
    # dL_dS: gradient of loss with respect to softmax output
    # dL_dS: B x n_head x T2 x T1
    # S: softmax output
    
    # First term: element-wise product
    grad = S * dL_dS
    
    # Second term: sum reduction along last dimension and outer product
    sum_term = (S * dL_dS).sum(dim=-1, keepdim=True)
    grad -= S * sum_term
    
    return grad

class CausalAttn(nn.Module):

    def __init__(self, n_head, h_dim, max_len=128):
        super().__init__()

        self.n_head = n_head
        self.h_dim = h_dim

        self.attn_proj = Linear(h_dim, 3 * h_dim)
        self.output_proj = Linear(h_dim, h_dim)

        # get max len
        self.causal_mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool)).view(1, 1, max_len, max_len)

        # backward buffer
        self.cache_input = None
        self.cache_attn_weights = None
        self.cache_v = None
        self.cache_k = None
        self.cache_q = None
        self.cache_o = None
        self.cache_mask = None

    def forward(self, input, mask):
        self.cache_input = input
        # input: B x T x H
        # mask: B x T x T
        B, T, H = input.size()

        # q, k, v: B x T x H
        # reshape as head_dim = H / self.n_had
        # q, k, v: B x n_head x T x head_dim
        q, k, v = self.attn_proj(input).split(self.h_dim, dim=-1)
        q = q.view(B, T, self.n_head, H // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, H // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, H // self.n_head).transpose(1, 2)

        self.cache_q = q
        self.cache_k = k
        self.cache_v = v

        # attn matrix
        # if K is T1 x head_dim, Q is T2 x head_dim
        # attn_weights: B x n_head x T2 x T1
        attn_weights = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        # mask: B x T
        key_padding_mask = mask.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x T
        mask = torch.logical_and(self.causal_mask[:, :, :T, :T].to(key_padding_mask.device), key_padding_mask)

        # mask fill
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = attn_weights.softmax(dim=-1)

        self.cache_attn_weights = attn_weights
        self.cache_mask = mask

        # combine with v
        # o: B x n_head x T2 x head_dim
        o = attn_weights @ v
        o = o.transpose(1, 2).contiguous().view(B, T, H)

        y = self.output_proj(o)

        return y

    def man_backward(self, d_output):
        B, T, H = d_output.size()
        d_o = self.output_proj.man_backward(d_output)

        # 
        d_o = d_o.view(B, T, self.n_head, H // self.n_head).transpose(1, 2)
        d_v = self.cache_attn_weights.transpose(-2, -1) @ d_o
        # d_attn_weights: B x self.n_head x T2 x T1
        d_attn_weights = d_o @ self.cache_v.transpose(-2, -1)

        # gradient through softmax
        d_logits = softmax_backward(d_attn_weights, self.cache_attn_weights)
        d_logits = d_logits.masked_fill(self.cache_mask == 0, 0)

        # d_logits: B x n_head x T2 x T1
        d_k = d_logits.transpose(-2, -1) @ self.cache_q / math.sqrt(self.cache_q.size(-1))
        d_q = d_logits @ self.cache_k / math.sqrt(self.cache_q.size(-1))

        # BxTxH
        d_q = d_q.transpose(1, 2).contiguous().view(B, T, H)
        d_k = d_k.transpose(1, 2).contiguous().view(B, T, H)
        d_v = d_v.transpose(1, 2).contiguous().view(B, T, H)
        d_proj = torch.cat([d_q, d_k, d_v], dim=2)

        d_input = self.attn_proj.man_backward(d_proj)

        return d_input

def test_causal_attn_backward():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Initialize parameters
    batch_size = 2
    seq_len = 10
    n_head = 4
    h_dim = 64
    
    # Create model
    attn = CausalAttn(n_head=n_head, h_dim=h_dim, max_len=seq_len).to("cuda:0")
    
    # Create identical model for autograd
    attn_auto = CausalAttn(n_head=n_head, h_dim=h_dim, max_len=seq_len).to("cuda:0")
    attn_auto.attn_proj.weight.data.copy_(attn.attn_proj.weight.data)
    attn_auto.attn_proj.bias.data.copy_(attn.attn_proj.bias.data)
    attn_auto.output_proj.weight.data.copy_(attn.output_proj.weight.data)
    attn_auto.output_proj.bias.data.copy_(attn.output_proj.bias.data)
    
    # Create inputs
    x = torch.randn(batch_size, seq_len, h_dim).to("cuda:0")
    mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool).to("cuda:0")  # Full attention mask
    
    # Forward pass with manual implementation
    with torch.no_grad():
        y_manual = attn(x, mask)
    
    # Make a copy with gradients for autograd
    x_auto = x.clone().detach().requires_grad_(True)
    
    # Forward pass with autograd
    y_auto = attn_auto(x_auto, mask)
    
    # Verify forward passes match
    assert torch.allclose(y_manual, y_auto, rtol=1e-5, atol=1e-5), "Forward passes don't match!"
    print("✓ Forward passes match")
    
    # Create random gradient for backward pass
    grad_output = torch.randn_like(y_manual)
    
    # Manual backward with torch.no_grad()
    with torch.no_grad():
        dx_manual = attn.man_backward(grad_output)
    
    # Autograd backward
    y_auto.backward(grad_output)
    dx_auto = x_auto.grad
    
    # Compare results
    manual_norm = dx_manual.norm().item()
    auto_norm = dx_auto.norm().item()
    diff_norm = (dx_manual - dx_auto).norm().item()
    rel_error = diff_norm / auto_norm
    
    print(f"Manual gradient norm: {manual_norm:.6f}")
    print(f"Autograd gradient norm: {auto_norm:.6f}")
    print(f"Difference norm: {diff_norm:.6f}")
    print(f"Relative error: {rel_error:.6f}")
    
    # Check if gradients match within tolerance
    assert torch.allclose(dx_manual, dx_auto, rtol=1e-4, atol=1e-5), "Gradients don't match!"
    print("✓ Gradients match within tolerance")
    
    return True

if __name__ == "__main__":
    print("Testing CausalAttn backward pass...")
    test_passed = test_causal_attn_backward()
    if test_passed:
        print("All tests passed!")