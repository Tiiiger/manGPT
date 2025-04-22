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
        # Use unbiased=False to match PyTorch's implementation
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, keepdim=True, unbiased=False)
        std = (var + 1e-5).sqrt()  # Using 1e-5 to match PyTorch default epsilon
        centered = (input - mean) / std
        return centered * self.scale + self.bias

    def man_backward(self, d_output):
        # Recompute centered and std with corrected parameters
        mean = self.cache_input.mean(dim=-1, keepdim=True)
        var = self.cache_input.var(dim=-1, keepdim=True, unbiased=False)
        std = (var + 1e-5).sqrt()
        centered = (self.cache_input - mean) / std

        # Gradient for scale
        d_scale = (d_output * centered).view(-1, self.ndim).sum(dim=0)

        if self.scale.grad is None:
            self.scale.grad = d_scale
        else:
            self.scale.grad += d_scale

        # Gradient for bias
        d_bias = d_output.view(-1, self.ndim).sum(dim=0)
        if self.bias.grad is None:
            self.bias.grad = d_bias
        else:
            self.bias.grad += d_bias

        d_centered = d_output * self.scale

        # d_center / d_input = 
        d_input = (1 / std) * (
            d_centered - 
            d_centered.sum(dim=-1, keepdim=True) / self.ndim - 
            centered * (d_centered * centered).sum(dim=-1, keepdim=True) / self.ndim
        )

        return d_input

def test_layernorm():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Parameters
    batch_size = 3
    seq_len = 5
    hidden_dim = 64
    
    # Create custom and PyTorch LayerNorm layers
    custom_norm = LayerNorm(hidden_dim)
    torch_norm = nn.LayerNorm(hidden_dim)
    
    # Copy weights from custom to PyTorch for fair comparison
    with torch.no_grad():
        torch_norm.weight.copy_(custom_norm.scale)
        torch_norm.bias.copy_(custom_norm.bias)
    
    # Create random input
    input_data = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
    input_data_clone = input_data.detach().clone().requires_grad_(True)
    
    # Forward pass for both implementations
    with torch.no_grad():
        custom_output = custom_norm(input_data)
    
    torch_output = torch_norm(input_data_clone)
    
    # Create a random gradient for backward pass
    grad_output = torch.randn_like(custom_output)
    grad_output_clone = grad_output.clone()
    
    # Backward pass for PyTorch implementation
    torch_output.backward(grad_output_clone)
    
    # Manual backward pass for custom implementation
    with torch.no_grad():
        d_input = custom_norm.man_backward(grad_output)
        
        # Set input gradients manually since our implementation doesn't do this automatically
        if input_data.grad is None:
            input_data.grad = d_input
        else:
            input_data.grad += d_input
    
    # Compare forward results
    forward_diff = torch.abs(custom_output - torch_output).max().item()
    print(f"Max forward difference: {forward_diff:.6f}")
    
    # Compare gradients for weights and biases
    scale_grad_diff = torch.abs(custom_norm.scale.grad - torch_norm.weight.grad).max().item()
    bias_grad_diff = torch.abs(custom_norm.bias.grad - torch_norm.bias.grad).max().item()
    input_grad_diff = torch.abs(input_data.grad - input_data_clone.grad).max().item()
    
    print(f"Max scale gradient difference: {scale_grad_diff:.6f}")
    print(f"Max bias gradient difference: {bias_grad_diff:.6f}")
    print(f"Max input gradient difference: {input_grad_diff:.6f}")
    
    # Check if implementations match within tolerance
    forward_match = forward_diff < 1e-5
    scale_grad_match = scale_grad_diff < 1e-5
    bias_grad_match = bias_grad_diff < 1e-5
    input_grad_match = input_grad_diff < 1e-5
    
    if forward_match and scale_grad_match and bias_grad_match and input_grad_match:
        print("✓ Test passed! Custom implementation matches PyTorch.")
    else:
        print("✗ Test failed! Implementations don't match.")
        if not forward_match:
            print("  - Forward pass outputs differ")
        if not scale_grad_match:
            print("  - Scale gradients differ")
        if not bias_grad_match:
            print("  - Bias gradients differ")
        if not input_grad_match:
            print("  - Input gradients differ")
    
    return forward_match and scale_grad_match and bias_grad_match and input_grad_match


if __name__ == "__main__":
    test_layernorm()
        

