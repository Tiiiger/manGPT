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


def test_linear_layer():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Parameters for the linear layers
    in_features = 10
    out_features = 5
    batch_size = 3
    
    # Create custom and PyTorch linear layers
    custom_linear = Linear(in_features, out_features)
    torch_linear = nn.Linear(in_features, out_features)
    
    # Copy weights and biases from custom to PyTorch for fair comparison
    with torch.no_grad():
        torch_linear.weight.copy_(custom_linear.weight.t())  # PyTorch stores weights as out_features x in_features
        torch_linear.bias.copy_(custom_linear.bias)
    
    # Create random input
    input_data = torch.randn(batch_size, in_features, requires_grad=True)
    input_data_clone = input_data.detach().clone().requires_grad_(True)
    
    # Forward pass for both implementations
    with torch.no_grad():
        custom_output = custom_linear(input_data)

    torch_output = torch_linear(input_data_clone)
    
    # Create a random gradient for backward pass
    grad_output = torch.randn_like(custom_output)
    grad_output_clone = grad_output.clone()
    
    # Backward pass for PyTorch implementation
    torch_output.backward(grad_output_clone)
    
    # Manual backward pass for custom implementation
    with torch.no_grad():
        d_input = custom_linear.man_backward(grad_output)
        
        # Set input gradients manually since our implementation doesn't do this automatically
        if input_data.grad is None:
            input_data.grad = d_input
        else:
            input_data.grad += d_input
    
    # Compare forward results
    forward_diff = torch.abs(custom_output - torch_output).max().item()
    print(f"Max forward difference: {forward_diff:.6f}")
    
    # Compare gradients for weights and biases
    weight_grad_diff = torch.abs(custom_linear.weight.grad - torch_linear.weight.grad.t()).max().item()
    bias_grad_diff = torch.abs(custom_linear.bias.grad - torch_linear.bias.grad).max().item()
    input_grad_diff = torch.abs(input_data.grad - input_data_clone.grad).max().item()
    
    print(f"Max weight gradient difference: {weight_grad_diff:.6f}")
    print(f"Max bias gradient difference: {bias_grad_diff:.6f}")
    print(f"Max input gradient difference: {input_grad_diff:.6f}")
    
    # Check if implementations match within tolerance
    forward_match = forward_diff < 1e-5
    weight_grad_match = weight_grad_diff < 1e-5
    bias_grad_match = bias_grad_diff < 1e-5
    input_grad_match = input_grad_diff < 1e-5
    
    if forward_match and weight_grad_match and bias_grad_match and input_grad_match:
        print("✓ Test passed! Custom implementation matches PyTorch.")
    else:
        print("✗ Test failed! Implementations don't match.")
        if not forward_match:
            print("  - Forward pass outputs differ")
        if not weight_grad_match:
            print("  - Weight gradients differ")
        if not bias_grad_match:
            print("  - Bias gradients differ")
        if not input_grad_match:
            print("  - Input gradients differ")
    
    return forward_match and weight_grad_match and bias_grad_match and input_grad_match


if __name__ == "__main__":
    test_linear_layer()
