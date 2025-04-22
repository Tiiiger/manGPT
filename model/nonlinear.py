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

def test_relu_layer():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Parameters
    batch_size = 3
    features = 10
    
    # Create custom and PyTorch ReLU layers
    custom_relu = ReLU()
    torch_relu = nn.ReLU()
    
    # Create random input with both positive and negative values
    input_data = torch.randn(batch_size, features, requires_grad=True)
    input_data_clone = input_data.detach().clone().requires_grad_(True)
    
    # Forward pass for both implementations
    with torch.no_grad():
        custom_output = custom_relu(input_data)
    
    torch_output = torch_relu(input_data_clone)
    
    # Create a random gradient for backward pass
    grad_output = torch.randn_like(custom_output)
    grad_output_clone = grad_output.clone()
    
    # Backward pass for PyTorch implementation
    torch_output.backward(grad_output_clone)
    
    # Manual backward pass for custom implementation
    with torch.no_grad():
        d_input = custom_relu.man_backward(grad_output)
        
        # Set input gradients manually since our implementation doesn't do this automatically
        if input_data.grad is None:
            input_data.grad = d_input
        else:
            input_data.grad += d_input
    
    # Compare forward results
    forward_diff = torch.abs(custom_output - torch_output).max().item()
    print(f"Max forward difference: {forward_diff:.6f}")
    
    # Compare input gradients
    input_grad_diff = torch.abs(input_data.grad - input_data_clone.grad).max().item()
    print(f"Max input gradient difference: {input_grad_diff:.6f}")
    
    # Check if implementations match within tolerance
    forward_match = forward_diff < 1e-5
    input_grad_match = input_grad_diff < 1e-5
    
    if forward_match and input_grad_match:
        print("✓ Test passed! Custom implementation matches PyTorch.")
    else:
        print("✗ Test failed! Implementations don't match.")
        if not forward_match:
            print("  - Forward pass outputs differ")
        if not input_grad_match:
            print("  - Input gradients differ")
    
    return forward_match and input_grad_match


if __name__ == "__main__":
    test_relu_layer()