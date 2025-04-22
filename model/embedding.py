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

def test_embedding_layer():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Parameters for the embedding layers
    n_words = 1000
    h_dim = 16
    batch_size = 3
    seq_len = 8
    
    # Create custom and PyTorch embedding layers
    custom_embedding = Embedding(n_words, h_dim)
    torch_embedding = nn.Embedding(n_words, h_dim)
    
    # Copy weights from custom to PyTorch for fair comparison
    with torch.no_grad():
        torch_embedding.weight.copy_(custom_embedding.embeddings)
    
    # Create random input indices
    input_indices = torch.randint(0, n_words, (batch_size, seq_len))
    input_indices_clone = input_indices.clone()
    
    # Forward pass for both implementations
    with torch.no_grad():
        custom_output = custom_embedding(input_indices)
    
    torch_output = torch_embedding(input_indices_clone)
    
    # Create a random gradient for backward pass
    grad_output = torch.randn_like(custom_output)
    grad_output_clone = grad_output.clone()
    
    # Backward pass for PyTorch implementation
    torch_output.backward(grad_output_clone)
    
    # Manual backward pass for custom implementation
    with torch.no_grad():
        custom_embedding.man_backward(grad_output)
    
    # Compare forward results
    forward_diff = torch.abs(custom_output - torch_output).max().item()
    print(f"Max forward difference: {forward_diff:.6f}")
    
    # Compare gradients for embeddings
    embedding_grad_diff = torch.abs(custom_embedding.embeddings.grad - torch_embedding.weight.grad).max().item()
    print(f"Max embedding gradient difference: {embedding_grad_diff:.6f}")
    
    # Check if implementations match within tolerance
    forward_match = forward_diff < 1e-5
    embedding_grad_match = embedding_grad_diff < 1e-5
    
    if forward_match and embedding_grad_match:
        print("✓ Test passed! Custom implementation matches PyTorch.")
    else:
        print("✗ Test failed! Implementations don't match.")
        if not forward_match:
            print("  - Forward pass outputs differ")
        if not embedding_grad_match:
            print("  - Embedding gradients differ")
    
    return forward_match and embedding_grad_match


if __name__ == "__main__":
    test_embedding_layer()