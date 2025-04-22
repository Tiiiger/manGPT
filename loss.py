import torch

def softmax_crossentropy(logits, mask, target):
    # logits: BxTxN
    # mask: BxT
    # target: BxT
    log_prob = logits.log_softmax(dim=-1)
    # cross entropy
    loss = -torch.gather(log_prob, -1, target.unsqueeze(-1)).squeeze(-1)
    loss.masked_fill_(mask == 0, 0)

    return loss.sum() / mask.sum(), log_prob.exp()

def softmax_crossentropy_backward(prob, mask, target):
    target_prob = prob.gather(-1, target.unsqueeze(-1)) - 1
    d_logits = prob
    d_logits.scatter_(-1, target.unsqueeze(-1), target_prob)
    d_logits.masked_fill_((mask == 0).unsqueeze(-1), 0)

    d_logits = d_logits / mask.sum()

    return d_logits

def test_softmax_crossentropy():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Generate random test data
    batch_size = 4
    seq_len = 10
    vocab_size = 100
    
    # Create random logits, clone for two separate computations
    logits_custom = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    logits_torch = logits_custom.detach().clone().requires_grad_(True)
    
    # Create random targets
    target = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Generate random sequence lengths and create a mask based on them
    # Each sequence will have length between 1 and seq_len
    lengths = torch.randint(1, seq_len + 1, (batch_size,))
    
    # Create position indices: [0, 1, 2, ..., seq_len-1] for each batch
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # mask: BxT - 1 for valid positions (position < length), 0 for padding
    mask = (positions < lengths.unsqueeze(1)).float()
    
    # Custom implementation - forward pass
    with torch.no_grad():
        custom_loss, prob = softmax_crossentropy(logits_custom, mask, target)
    
    # PyTorch implementation
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    reshaped_logits = logits_torch.reshape(-1, vocab_size)
    reshaped_target = target.reshape(-1)
    reshaped_mask = mask.reshape(-1)
    
    # Calculate loss with PyTorch
    torch_losses = criterion(reshaped_logits, reshaped_target)
    torch_losses = torch_losses * reshaped_mask
    torch_loss = torch_losses.sum() / mask.sum()
    
    # Backward pass for both implementations
    torch_loss.backward()
    
    # For custom implementation, use custom backward function and create fake context
    with torch.no_grad():
        d_logits = softmax_crossentropy_backward(prob, mask, target)
        logits_custom.grad = d_logits
    
    # Compare results
    print(f"Custom loss: {custom_loss.item():.6f}")
    print(f"PyTorch loss: {torch_loss.item():.6f}")
    print(f"Loss difference: {abs(custom_loss.item() - torch_loss.item()):.6f}")
    
    grad_diff = torch.abs(logits_custom.grad - logits_torch.grad).max().item()
    print(f"Max gradient difference: {grad_diff:.6f}")
    
    # Check if the implementations match within tolerance
    loss_match = abs(custom_loss.item() - torch_loss.item()) < 1e-5
    grad_match = grad_diff < 1e-5
    
    if loss_match and grad_match:
        print("✓ Test passed! Custom implementation matches PyTorch.")
    else:
        print("✗ Test failed! Implementations don't match.")
        if not loss_match:
            print("  - Loss values differ")
        if not grad_match:
            print("  - Gradients differ")
    
    return loss_match and grad_match

if __name__ == "__main__":
    test_softmax_crossentropy()

    