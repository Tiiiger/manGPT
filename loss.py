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
    return d_logits

    