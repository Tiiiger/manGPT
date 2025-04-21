import torch
import torch.distributed as dist
import os
from model import Transformer
from loss import softmax_crossentropy, softmax_crossentropy_backward
torch.enable_grad(False)

# rank = int(os.environ.get("RANK", -1))
# size = int(os.environ.get("WORLD_SIZE", -1))
# dist.init_process_group(backend="nccl", rank=rank, world_size=size)
# print(f"Rank {rank}, World size {size}")
# torch.manual_seed(0)

# l = Linear(5, 10).to(f"cuda:{rank}")
# X = torch.rand(32, 5, device=f"cuda:{rank}")

# y = l(X)
# print(l.weight.grad)

from datasets import load_dataset
from transformers import GPT2TokenizerFast

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


## CONFIG
H_DIM = 64
N_HEAD = 4
N_LAYER = 6
INTERMEDIATE_DIM = H_DIM * 4
N_WORDS = len(tokenizer)
MAX_LEN=128

## DATA
train_set = ds["train"].filter(lambda row: row["text"] != "")
batch = tokenizer(train_set[:10]["text"], max_length=MAX_LEN+1, truncation=True, padding=True, add_special_tokens=True, return_tensors="pt")


## MODEL
model = Transformer(
    n_layer=N_LAYER,
    n_head=N_HEAD,
    h_dim=H_DIM,
    intermediate_dim=INTERMEDIATE_DIM,
    n_words=N_WORDS,
    max_len=MAX_LEN
)
model.to("cuda:0")

with torch.no_grad():
    input_ids = batch["input_ids"][:, :-1].to("cuda:0")
    mask = batch["attention_mask"][:, :-1].to("cuda:0")
    targets = batch["input_ids"][:, 1:].to("cuda:0")

    logits = model(input_ids, mask)
    loss, prob = softmax_crossentropy(logits, mask, targets)
    d_logits = softmax_crossentropy_backward(prob, mask, targets)
    model.man_backward(d_logits)
    breakpoint()