import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast

# Enable gradients for training
torch.set_grad_enabled(True)

# Configuration (same as before)
H_DIM = 128
N_HEAD = 8
N_LAYER = 6
INTERMEDIATE_DIM = H_DIM * 4
MAX_LEN = 128
BATCH_SIZE = 32
NUM_EPOCHS = 3
EVAL_EVERY = 100

# Load dataset and tokenizer
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
N_WORDS = len(tokenizer)

# Create custom GPT-2 configuration
config = GPT2Config(
    vocab_size=N_WORDS,
    n_positions=MAX_LEN,
    n_embd=H_DIM,
    n_layer=N_LAYER,
    n_head=N_HEAD,
    n_inner=INTERMEDIATE_DIM,
    activation_function="relu",  # Using ReLU as in your modification
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    use_cache=False
)

# Create model with custom config but without loading pretrained weights
model = GPT2LMHeadModel(config)
model.to("cuda:0")
optimizer = torch.optim.Adam(model.parameters())

# Create training set
train_set = ds["train"].filter(lambda row: row["text"] != "")

# Training loop
def process_batch(batch_texts):
    batch = tokenizer(batch_texts, max_length=MAX_LEN+1, truncation=True, 
                    padding=True, add_special_tokens=True, return_tensors="pt")
    
    input_ids = batch["input_ids"][:, :-1].to("cuda:0")
    attention_mask = batch["attention_mask"][:, :-1].to("cuda:0")
    labels = batch["input_ids"][:, 1:].to("cuda:0")
    
    # Create label tensor with -100 where attention_mask is 0 (padding)
    labels = labels.masked_fill(attention_mask == 0, -100)
    
    # Forward pass with HuggingFace model
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    loss = outputs.loss
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Main training loop
print("Starting training with HuggingFace GPT-2 model")
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    batch_count = 0
    
    # Create batches from the dataset
    for i in range(0, len(train_set), BATCH_SIZE):
        end_idx = min(i + BATCH_SIZE, len(train_set))
        batch_texts = train_set[i:end_idx]["text"]
        
        if len(batch_texts) == 0:
            continue
            
        loss = process_batch(batch_texts)
        total_loss += loss
        batch_count += 1
        
        if batch_count % EVAL_EVERY == 0:
            avg_loss = total_loss / EVAL_EVERY
            print(f"Epoch {epoch+1}, Batch {batch_count}, Avg Loss: {avg_loss:.4f}")
            total_loss = 0
    
    # Print epoch stats
    print(f"Epoch {epoch+1} completed. Processing {batch_count} batches.")

print("Training completed.") 