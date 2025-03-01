import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import numpy as np
import time
import os
import math
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_

# Keep your original model implementation
# (PositionEncoding, FeedForwardNN, DecoderTransformerBlock, Transformer classes)

# 1. Data Loading and Processing
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        self.max_length = max_length
        
        for text in tqdm(texts, desc="Tokenizing texts"):
            # Tokenize the text and add special tokens
            tokens = self.tokenizer(text, return_tensors="pt").input_ids[0]
            
            # For each possible starting position
            for i in range(0, len(tokens) - max_length):
                input_sequence = tokens[i:i+max_length]
                target_sequence = tokens[i+1:i+max_length+1]
                
                if len(input_sequence) == max_length and len(target_sequence) == max_length:
                    self.input_ids.append(input_sequence)
                    self.target_ids.append(target_sequence)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# 2. Cosine Learning Rate Scheduler with Warmup
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, max_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr_scale = self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * lr_scale

# 3. Training Function
def train_transformer(model, train_loader, val_loader, optimizer, scheduler, device, 
                      epochs=3, log_interval=100, save_interval=500, grad_accumulation_steps=1,
                      max_grad_norm=1.0, checkpoint_dir='checkpoints'):
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training and validation losses
    train_losses = []
    val_losses = []
    
    model.to(device)
    model.train()
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0
        batch_count = 0
        
        # Training loop
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            _, loss = model(data, targets)
            
            # Scale loss for gradient accumulation
            loss = loss / grad_accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * grad_accumulation_steps
            batch_count += 1
            
            # Gradient accumulation
            if (batch_idx + 1) % grad_accumulation_steps == 0:
                # Gradient clipping
                clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                global_step += 1
                
                # Log training progress
                if global_step % log_interval == 0:
                    avg_loss = total_loss / batch_count
                    train_losses.append((global_step, avg_loss))
                    print(f"Step {global_step} | Train Loss: {avg_loss:.4f}")
                    total_loss = 0
                    batch_count = 0
                
                # Checkpoint model
                if global_step % save_interval == 0:
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.__dict__,
                        'train_losses': train_losses,
                        'val_losses': val_losses
                    }, f"{checkpoint_dir}/model_step_{global_step}.pt")
                    
                    # Validation
                    val_loss = evaluate(model, val_loader, device)
                    val_losses.append((global_step, val_loss))
                    
                    print(f"Step {global_step} | Validation Loss: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.2f}")
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.__dict__,
                            'train_losses': train_losses,
                            'val_losses': val_losses,
                            'val_loss': val_loss,
                            'perplexity': math.exp(val_loss)
                        }, f"{checkpoint_dir}/best_model.pt")
        
        # End of epoch validation
        val_loss = evaluate(model, val_loader, device)
        val_losses.append((global_step, val_loss))
        
        print(f"Epoch {epoch+1}/{epochs} completed in {(time.time() - epoch_start_time):.2f}s")
        print(f"Validation Loss: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.2f}")
    
    # Final checkpoint
    torch.save({
        'epoch': epochs,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.__dict__,
        'train_losses': train_losses,
        'val_losses': val_losses
    }, f"{checkpoint_dir}/final_model.pt")
    
    return train_losses, val_losses

# 4. Evaluation Function
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, targets in tqdm(val_loader, desc="Evaluating"):
            data, targets = data.to(device), targets.to(device)
            _, loss = model(data, targets)
            total_loss += loss.item()
    
    model.train()
    return total_loss / len(val_loader)

# 5. Text Generation Functions
def greedy_decode(model, context, max_new_tokens, tokenizer, device):
    """Greedy decoding: always selects the most likely next token."""
    model.eval()
    context = context.to(device)
    generated = context.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits for the next token
            logits, _ = model(generated[:, -min(generated.size(1), 128):])
            next_token_logits = logits[0, -1, :]
            
            # Get the most likely token (greedy)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Add the token to our generated sequence
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    
    return generated[0]

def top_k_decode(model, context, max_new_tokens, tokenizer, device, k=50, temperature=1.0):
    """Top-k sampling: sample from the k most likely tokens."""
    model.eval()
    context = context.to(device)
    generated = context.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits for the next token
            logits, _ = model(generated[:, -min(generated.size(1), 128):])
            next_token_logits = logits[0, -1, :] / temperature
            
            # Filter to top k tokens
            top_k_logits, top_k_indices = torch.topk(next_token_logits, k, dim=-1)
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(top_k_logits, dim=-1)
            
            # Sample from the probability distribution
            next_token_idx = torch.multinomial(probabilities, num_samples=1)
            next_token = top_k_indices[next_token_idx]
            
            # Add the token to our generated sequence
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    
    return generated[0]

def nucleus_decode(model, context, max_new_tokens, tokenizer, device, p=0.9, temperature=1.0):
    """Nucleus (top-p) sampling: sample from tokens with cumulative probability p."""
    model.eval()
    context = context.to(device)
    generated = context.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits for the next token
            logits, _ = model(generated[:, -min(generated.size(1), 128):])
            next_token_logits = logits[0, -1, :] / temperature
            
            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            
            # Calculate cumulative probabilities
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > p
            # Shift the indices to the right to keep the first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Create a mask for the indices to keep
            indices_to_keep = ~sorted_indices_to_remove
            
            # Filter the probabilities and indices
            filtered_logits = sorted_logits[indices_to_keep]
            filtered_indices = sorted_indices[indices_to_keep]
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(filtered_logits, dim=-1)
            
            # Sample from the probability distribution
            next_token_idx = torch.multinomial(probabilities, num_samples=1)
            next_token = filtered_indices[next_token_idx]
            
            # Add the token to our generated sequence
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    
    return generated[0]

# 6. Plot Training Progress
def plot_losses(train_losses, val_losses):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    
    train_steps, train_loss_values = zip(*train_losses)
    plt.plot(train_steps, train_loss_values, label='Train Loss')
    
    if val_losses:
        val_steps, val_loss_values = zip(*val_losses)
        plt.plot(val_steps, val_loss_values, label='Validation Loss')
    
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()

# 7. Main function to run the entire pipeline
def main():
    # Hyperparameters
    max_length = 128
    batch_size = 16
    learning_rate = 3e-4
    weight_decay = 0.01
    epochs = 3
    warmup_steps = 1000
    grad_accumulation_steps = 1  # Increase if needed for memory constraints
    
    # Load Wikitext-2 dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Initialize tokenizer (GPT-2 tokenizer is compatible with our model)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token
    
    # Create datasets
    train_dataset = TextDataset(dataset["train"]["text"], tokenizer, max_length)
    val_dataset = TextDataset(dataset["validation"]["text"], tokenizer, max_length)
    test_dataset = TextDataset(dataset["test"]["text"], tokenizer, max_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model hyperparameters
    d_model = 512  # Embedding dimension
    num_heads = 8  # Number of attention heads
    layers = 6  # Number of transformer blocks
    vocab_size = len(tokenizer)  # Vocabulary size
    dropout = 0.1  # Dropout probability
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(d_model, num_heads, max_length, vocab_size, layers, bias=False, dropout=dropout)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Initialize learning rate scheduler
    max_steps = len(train_loader) * epochs // grad_accumulation_steps
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, max_steps)
    
    # Train model
    print(f"Starting training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_losses, val_losses = train_transformer(
        model, train_loader, val_loader, optimizer, scheduler, device,
        epochs=epochs, grad_accumulation_steps=grad_accumulation_steps
    )
    
    # Plot training progress
    plot_losses(train_losses, val_losses)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss = evaluate(model, test_loader, device)
    test_perplexity = math.exp(test_loss)
    print(f"Test Loss: {test_loss:.4f} | Perplexity: {test_perplexity:.2f}")
    
    # Compare with GPT-2 baseline
    print("Comparing with GPT-2 baseline...")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_model.to(device)
    
    # Generate text samples with different decoding strategies
    print("Generating text samples...")
    prompt = "The history of artificial intelligence"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Greedy decoding
    greedy_output = greedy_decode(model, input_ids, max_new_tokens=50, tokenizer=tokenizer, device=device)
    greedy_text = tokenizer.decode(greedy_output, skip_special_tokens=True)
    print("\nGreedy Decoding:")
    print(greedy_text)
    
    # Top-k sampling
    topk_output = top_k_decode(model, input_ids, max_new_tokens=50, tokenizer=tokenizer, device=device, k=50)
    topk_text = tokenizer.decode(topk_output, skip_special_tokens=True)
    print("\nTop-k Sampling (k=50):")
    print(topk_text)
    
    # Nucleus sampling
    nucleus_output = nucleus_decode(model, input_ids, max_new_tokens=50, tokenizer=tokenizer, device=device, p=0.9)
    nucleus_text = tokenizer.decode(nucleus_output, skip_special_tokens=True)
    print("\nNucleus Sampling (p=0.9):")
    print(nucleus_text)

if __name__ == "__main__":
    main()