import os
import math
import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import json
import pickle

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Import your Transformer model
from torch import nn
import torch.nn.functional as F

class FeedForwardNN(nn.Module):
    def __init__(self, d_model, bias=False, dropout=0.2):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class DecoderTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, max_length, bias=False, dropout=0.2):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, bias=bias)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffnn = FeedForwardNN(d_model, bias, dropout)

    def forward(self, x, padding_mask=None):
        bs, l, h = x.shape
        mask = torch.triu(torch.ones(l, l, device=x.device), 1).bool()
        norm_x = self.ln_1(x)
        x = x + self.attn(norm_x.transpose(0, 1), norm_x.transpose(0, 1), norm_x.transpose(0, 1),
                         attn_mask=mask, key_padding_mask=padding_mask)[0].transpose(0, 1)
        norm_x = self.ln_2(x)
        x = x + self.ffnn(norm_x)
        return x

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, max_length, vocab_size, layers, bias=False, dropout=0.2):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, d_model),
            wpe=nn.Embedding(max_length, d_model),
            drop=nn.Dropout(dropout),
            blocks=nn.ModuleList([DecoderTransformerBlock(d_model, num_heads, max_length, bias, dropout) for _ in range(layers)]),
            ln_f=nn.LayerNorm(d_model),
            head=nn.Linear(d_model, vocab_size, bias=bias),
        ))

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos).unsqueeze(0).expand_as(tok_emb)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.transformer.head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            logits = self.transformer.head(x[:, [-1], :])
            loss = None

        return logits, loss

class BPE:
    def __init__(self):
        self.vocab = defaultdict(int)
        self.merges = {}
        self.token_to_id = {}
        self.id_to_token = {}

    def build_vocab(self, data):
        for word in data:
            self.vocab[' '.join(word)] += 1

    def count_pairs(self):
        pairs = defaultdict(int)
        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, best_pair):
        new_vocab = {}
        bigram = ' '.join(best_pair)
        for word in self.vocab:
            new_word = word.replace(bigram, ''.join(best_pair))
            new_vocab[new_word] = self.vocab[word]
        self.vocab = new_vocab

    def train_bpe(self, data, num_merges):
        self.build_vocab(data)
        for i in range(num_merges):
            pairs = self.count_pairs()
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merges[best_pair] = i
            self.merge_vocab(best_pair)

        # Build token-to-id and id-to-token mappings
        all_tokens = set()
        for word in self.vocab.keys():
            for token in word.split():
                all_tokens.add(token)

        # Special tokens
        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        for i, token in enumerate(special_tokens + list(all_tokens)):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

    def tokenize(self, word):
        tokens = list(word)

        while True:
            lowest_merge_index = float('inf')
            lowest_merge = None

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    if self.merges[pair] < lowest_merge_index:
                        lowest_merge_index = self.merges[pair]
                        lowest_merge = pair

            if not lowest_merge:
                break

            for i in range(len(tokens) - 1):
                if (tokens[i], tokens[i + 1]) == lowest_merge:
                    tokens[i:i + 2] = [''.join(tokens[i:i + 2])]
                    break

        return tokens

    def encode(self, text):
        words = text.lower().split()
        tokens = []
        for word in words:
            word_tokens = self.tokenize(word)
            tokens.extend(word_tokens)

        # Convert tokens to IDs
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.token_to_id["<UNK>"])

        return ids

    def decode(self, ids):
        tokens = [self.id_to_token.get(id, "<UNK>") for id in ids]
        text = ''.join(tokens)
        return text

# Text dataset class
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenized_texts = []
        self.max_length = max_length

        for text in tqdm(texts, desc="Tokenizing texts"):
            tokens = tokenizer.encode(text)
            # Truncate to max_length - 1 to leave room for target
            if len(tokens) > max_length - 1:
                tokens = tokens[:max_length - 1]

            self.tokenized_texts.append(tokens)

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        tokens = self.tokenized_texts[idx]

        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor(tokens[1:] + [0], dtype=torch.long)  # Next tokens as targets with padding at end

        return x, y

# Collate function for DataLoader
def collate_fn(batch):
    inputs, targets = zip(*batch)

    # Pad sequences
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=-1)  # -1 will be ignored in loss

    return inputs_padded, targets_padded

# Learning rate scheduler
import torch.optim.lr_scheduler as lr_scheduler

def get_lr_scheduler(optimizer, warmup_steps, max_steps, min_lr=1e-6):
    # Warmup function: Linear increase from 0 to initial LR
    warmup_scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: step / warmup_steps if step < warmup_steps else 1.0
    )

    # Cosine Annealing Scheduler (after warmup)
    cosine_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_steps - warmup_steps,  # Duration of Cosine Decay
        eta_min=min_lr  # Minimum learning rate
    )

    # Combine Warmup and Cosine using SequentialLR
    return lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

# Text generation functions
def greedy_decode(model, input_ids, max_length, tokenizer):
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            logits, _ = model(input_ids)
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

    return tokenizer.decode(input_ids[0].tolist())

def top_k_sampling(model, input_ids, max_length, tokenizer, k=50, temperature=1.0):
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            logits, _ = model(input_ids)
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)

            # Sample from the filtered distribution
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token_id = top_k_indices.gather(-1, next_token_idx)

            input_ids = torch.cat([input_ids, next_token_id], dim=1)

    return tokenizer.decode(input_ids[0].tolist())

def nucleus_sampling(model, input_ids, max_length, tokenizer, p=0.9, temperature=1.0):
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            logits, _ = model(input_ids)
            logits = logits[:, -1, :] / temperature

            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > p
            # Shift the indices to the right to keep the first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token_id], dim=1)

    return tokenizer.decode(input_ids[0].tolist())

# Calculate perplexity
def calculate_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Calculating perplexity"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits, loss = model(inputs, targets)

            # Count non-padding tokens
            non_pad_mask = targets != -1
            num_tokens = non_pad_mask.sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity

def package_materials(model, tokenizer, train_losses, val_losses, train_steps,val_steps, test_perplexity, generation_examples):
    """
    Package all the required materials for submission.

    Args:
        model: The trained transformer model
        tokenizer: The trained BPE tokenizer
        train_losses: List of training losses
        val_losses: List of validation losses
        test_perplexity: The calculated perplexity on test set
        generation_examples: Dictionary of text generation examples
    """
    # Create submission directory
    submission_dir = "./submission_package"
    os.makedirs(submission_dir, exist_ok=True)

    # 1. Save the model weights
    torch.save(model.state_dict(), os.path.join(submission_dir, "transformer_model.pt"))
    print(f"Model weights saved to {os.path.join(submission_dir, 'transformer_model.pt')}")

    # 2. Save the trained BPE tokenizer
    with open(os.path.join(submission_dir, "bpe_tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"BPE tokenizer saved to {os.path.join(submission_dir, 'bpe_tokenizer.pkl')}")

    # 3. Save the vocabulary file
    vocab_data = {
        "token_to_id": tokenizer.token_to_id,
        "id_to_token": {int(k): v for k, v in tokenizer.id_to_token.items()},
        "vocab_size": len(tokenizer.token_to_id)
    }
    with open(os.path.join(submission_dir, "vocabulary.json"), "w") as f:
        json.dump(vocab_data, f, indent=2)
    print(f"Vocabulary file saved to {os.path.join(submission_dir, 'vocabulary.json')}")

    # 4. Generate and save the loss curve plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_losses, label='Training Loss', color='blue', linestyle='-', marker='o')
    plt.plot(val_steps, val_losses, label='Validation Loss', color='red', linestyle='-', marker='x')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(submission_dir, "loss_curve.png"), dpi=300)
    plt.close()
    print(f"Loss curve saved to {os.path.join(submission_dir, 'loss_curve.png')}")

    # 5. Save perplexity results
    with open(os.path.join(submission_dir, "perplexity.txt"), "w") as f:
        f.write(f"Test perplexity: {test_perplexity:.2f}")
    print(f"Perplexity results saved to {os.path.join(submission_dir, 'perplexity.txt')}")

    # 6. Save generated text samples
    with open(os.path.join(submission_dir, "generated_samples.txt"), "w") as f:
        f.write("# Generated Text Samples\n\n")
        for seed_text, results in generation_examples.items():
            f.write(f"## Seed text: '{seed_text}'\n\n")
            f.write(f"- Greedy decoding: '{results['greedy']}'\n")
            f.write(f"- Top-k sampling (k=50): '{results['top_k']}'\n")
            f.write(f"- Nucleus sampling (p=0.9): '{results['nucleus']}'\n\n")
    print(f"Generated text samples saved to {os.path.join(submission_dir, 'generated_samples.txt')}")

    # Create a README file with instructions
    with open(os.path.join(submission_dir, "README.md"), "w") as f:
        f.write("# GPT-like Transformer Model Submission\n\n")
        f.write("This package contains the following materials:\n\n")
        f.write("1. `transformer_model.pt`: Trained model weights\n")
        f.write("2. `bpe_tokenizer.pkl`: Trained BPE tokenizer\n")
        f.write("3. `vocabulary.json`: Vocabulary file with token mappings\n")
        f.write("4. `loss_curve.png`: Training and validation loss curves\n")
        f.write("5. `perplexity.txt`: Perplexity results on the test dataset\n")
        f.write("6. `generated_samples.txt`: Examples of text generated by the model\n\n")
        f.write("## How to Load the Model\n\n")
        f.write("```python\n")
        f.write("import torch\n")
        f.write("import pickle\n\n")
        f.write("# Load the model\n")
        f.write("model = Transformer(d_model=256, num_heads=8, max_length=128, vocab_size=len(vocab_data['token_to_id']), layers=6)\n")
        f.write("model.load_state_dict(torch.load('transformer_model.pt'))\n\n")
        f.write("# Load the tokenizer\n")
        f.write("with open('bpe_tokenizer.pkl', 'rb') as f:\n")
        f.write("    tokenizer = pickle.load(f)\n")
        f.write("```\n")

    print(f"\nAll materials have been packaged in the '{submission_dir}' directory.")
    return submission_dir


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    d_model = 256
    num_heads = 8
    max_length = 128
    layers = 6
    dropout = 0.1
    batch_size = 16
    num_epochs = 3
    learning_rate = 0.0001
    weight_decay = 0.01
    checkpoint_dir = "./checkpoints"
    log_interval = 100
    num_merges = 32000  # For BPE

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load the dataset (Wikitext-2)
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Extract text from dataset
    train_texts = dataset["train"]["text"]
    valid_texts = dataset["validation"]["text"]
    test_texts = dataset["test"]["text"]

    # Filter out empty lines
    train_texts = [text for text in train_texts if text.strip()]
    valid_texts = [text for text in valid_texts if text.strip()]
    test_texts = [text for text in test_texts if text.strip()]

    # Train BPE tokenizer
    print("Training BPE tokenizer...")
    tokenizer = BPE()
    # Prepare data for BPE training (character level)
    bpe_data = []
    for text in train_texts[:10000]:  # Limit to 10000 examples for faster BPE training
        bpe_data.extend([list(word.lower()) for word in text.split() if word])

    tokenizer.train_bpe(bpe_data, num_merges)
    vocab_size = len(tokenizer.token_to_id)
    print(f"Vocabulary size: {vocab_size}")

    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, max_length)
    valid_dataset = TextDataset(valid_texts, tokenizer, max_length)
    test_dataset = TextDataset(test_texts, tokenizer, max_length)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize the model
    model = Transformer(
        d_model=d_model,
        num_heads=num_heads,
        max_length=max_length,
        vocab_size=vocab_size,
        layers=layers,
        dropout=dropout,
        bias=True
    ).to(device)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Calculate total training steps and warmup steps
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup

    # Initialize learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    # Training loop
    best_val_loss = float('inf')
    global_step = 0

    # Initialize lists to store the losses
    train_losses = []
    val_losses = []
    train_steps = []
    val_steps = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0  # Track loss for the whole epoch
        batch_loss = 0  # Track loss for logging intervals
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            logits, loss = model(inputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # Add to batch loss and epoch loss
            current_loss = loss.item()
            batch_loss += current_loss
            epoch_loss += current_loss
            global_step += 1

            # Log training progress
            if batch_idx % log_interval == 0 and batch_idx > 0:
                avg_loss = batch_loss / log_interval
                elapsed = time.time() - start_time
                print(f"| epoch {epoch+1:3d} | {batch_idx:5d}/{len(train_loader):5d} batches | "
                      f"lr {scheduler.get_last_lr()[0]:.6f} | ms/batch {elapsed * 1000 / log_interval:5.2f} | "
                      f"loss {avg_loss:5.2f}")

                # Track the training loss at this step
                train_losses.append(avg_loss)
                train_steps.append(global_step)

                # Reset batch loss and timer
                batch_loss = 0
                start_time = time.time()

            # Evaluate on validation set
            if global_step % (10 * log_interval) == 0:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_inputs, val_targets in valid_loader:
                        val_inputs = val_inputs.to(device)
                        val_targets = val_targets.to(device)

                        _, val_batch_loss = model(val_inputs, val_targets)
                        val_loss += val_batch_loss.item()

                avg_val_loss = val_loss / len(valid_loader)
                print(f"| Validation | loss {avg_val_loss:5.2f}")

                # Track validation loss
                val_losses.append(avg_val_loss)
                val_steps.append(global_step)

                # Save model if validation loss improved
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': best_val_loss,
                    }, os.path.join(checkpoint_dir, 'best_model.pt'))
                    print(f"| Saved best model to {os.path.join(checkpoint_dir, 'best_model.pt')}")

                model.train()

        # Calculate and store average epoch loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"| End of epoch {epoch+1} | Average loss: {avg_epoch_loss:5.2f}")

        # Save checkpoint at the end of epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_epoch_loss,
        }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))

    # After training, plot the loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_losses, label='Training Loss', color='blue', linestyle='-', marker='o')
    plt.plot(val_steps, val_losses, label='Validation Loss', color='red', linestyle='-', marker='x')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(checkpoint_dir, "loss_curve.png"))
    plt.show()

    # Load the best model
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on test set
    test_perplexity = calculate_perplexity(model, test_loader, device)
    print(f"Test perplexity: {test_perplexity:.2f}")

    # Generate text samples with different decoding strategies
    print("\nGenerating text samples:")

    # Seed texts
    seed_texts = [
        "The president of the United States",
        "In the beginning of the 20th century",
        "Scientists have discovered a new",
        "A dog is a type of",
        "To buy a house in the United States you need",
        "The history of artificial intelligence",
        "When I look at the stars"
    ]

    for seed_text in seed_texts:
        print(f"\nSeed text: '{seed_text}'")

        # Tokenize the seed text
        seed_tokens = tokenizer.encode(seed_text)
        seed_tensor = torch.tensor([seed_tokens], dtype=torch.long).to(device)

        # Generate with different strategies
        greedy_text = greedy_decode(model, seed_tensor, max_length=50, tokenizer=tokenizer)
        print(f"Greedy decoding: '{greedy_text}'")

        top_k_text = top_k_sampling(model, seed_tensor, max_length=50, tokenizer=tokenizer, k=50)
        print(f"Top-k sampling (k=50): '{top_k_text}'")

        nucleus_text = nucleus_sampling(model, seed_tensor, max_length=50, tokenizer=tokenizer, p=0.9)
        print(f"Nucleus sampling (p=0.9): '{nucleus_text}'")

    generation_examples = {}

    for seed_text in seed_texts:
        seed_tokens = tokenizer.encode(seed_text)
        seed_tensor = torch.tensor([seed_tokens], dtype=torch.long).to(device)

        greedy_text = greedy_decode(model, seed_tensor, max_length=50, tokenizer=tokenizer)
        top_k_text = top_k_sampling(model, seed_tensor, max_length=50, tokenizer=tokenizer, k=50)
        nucleus_text = nucleus_sampling(model, seed_tensor, max_length=50, tokenizer=tokenizer, p=0.9)

        generation_examples[seed_text] = {
            'greedy': greedy_text,
            'top_k': top_k_text,
            'nucleus': nucleus_text
        }

    # Package all materials
    submission_dir = package_materials(
        model=model,
        tokenizer=tokenizer,
        train_losses=train_losses,
        val_losses=val_losses,
        train_steps=train_steps,
        val_steps=val_steps,
        test_perplexity=test_perplexity,
        generation_examples=generation_examples
    )

    print(f"\nPlease submit the entire '{submission_dir}' folder to your professor.")

if __name__ == "__main__":
    main()