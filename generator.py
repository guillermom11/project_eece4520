import torch.nn.functional as F
import torch
import math
from tqdm import tqdm
class TextGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_examples = {}

    def greedy_decode(self, input_ids, max_length):
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_length):
                logits, _ = self.model(input_ids)
                next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token_id], dim=1)
        return self.tokenizer.decode(input_ids[0].tolist())

    def top_k_sampling(self, input_ids, max_length, k=50, temperature=1.0):
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_length):
                logits, _ = self.model(input_ids)
                logits = logits[:, -1, :] / temperature
                top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
                probs = F.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1)
                next_token_id = top_k_indices.gather(-1, next_token_idx)
                input_ids = torch.cat([input_ids, next_token_id], dim=1)
        return self.tokenizer.decode(input_ids[0].tolist())

    def nucleus_sampling(self, input_ids, max_length, p=0.9, temperature=1.0):
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_length):
                logits, _ = self.model(input_ids)
                logits = logits[:, -1, :] / temperature
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
                probs = F.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token_id], dim=1)
        return self.tokenizer.decode(input_ids[0].tolist())

    def generate_samples(self, seed_texts, device, max_length=50):
        
        for seed_text in seed_texts:
            seed_tokens = self.tokenizer.encode(seed_text)
            seed_tensor = torch.tensor([seed_tokens], dtype=torch.long).to(device)
            self.generation_examples[seed_text] = {
                'greedy': self.greedy_decode(seed_tensor, max_length),
                'top_k': self.top_k_sampling(seed_tensor, max_length, k=50),
                'nucleus': self.nucleus_sampling(seed_tensor, max_length, p=0.9)
            }
        return self.generation_examples
    
    def display_generated_texts(self):
        for seed_text, generations in self.generation_examples.items():
            print(f"\nSeed text: '{seed_text}'")
            print(f"Greedy decoding: '{generations['greedy']}'")
            print(f"Top-k sampling (k=50): '{generations['top_k']}'")
            print(f"Nucleus sampling (p=0.9): '{generations['nucleus']}'")