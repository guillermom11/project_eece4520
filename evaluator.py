import torch
import math
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
    
    def calculate_perplexity(self):
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for inputs, targets in tqdm(self.dataloader, desc="Calculating perplexity"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                logits, loss = self.model(inputs, targets)
                non_pad_mask = targets != -1
                num_tokens = non_pad_mask.sum().item()

                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        return perplexity