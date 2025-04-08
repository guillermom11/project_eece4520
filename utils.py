import os
import json
import pickle
import matplotlib.pyplot as plt
import torch

class Utils:
    @staticmethod
    def package_materials(model, tokenizer, train_losses, val_losses, train_steps, val_steps, generation_examples):
        submission_dir = "./submission_package"
        os.makedirs(submission_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(submission_dir, "transformer_model.pt"))
        with open(os.path.join(submission_dir, "bpe_tokenizer.pkl"), "wb") as f:
            pickle.dump(tokenizer, f)
        vocab_data = {"token_to_id": tokenizer.token_to_id, "id_to_token": {int(k): v for k, v in tokenizer.id_to_token.items()}, "vocab_size": len(tokenizer.token_to_id)}
        with open(os.path.join(submission_dir, "vocabulary.json"), "w") as f:
            json.dump(vocab_data, f, indent=2)
        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, train_losses, label='Training Loss', color='blue', linestyle='-', marker='o')
        plt.plot(val_steps, val_losses, label='Validation Loss', color='red', linestyle='x')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(submission_dir, "loss_curve.png"), dpi=300)
        #with open(os.path.join(submission_dir, "perplexity.txt"), "w") as f:
            #f.write(f"Test perplexity: {test_perplexity:.2f}")
        with open(os.path.join(submission_dir, "generated_samples.txt"), "w") as f:
            for seed_text, results in generation_examples.items():
                f.write(f"## Seed text: '{seed_text}'\n")
                f.write(f"- Greedy decoding: '{results['greedy']}'\n")
                f.write(f"- Top-k sampling (k=50): '{results['top_k']}'\n")
                f.write(f"- Nucleus sampling (p=0.9): '{results['nucleus']}'\n\n")
        return submission_dir
    
    @staticmethod
    def set_seeds(seed=42):
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)