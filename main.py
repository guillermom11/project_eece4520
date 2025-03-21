import os
import torch
from torch.optim import AdamW
# Import local modules
from config import ModelConfig
from model import Transformer
from tokenizer import BPE
from dataset import TextDataset, collate_fn
from utils import Utils
from dataloader import CustomDataLoader
from trainer import Trainer
from evaluator import Evaluator
from generator import TextGenerator

def main():
    # Set random seeds for reproducibility
    Utils.set_seeds()
    # Load configuration
    config = ModelConfig()
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Load and prepare data
    train_texts, valid_texts, test_texts = CustomDataLoader.load_and_prepare_data()
    
    # Initialize BPE tokenizer
    tokenizer = BPE()
    
    # Prepare data for BPE training
    bpe_data = CustomDataLoader.prepare_bpe_data(train_texts)
    
    # Train BPE tokenizer
    print("Training BPE tokenizer...")
    tokenizer.train_bpe(bpe_data, config.num_merges)
    vocab_size = len(tokenizer.token_to_id)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, config.max_length)
    valid_dataset = TextDataset(valid_texts, tokenizer, config.max_length)
    test_dataset = TextDataset(test_texts, tokenizer, config.max_length)
    
    # Create dataloaders
    train_loader = CustomDataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = CustomDataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = CustomDataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = Transformer(
        d_model=config.d_model,
        num_heads=config.num_heads,
        max_length=config.max_length,
        vocab_size=config.num_merges,  # Adjust vocab_size as needed
        layers=config.layers,
        dropout=config.dropout,
        bias=config.bias
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Calculate total training steps and warmup steps
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=config.num_epochs,
        checkpoint_dir=config.checkpoint_dir,
        warmup_steps=warmup_steps,
        max_steps=total_steps,
        log_interval=config.log_interval
    )

    # Start training
    train_losses, val_losses, train_steps, val_steps = trainer.train()
    
        # Load the best model
    checkpoint_dir = "./checkpoints"
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Initialize evaluator
    evaluator = Evaluator(model, test_loader, device)

    # Evaluate on test set
    test_perplexity = evaluator.calculate_perplexity()
    print(f"Test perplexity: {test_perplexity:.2f}")

    # Initialize text generator
    text_generator = TextGenerator(model, tokenizer)

    # Seed texts for text generation
    seed_texts = [
        "The president of the United States",
        "In the beginning of the 20th century",
        "Scientists have discovered a new",
        "A dog is a type of",
        "To buy a house in the United States you need",
        "The history of artificial intelligence",
        "When I look at the stars"
    ]

    # Generate text samples using different strategies
    print("\nGenerating text samples:")
    generation_examples = text_generator.generate_samples(seed_texts, device, max_length=50)

    # Display generated texts
    text_generator.display_generated_texts()

    # Package all materials for submission
    submission_dir = Utils.package_materials(
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
    