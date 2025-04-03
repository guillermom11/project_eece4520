import os
import torch
from torch.optim import AdamW
# Import local modules
from ModelConfig import ModelConfig
from HuggingFaceDatasetAdapter import HuggingFaceDatasetAdapter
from tokenizer import BPE
from dataset import TextDataset, collate_fn
from utils import Utils
from trainer import Trainer
from evaluator import Evaluator
from generator import TextGenerator
from model.ConcreteTransformerBuilder import ConcreteTransformerBuilder
from model.TransformerDirector import TransformerDirector
from observer import Subject  
from observer import ProgressLogger, EarlyStopping
from model.ModelContext import ModelContext

def main():
    # Set random seeds for reproducibility
    Utils.set_seeds()
    
    # Load configuration
    config = ModelConfig()
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create adapter instance
    adapter = HuggingFaceDatasetAdapter(dataset_name="wikitext-2-raw-v1")
    
    # Load all data through the adapter
    train_texts, valid_texts, test_texts = adapter.load_and_prepare_data()
    
    # Initialize BPE tokenizer
    tokenizer = BPE()
      
    # Prepare data for BPE training using the adapter
    bpe_data = adapter.prepare_bpe_data(train_texts)
    
    # Train BPE tokenizer
    print("Training BPE tokenizer...")
    tokenizer.train_bpe(bpe_data, config.num_merges)
    vocab_size = len(tokenizer.token_to_id)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, config.max_length)
    valid_dataset = TextDataset(valid_texts, tokenizer, config.max_length)
    test_dataset = TextDataset(test_texts, tokenizer, config.max_length)
    
    # Create dataloaders using the adapter
    train_loader = adapter.get_data_loader(train_dataset, batch_size=config.batch_size, 
                                         shuffle=True, collate_fn=collate_fn)
    valid_loader = adapter.get_data_loader(valid_dataset, batch_size=config.batch_size, 
                                         shuffle=False, collate_fn=collate_fn)
    test_loader = adapter.get_data_loader(test_dataset, batch_size=config.batch_size, 
                                        shuffle=False, collate_fn=collate_fn)
    
    
    builder = ConcreteTransformerBuilder()
    model = (
        builder
        .set_embedding_size(256)
        .set_num_heads(8)
        .set_max_length(512)
        .set_vocab_size(20000)
        .set_layers(6)
        .set_bias(True)
        .set_dropout(0.15)
        .build()
    )
    
    ## Using Director for a Predefined Model
    #director = TransformerDirector(ConcreteTransformerBuilder())
    #small_transformer = director.construct_small_transformer()
    #large_transformer = director.construct_large_transformer()
    
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

    # Add observers to trainer
    trainer.add_observer(ProgressLogger())
    trainer.add_observer(EarlyStopping(patience=3))

    context = ModelContext(config, model, trainer, test_loader, tokenizer, device)
    for _ in range(5):  # Execute state transitions
        context.execute()
    
    
if __name__ == "__main__":
    main()

    
