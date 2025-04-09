class ModelConfig:
    """Configuration for the Transformer model and training."""
    
    def __init__(self):
        # Model architecture
        self.d_model = 512
        self.num_heads = 8
        self.max_length = 256
        self.layers = 6
        self.dropout = 0.2
        self.bias = True
        
        # Training hyperparameters
        self.batch_size = 8
        self.num_epochs = 3
        self.learning_rate = 3e-4
        self.weight_decay = 0.01
        self.log_interval = 100
        
        # Tokenizer
        self.num_merges = 64000  # For BPE
        
        # File paths
        self.checkpoint_dir = "./checkpoints"