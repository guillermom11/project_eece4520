class ModelConfig:
    """Configuration for the Transformer model and training."""
    
    def __init__(self):
        # Model architecture
        self.d_model = 256
        self.num_heads = 8
        self.max_length = 128
        self.layers = 6
        self.dropout = 0.1
        self.bias = True
        
        # Training hyperparameters
        self.batch_size = 16
        self.num_epochs = 3
        self.learning_rate = 0.0001
        self.weight_decay = 0.01
        self.log_interval = 100
        
        # Tokenizer
        self.num_merges = 32000  # For BPE
        
        # File paths
        self.checkpoint_dir = "./checkpoints"