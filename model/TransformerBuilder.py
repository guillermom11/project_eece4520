import torch
from model.Transformer import Transformer

class TransformerBuilder:
    def __init__(self):
        """Initialize with default values."""
        self.d_model = 512
        self.num_heads = 8
        self.max_length = 512
        self.vocab_size = 30522  # Default vocab size for BPE
        self.layers = 6
        self.bias = False
        self.dropout = 0.2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_d_model(self, d_model):
        self.d_model = d_model
        return self

    def set_num_heads(self, num_heads):
        self.num_heads = num_heads
        return self

    def set_max_length(self, max_length):
        self.max_length = max_length
        return self

    def set_vocab_size(self, vocab_size):
        self.vocab_size = vocab_size
        return self

    def set_layers(self, layers):
        self.layers = layers
        return self

    def set_bias(self, bias):
        self.bias = bias
        return self

    def set_dropout(self, dropout):
        self.dropout = dropout
        return self

    def set_device(self, device):
        self.device = device
        return self

    def build(self):
        """Construct and return a Transformer instance with the configured parameters."""
        model = Transformer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            max_length=self.max_length,
            vocab_size=self.vocab_size,
            layers=self.layers,
            bias=self.bias,
            dropout=self.dropout
        ).to(self.device)
        return model
