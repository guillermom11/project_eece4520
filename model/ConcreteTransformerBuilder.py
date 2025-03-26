from model.TransformerBuilder import TransformerBuilder

class ConcreteTransformerBuilder(TransformerBuilder):
    def __init__(self):
        super().__init__()
        self.d_model = None
        self.num_heads = None
        self.max_length = None
        self.vocab_size = None
        self.layers = None
        self.bias = False
        self.dropout = 0.2

    def set_embedding_size(self, d_model):
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

    def build(self):
        from model.Transformer import Transformer  # Import here to avoid circular dependency
        return Transformer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            max_length=self.max_length,
            vocab_size=self.vocab_size,
            layers=self.layers,
            bias=self.bias,
            dropout=self.dropout,
        )
