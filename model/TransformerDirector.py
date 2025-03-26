from model.TransformerBuilder import TransformerBuilder

class TransformerDirector:
    def __init__(self, builder: TransformerBuilder):
        self.builder = builder

    def construct_small_transformer(self):
        return (
            self.builder
            .set_embedding_size(128)
            .set_num_heads(4)
            .set_max_length(256)
            .set_vocab_size(10000)
            .set_layers(4)
            .set_bias(False)
            .set_dropout(0.1)
            .build()
        )

    def construct_large_transformer(self):
        return (
            self.builder
            .set_embedding_size(512)
            .set_num_heads(8)
            .set_max_length(512)
            .set_vocab_size(50000)
            .set_layers(12)
            .set_bias(True)
            .set_dropout(0.3)
            .build()
        )
