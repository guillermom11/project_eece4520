from abc import ABC, abstractmethod
import torch.nn as nn
from model.DecoderTransformerBlock import DecoderTransformerBlock

class TransformerBuilder(ABC):
    def __init__(self):
        self.transformer = None

    @abstractmethod
    def set_embedding_size(self, d_model):
        pass

    @abstractmethod
    def set_num_heads(self, num_heads):
        pass

    @abstractmethod
    def set_max_length(self, max_length):
        pass

    @abstractmethod
    def set_vocab_size(self, vocab_size):
        pass

    @abstractmethod
    def set_layers(self, layers):
        pass

    @abstractmethod
    def set_bias(self, bias):
        pass

    @abstractmethod
    def set_dropout(self, dropout):
        pass

    @abstractmethod
    def build(self):
        pass
