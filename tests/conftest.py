import pytest
import torch
from unittest.mock import MagicMock
from tokenizer import BPE

@pytest.fixture
def trained_tokenizer():
    # Reset the singleton instance manually for testing so that .vocab is not None
    BPE._instance = None
    bpe = BPE()
    dummy_data = ["hello", "world"]
    bpe.train_bpe(dummy_data, num_merges=5)
    print("VOCAAAB",bpe.vocab)
    return bpe

@pytest.fixture
def dummy_model():
    class DummyModel:
        def __init__(self):
            self.vocab_size = 100
        def __call__(self, inputs, targets=None):
            logits = torch.randn(inputs.shape[0], inputs.shape[1], self.vocab_size)
            loss = torch.tensor(0.5)
            return logits, loss
    return DummyModel()

@pytest.fixture
def mock_dataset(monkeypatch):
    def mock_load_hf_dataset(self, name, split="train"):
        return ["Some text.", "Another sentence."]
    
    from HuggingFaceDataset import HuggingFaceDataset
    monkeypatch.setattr(HuggingFaceDataset, "load_hf_dataset", mock_load_hf_dataset)
