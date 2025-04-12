import pytest
import torch
from unittest.mock import MagicMock
from tokenizer import BPE


@pytest.fixture
def trained_tokenizer():
    # Reset singleton before each test
    BPE._instance = None
    tokenizer = BPE()
    dummy_data = ["hello", "world"]
    tokenizer.train_bpe(dummy_data, num_merges=5)
    return tokenizer

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
