import torch
import pytest

class DummyTransformer(torch.nn.Module):
    def __init__(self, vocab_size=50):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, self.vocab_size)
        loss = torch.tensor(0.5)
        return logits, loss

@pytest.fixture
def dummy_input():
    return torch.randint(0, 50, (4, 10))

def test_model_output_shape(dummy_input):
    model = DummyTransformer(vocab_size=50)
    logits, loss = model(dummy_input)
    assert logits.shape == (4, 10, model.vocab_size)
    assert isinstance(loss, torch.Tensor)

def test_model_accepts_different_batch_sizes():
    model = DummyTransformer()
    for batch_size in [1, 2, 8]:
        inputs = torch.randint(0, 50, (batch_size, 12))
        logits, _ = model(inputs)
        assert logits.shape[0] == batch_size

def test_model_handles_zero_length_sequence():
    model = DummyTransformer()
    inputs = torch.empty((2, 0), dtype=torch.long)
    logits, _ = model(inputs)
    assert logits.shape[1] == 0
