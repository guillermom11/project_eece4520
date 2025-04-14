import torch
from unittest.mock import MagicMock
from generator import TextGenerator
import pytest



class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=10):
        super().__init__()
        self.vocab_size = vocab_size
        self.tokenizer = MagicMock()
        self.tokenizer.decode.return_value = "decoded sequence"

    def forward(self, input_ids, targets=None):
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, self.vocab_size)
        logits[:, -1, 2] = 10.0  # Always choose token ID 2
        return logits, torch.tensor(0.0)

def test_greedy_decode_returns_string():
    model = DummyModel()
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2]
    tokenizer.decode.return_value = "decoded sequence"

    generator = TextGenerator(model, tokenizer)
    input_ids = torch.tensor([tokenizer.encode("hello")])
    decoded = generator.greedy_decode(input_ids, max_length=3)

    assert isinstance(decoded, str)
    tokenizer.decode.assert_called_once()

def test_greedy_decode_handles_empty_input():
    model = DummyModel()
    tokenizer = MagicMock()
    tokenizer.encode.return_value = []
    tokenizer.decode.return_value = ""

    input_ids = torch.tensor([[]], dtype=torch.long)

    generator = TextGenerator(model, tokenizer)

    with pytest.raises(IndexError):
        generator.greedy_decode(input_ids, max_length=3)

def test_greedy_decode_respects_max_length():
    model = DummyModel()
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1]
    tokenizer.decode.return_value = "a b c"

    generator = TextGenerator(model, tokenizer)
    input_ids = torch.tensor([tokenizer.encode("x")])
    output = generator.greedy_decode(input_ids, max_length=5)

    tokenizer.decode.assert_called_once()
    assert isinstance(output, str)

def test_greedy_decode_deterministic_output():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vocab_size = 5

        def forward(self, input_ids, targets=None):
            # Always predict token ID 3
            batch, seq = input_ids.shape
            logits = torch.zeros(batch, seq, self.vocab_size)
            logits[:, -1, 3] = 10.0  # Makes argmax select index 3
            return logits, torch.tensor(0.0)

    class DummyTokenizer:
        def encode(self, text):
            return [1]  # Starting token

        def decode(self, token_ids):
            return "start " + " ".join(f"token{i}" for i in token_ids[1:])  # skip input token

    model = DummyModel()
    tokenizer = DummyTokenizer()
    generator = TextGenerator(model, tokenizer)

    input_ids = torch.tensor([tokenizer.encode("start")])
    decoded = generator.greedy_decode(input_ids, max_length=3)

    expected_token_ids = [1, 3, 3, 3]  # 1 original + 3 new tokens
    expected_output = tokenizer.decode(expected_token_ids)

    assert decoded == expected_output

