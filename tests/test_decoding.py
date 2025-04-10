import torch
from unittest.mock import MagicMock

def test_greedy_decode_returns_expected_string():
    class DummyModel:
        def __init__(self):
            self.vocab_size = 5
            self.tokenizer = MagicMock()
            self.tokenizer.decode.return_value = "decoded sequence"
        
        def eval(self): pass

        def __call__(self, input_ids, targets=None):
            batch, seq = input_ids.shape
            logits = torch.zeros(batch, seq, self.vocab_size)
            logits[:, -1, 2] = 10.0  # Always pick token 2
            return logits, torch.tensor(0.0)

        def greedy_decode(self, input_ids, max_length):
            for _ in range(max_length):
                logits, _ = self(input_ids)
                next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token_id], dim=1)
            return self.tokenizer.decode(input_ids[0].tolist())

    model = DummyModel()
    input_ids = torch.tensor([[1, 1]])
    decoded = model.greedy_decode(input_ids, max_length=3)

    expected_tokens = input_ids[0].tolist() + [2] * 3
    model.tokenizer.decode.assert_called_with(expected_tokens)
