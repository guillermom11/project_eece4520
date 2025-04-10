import torch

def test_model_forward_shape(dummy_model):
    batch_size, seq_len = 4, 10
    inputs = torch.randint(0, 100, (batch_size, seq_len))
    targets = torch.randint(0, 100, (batch_size, seq_len))
    logits, loss = dummy_model(inputs, targets)

    assert logits.shape == (batch_size, seq_len, dummy_model.vocab_size)
    assert isinstance(loss, torch.Tensor)
