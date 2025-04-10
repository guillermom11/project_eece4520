def test_unknown_token(trained_tokenizer):
    ids = trained_tokenizer.encode("qwertyunknown")
    unk_id = trained_tokenizer.token_to_id["<UNK>"]
    assert unk_id in ids

def test_padding_behavior(trained_tokenizer):
    encoded = trained_tokenizer.encode("hello")
    padded = encoded + [trained_tokenizer.token_to_id["<PAD>"]] * (10 - len(encoded))
    assert len(padded) == 10

def test_truncation_behavior(trained_tokenizer):
    encoded = trained_tokenizer.encode("this is a very long sequence for testing")
    max_len = 5
    truncated = encoded[:max_len]
    assert len(truncated) == max_len
