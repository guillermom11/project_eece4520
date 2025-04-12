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
    
def test_save_and_load_vocab(trained_tokenizer, tmp_path):
    save_dir = tmp_path / "vocab"
    trained_tokenizer.save_vocab(str(save_dir))

    # Reset singleton again to ensure clean load
    from tokenizer import BPE
    BPE._instance = None
    new_tokenizer = BPE()
    new_tokenizer.load_vocab(
        token_to_id_path=str(save_dir / "token_to_id.json"),
        id_to_token_path=str(save_dir / "id_to_token.json")
    )

    # Check exact equality
    assert trained_tokenizer.token_to_id == new_tokenizer.token_to_id
    assert trained_tokenizer.id_to_token == new_tokenizer.id_to_token
