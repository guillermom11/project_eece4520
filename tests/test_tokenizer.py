import os
from tokenizer import BPE

def test_encode_decode_basic(fresh_tokenizer):
    text = "The quick brown fox"
    encoded = fresh_tokenizer.encode(text)
    decoded = fresh_tokenizer.decode(encoded)
    assert isinstance(encoded, list)
    assert all(isinstance(i, int) for i in encoded)
    assert isinstance(decoded, str)
    assert "quick" in decoded  # Should preserve core structure

def test_empty_input(fresh_tokenizer):
    encoded = fresh_tokenizer.encode("")
    decoded = fresh_tokenizer.decode(encoded)
    assert encoded == []
    assert decoded.strip() == ""

def test_special_characters(fresh_tokenizer):
    text = "!@#%^&*()_+-=[]{}|;':,.<>/?`~"
    encoded = fresh_tokenizer.encode(text)
    decoded = fresh_tokenizer.decode(encoded)
    assert len(encoded) > 0
    assert decoded.strip() != ""

def test_unicode_and_multilingual(fresh_tokenizer):
    inputs = ["こんにちは", "Привет", "مرحبا", "😊🔥"]
    for text in inputs:
        encoded = fresh_tokenizer.encode(text)
        decoded = fresh_tokenizer.decode(encoded)
        assert isinstance(encoded, list)
        assert isinstance(decoded, str)
        assert len(encoded) > 0

def test_long_text(fresh_tokenizer):
    text = "hello " * 1000
    encoded = fresh_tokenizer.encode(text)
    decoded = fresh_tokenizer.decode(encoded)
    assert len(encoded) > 0
    assert "hello" in decoded

def test_tokenizer_saves_and_loads(tmp_path, fresh_tokenizer):
    # Save
    save_dir = tmp_path / "hf_tokenizer"
    fresh_tokenizer.save_vocab(str(save_dir))

    # Reload into a new tokenizer instance
    BPE._instance = None
    tokenizer_reloaded = BPE()
    tokenizer_reloaded.load_vocab(str(save_dir))

    text = "Testing save and load"
    encoded = tokenizer_reloaded.encode(text)
    decoded = tokenizer_reloaded.decode(encoded)

    assert isinstance(encoded, list)
    assert isinstance(decoded, str)
    assert "Testing" in decoded

def test_consistency_after_reload(tmp_path):
    BPE._instance = None
    tokenizer = BPE()
    text = "consistency test"
    original_ids = tokenizer.encode(text)

    save_dir = tmp_path / "hf_tokenizer_consistency"
    tokenizer.save_vocab(str(save_dir))

    # Reload
    BPE._instance = None
    tokenizer2 = BPE()
    tokenizer2.load_vocab(str(save_dir))
    reloaded_ids = tokenizer2.encode(text)

    assert original_ids == reloaded_ids
