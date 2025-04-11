from transformers import GPT2Tokenizer
import threading
import os
import json

class BPE:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(BPE, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'tokenizer'):
            # Load GPT2 tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.token_to_id = self.tokenizer.get_vocab()
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def encode(self, text):
        """Encodes text to a list of token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids):
        """Decodes a list of token IDs back to text."""
        return self.tokenizer.decode(ids)

    def save_vocab(self, dir_path):
        """Saves tokenizer files to directory."""
        os.makedirs(dir_path, exist_ok=True)
        self.tokenizer.save_pretrained(dir_path)

    def load_vocab(self, dir_path):
        """Loads tokenizer files from directory."""
        self.tokenizer = GPT2Tokenizer.from_pretrained(dir_path)
        self.token_to_id = self.tokenizer.get_vocab()
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
