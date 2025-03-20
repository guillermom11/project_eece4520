from datasets import load_dataset
from torch.utils.data import DataLoader

class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True,collate_fn=None):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,collate_fn=collate_fn)

    @staticmethod
    def load_and_prepare_data():
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_texts = dataset["train"]["text"]
        valid_texts = dataset["validation"]["text"]
        test_texts = dataset["test"]["text"]
        return train_texts, valid_texts, test_texts

    @staticmethod
    def prepare_bpe_data(train_texts, limit=10000):
        return [list(word.lower()) for text in train_texts[:limit] for word in text.split() if word]