
from torch.utils.data import DataLoader

class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True,collate_fn=None):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,collate_fn=collate_fn)
    """
    @staticmethod
    def load_and_prepare_data():
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_texts = dataset["train"]["text"]
        valid_texts = dataset["validation"]["text"]
        test_texts = dataset["test"]["text"]
        return train_texts, valid_texts, test_texts"""

    @staticmethod
    def prepare_bpe_data(train_texts, limit=10000):
        #print("THis is my train_texts",train_texts[:limit])
        bpe_data = []
        for text in train_texts[:limit]:
            bpe_data.extend([list(word.lower()) for word in text.split() if word])
        return bpe_data