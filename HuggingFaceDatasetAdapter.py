# HuggingFaceDatasetAdapter.py
from DataLoaderInterface import DataLoaderInterface
from CustomDataLoader import CustomDataLoader
from HuggingFaceDataset import HuggingFaceDataset

class HuggingFaceDatasetAdapter(DataLoaderInterface):
    """Adapter: Makes HuggingFaceDataset work with our expected interface"""
    
    def __init__(self, dataset_name="wikitext-2-raw-v1"):
        self.hf_dataset = HuggingFaceDataset()
        self.dataset_name = dataset_name

    def load_and_prepare_data(self):
        """Loads and prepares all splits of data"""
        train_texts = self.hf_dataset.load_hf_dataset(self.dataset_name, "train")
        valid_texts = self.hf_dataset.load_hf_dataset(self.dataset_name, "validation")
        test_texts = self.hf_dataset.load_hf_dataset(self.dataset_name, "test")
        return train_texts, valid_texts, test_texts

    def prepare_bpe_data(self, texts, limit=10000):
        """Uses CustomDataLoader's method to prepare BPE data"""
        return CustomDataLoader.prepare_bpe_data(texts, limit)

    def get_data_loader(self, dataset, batch_size=32, shuffle=True, collate_fn=None):
        """Creates a CustomDataLoader instance"""
        return CustomDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)