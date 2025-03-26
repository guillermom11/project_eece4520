from CustomDataLoader import CustomDataLoader
from DataLoaderInterface import DataLoaderInterface
from HuggingFaceDataset import HuggingFaceDataset

class HuggingFaceDatasetAdapter(DataLoaderInterface):
    """Adapter: Converts Hugging Face dataset into a format compatible with CustomDataLoader."""

    def __init__(self, dataset_name, split="train"):
        self.hf_dataset = HuggingFaceDataset()
        self.dataset_name = dataset_name
        self.split = split

    def load_data(self):
        """Loads and prepares data for CustomDataLoader."""
        raw_texts = self.hf_dataset.load_hf_dataset(self.dataset_name, self.split)
        return CustomDataLoader.prepare_bpe_data(raw_texts)  # Convert into tokenized format
