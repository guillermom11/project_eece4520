from datasets import load_dataset
from torch.utils.data import Dataset

class HuggingFaceDatasetAdapter(Dataset):
    """
    Adapter to convert Hugging Face datasets into a format compatible with CustomDataLoader.
    """
    def __init__(self, dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", split="train"):
        """
        Initializes the adapter by loading the dataset.
        
        Arguments:
        - dataset_name (str): Name of the dataset on Hugging Face.
        - dataset_config (str): Specific configuration/version of the dataset.
        - split (str): Data split ('train', 'validation', or 'test').
        """
        self.dataset = load_dataset(dataset_name, dataset_config)[split]
        self.text_data = self.dataset["text"]

    def __len__(self):
        """Returns the number of data samples."""
        return len(self.text_data)

    def __getitem__(self, idx):
        """Returns a single text sample."""
        return self.text_data[idx]

    def get_texts(self):
        """Returns all text samples as a list."""
        return self.text_data
