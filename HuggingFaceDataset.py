from datasets import load_dataset

class HuggingFaceDataset:
    """Adaptee: A class that loads datasets from Hugging Face."""
    
    def load_hf_dataset(self, dataset_name, split="train"):
        dataset = load_dataset("wikitext",dataset_name, split=split)
        return dataset["text"]  # Assuming we need the 'text' field
