# DataLoaderInterface.py
from abc import ABC, abstractmethod

class DataLoaderInterface(ABC):
    """Target: Defines the interface expected by the system."""
    
    @abstractmethod
    def load_and_prepare_data(self):
        """Should return train, validation, test texts"""
        pass

    @abstractmethod
    def prepare_bpe_data(self, texts, limit=10000):
        """Prepare data for BPE training"""
        pass

    @abstractmethod
    def get_data_loader(self, dataset, batch_size=32, shuffle=True, collate_fn=None):
        """Should return a DataLoader instance"""
        pass