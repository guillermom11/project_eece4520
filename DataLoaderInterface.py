from abc import ABC, abstractmethod

class DataLoaderInterface(ABC):
    """Target: Defines the interface expected by the system."""
    
    @abstractmethod
    def load_data(self):
        """Loads data and returns it in a compatible format."""
        pass
