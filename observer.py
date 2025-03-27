from abc import ABC, abstractmethod
from typing import Dict, Any, Set

class TrainingObserver(ABC):
    @abstractmethod
    def update(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Observers should return True if training should stop, otherwise False."""
        pass

class Subject:
    def __init__(self):
        self._observers: Set[TrainingObserver] = set()
    
    def add_observer(self, observer: TrainingObserver):
        self._observers.add(observer)
    
    def notify_observers(self, event_type: str, data: Dict[str, Any]) -> bool:
        should_stop = False
        for observer in self._observers:
            if observer.update(event_type, data):
                should_stop = True
        return should_stop

class ProgressLogger(TrainingObserver):
    def update(self, event_type: str, data: Dict[str, Any]) -> bool:
        if event_type == "batch_end":
            print(f"Batch {data['batch_idx']}/{data['total_batches']} | Loss: {data['loss']:.4f} | LR: {data['lr']:.6f}")
        elif event_type == "validation":
            print(f"Validation Loss: {data['val_loss']:.4f}")
        return False  # This observer doesn't stop training

class EarlyStopping(TrainingObserver):
    def __init__(self, patience=3):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
    
    def update(self, event_type: str, data: Dict[str, Any]) -> bool:
        if event_type == "validation":
            if data["val_loss"] < self.best_loss:
                self.best_loss = data["val_loss"]
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print("Early stopping triggered!")
                    return True  # Return True to indicate stopping
        return False
