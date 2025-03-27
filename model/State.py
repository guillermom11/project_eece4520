from abc import ABC, abstractmethod

# State Interface
class State(ABC):
    @abstractmethod
    def handle(self, context):
        pass
