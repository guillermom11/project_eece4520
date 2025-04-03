from model.ConcreteStates import TrainingState
from model.State import State
class ModelContext:
    def __init__(self, config, model, trainer, test_loader, tokenizer, device):
        self.config = config
        self.model = model
        self.trainer = trainer
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.device = device
        self._state = TrainingState()
    
    def set_state(self, state: State):
        self._state = state
    
    def execute(self):
        self._state.handle(self)
