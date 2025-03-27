import State from State.py
class TrainingState(State):
    def handle(self, context):
        print("Model is training...")
        # Add training logic here
        context.set_state(ValidationState())  # Transition to validation

class ValidationState(State):
    def handle(self, context):
        print("Model is validating...")
        # Add validation logic here
        context.set_state(TestingState())  # Transition to testing

class TestingState(State):
    def handle(self, context):
        print("Model is testing...")
        # Add testing logic here
        context.set_state(TrainingState())  # Loop back to training if needed
