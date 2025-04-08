from model.State import State
import torch
import os
from evaluator import Evaluator
from generator import TextGenerator
from utils import Utils

class TrainingState(State):
    def handle(self, context):
        print("Training model...")
        train_losses, val_losses, train_steps, val_steps = context.trainer.train()
        context.train_losses = train_losses
        context.val_losses = val_losses
        context.train_steps = train_steps
        context.val_steps = val_steps
        context.set_state(ValidationState())

class ValidationState(State):
    def handle(self, context):
        print("Validating model...")
        checkpoint_dir = "./checkpoints"
        checkpoint = torch.load(os.path.join(context.config.checkpoint_dir, 'best_model.pt'))
        context.model.load_state_dict(checkpoint['model_state_dict'])
        context.set_state(TestingState())

class TestingState(State):
    def handle(self, context):
        print("Testing model...")
        evaluator = Evaluator(context.model, context.test_loader, context.device)
        test_perplexity = evaluator.calculate_perplexity()
        print(f"Test perplexity: {test_perplexity:.2f}")
        context.set_state(TextGenerationState())

class TextGenerationState(State):
    def handle(self, context):
        print("Generating text samples...")
        text_generator = TextGenerator(context.model, context.tokenizer)
        seed_texts = [
        "The president of the United States",
        "In the beginning of the 20th century",
        "Scientists have discovered a new",
        "A dog is a type of",
        "To buy a house in the United States you need",
        "The history of artificial intelligence",
        "When I look at the stars"
        ]
        print("\nGenerating text samples:")
        generation_examples = text_generator.generate_samples(seed_texts, context.device, max_length=50)
        context.generation_examples = generation_examples
        text_generator.display_generated_texts()
        context.set_state(FinalState())

class FinalState(State):
    def handle(self, context):
        print("Packaging materials for submission...")
        submission_dir = Utils.package_materials(
            model=context.model,
            tokenizer=context.tokenizer,
            train_losses=context.train_losses,
            val_losses=context.val_losses,
            train_steps=context.train_steps,
            val_steps=context.val_steps,
            generation_examples=context.generation_examples
        )
        print(f"Please submit the entire '{submission_dir}' folder.")
