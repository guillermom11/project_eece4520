import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "gpt2_api")))
from fastapi.testclient import TestClient
from gpt2_api.main import app  # adjust this import path to your FastAPI app
import torch
from unittest.mock import MagicMock
from tokenizer import BPE
from generator import TextGenerator
from gpt2_api.model_loader import load_model
from model.ConcreteTransformerBuilder import ConcreteTransformerBuilder
client = TestClient(app)

def test_generate_endpoint_responds_successfully():
    payload = {
        "prompt": "The robot began to",
        "max_length": 5
    }
    response = client.post("/generate", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "prompt" in data
    assert "response" in data

    assert isinstance(data["response"], str)
    assert len(data["response"].strip()) > 0

def test_generate_rejects_empty_input():
    payload = {
        "prompt": "   ",
        "max_length": 5
    }
    response = client.post("/generate", json=payload)

    assert response.status_code == 400
    assert "detail" in response.json()


def test_end_to_end_generation():
    

    # Load tokenizer
    tokenizer = BPE()
    import os
    tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tokenizer"))
    tokenizer.load_vocab(tokenizer_path)


    # Build model
    builder = ConcreteTransformerBuilder()
    model = (
        builder
        .set_embedding_size(512)
        .set_num_heads(8)
        .set_max_length(256)
        .set_vocab_size(len(tokenizer.token_to_id))
        .set_layers(6)
        .set_bias(True)
        .set_dropout(0.2)
        .build()
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "best_model.pt"))
    model = load_model(model, model_path, device)

    # Run decoding
    generator = TextGenerator(model, tokenizer)
    input_ids = tokenizer.encode("A wise man once said")
    torch_input = torch.tensor([input_ids], dtype=torch.long)
    output = generator.greedy_decode(torch_input, max_length=10)

    assert isinstance(output, str)
    assert len(output) > 0
