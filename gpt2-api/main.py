import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from generation import greedy_decode
from model_loader import load_model
from tokenizer import BPE  # Use your actual tokenizer
import torch
from logging_config import setup_logging
import time
from model.ConcreteTransformerBuilder import ConcreteTransformerBuilder
from ModelConfig import ModelConfig

setup_logging()
import logging
config = ModelConfig()
app = FastAPI()

tokenizer = BPE()
tokenizer.load_vocab(
    "tokenizer/"
)

# Load model + tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(tokenizer.token_to_id)

builder = ConcreteTransformerBuilder()
model = (
    builder
    .set_embedding_size(config.d_model)
    .set_num_heads(config.num_heads)
    .set_max_length(config.max_length)
    .set_vocab_size(vocab_size)
    .set_layers(config.layers)
    .set_bias(True)
    .set_dropout(config.dropout)
    .build()
).to(device)

model = load_model(model, "../model/best_model.pt", device)

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100 #change this?

@app.post("/generate")
async def generate_text(req: GenerationRequest, request: Request):
    start_time = time.time()
    try:
        if not req.prompt.strip():
            raise ValueError("Empty prompt.")

        output = greedy_decode(model, tokenizer, req.prompt, max_length=req.max_length, device=device)

        logging.info(f"[{request.client.host}] Prompt: {req.prompt} | Time: {time.time() - start_time:.2f}s")
        return {"prompt": req.prompt, "response": output}

    except Exception as e:
        logging.error(f"Error during generation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
