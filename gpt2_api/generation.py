import torch

def greedy_decode(model, tokenizer, input_text, max_length=500, device='cpu'):
    input_ids = torch.tensor([tokenizer.encode(input_text)]).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            logits, _ = model(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    return tokenizer.decode(input_ids[0].tolist())
