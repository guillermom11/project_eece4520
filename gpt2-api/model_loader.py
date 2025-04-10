import torch

def load_model(model_class, path, device='cpu'):
    model = model_class()  # Replace with actual model init if needed
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model