from torch import nn
import torch

def tokenize(input: str) -> list[str]:
    return list(input)  

def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)

def load_model(model: nn.Module, path: str):
    model.load_state_dict(torch.load(path))
    return model