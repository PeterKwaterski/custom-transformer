from torch import nn
import torch
from config import Config

class FFN(nn.Module):
    def __init__(self, config: Config, input_size: int, hidden_sizes: list[int]):
        super().__init__()
        self.config = config
        self.w_1 = nn.Linear(config.model_dim, config.model_dim)
        self.w_2 = nn.Linear(config.model_dim, config.model_dim)
        self.dropout = nn.Dropout(config.dropout)
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, config.model_dim))
        self.ffn = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.ffn(x)