from torch import nn
import torch
from config import Config

class FFN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.w_1 = nn.Linear(config.model_dim, config.model_dim)
        self.w_2 = nn.Linear(config.model_dim, config.model_dim)
        self.dropout = nn.Dropout(config.dropout)

        def _init_weights(self):
            nn.init.normal_(self.w_1.weight, std=0.02)
            nn.init.normal_(self.w_2.weight, std=0.02)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            pass