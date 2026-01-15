from torch import nn
import torch

from src.config import Config
from src.layers.attention import Attention
from src.layers.ffn import FFN

class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attention = Attention(config)
        self.ffn = FFN(config, config.model_dim, [config.hidden_dim])
        self.layer_norm_1 = nn.LayerNorm(config.model_dim)
        self.dropout_1 = nn.Dropout(config.dropout)
        self.layer_norm_2 = nn.LayerNorm(config.model_dim)
        self.dropout_2 = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        attention = self.attention(x, padding_mask=padding_mask)
        attention = x + self.layer_norm_1(attention)
        attention = self.dropout_1(attention)
        ffn = self.ffn(attention)
        ffn = attention + self.layer_norm_2(ffn)
        output = self.dropout_2(ffn)
        return output
