from torch import nn
import torch

from config import Config
from layers.transformer_block import TransformerBlock
from layers.embedding import Embedding

class Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = Embedding(config)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.output_projection = nn.Linear(config.model_dim, config.vocab_size)

        def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
            inner_tensor = self.embedding(x)
            for transformer_block in self.transformer_blocks:
                inner_tensor = transformer_block(inner_tensor, padding_mask=padding_mask)
            output = self.output_projection(inner_tensor)
            return output