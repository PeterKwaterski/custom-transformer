from src.config import Config
from torch import nn
import torch

class Embedding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        # Config and hyperparameters
        self.config = config
        self.input_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.model_dim)
        self.layer_norm = nn.LayerNorm(config.model_dim)
        self.dropout = nn.Dropout(config.dropout)
        nn.init.normal_(self.input_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]
        device = x.device
        _input_embedding = self.input_embedding(x)
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        _position_embedding = self.position_embedding(pos)
        embeddings = _input_embedding + _position_embedding

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings