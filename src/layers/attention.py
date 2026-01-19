from torch import nn
import torch
from src.config import Config

class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.w_q = nn.Linear(config.model_dim, config.model_dim)
        self.w_k = nn.Linear(config.model_dim, config.model_dim)
        self.w_v = nn.Linear(config.model_dim, config.model_dim)
        self.w_o = nn.Linear(config.model_dim, config.model_dim)
        self.dropout = nn.Dropout(config.dropout)
        nn.init.normal_(self.w_q.weight, std=0.02)
        nn.init.normal_(self.w_k.weight, std=0.02)
        nn.init.normal_(self.w_v.weight, std=0.02)
        nn.init.normal_(self.w_o.weight, std=0.02)
        self.num_heads = config.num_heads
        self.head_dim = config.model_dim // config.num_heads

    def _calculate_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        numerator = q @ k.transpose(-2, -1)
        denominator = torch.sqrt(torch.tensor(self.head_dim, dtype=q.dtype, device=q.device))
        attention_scores = numerator / denominator
        attention_scores = self._apply_casual_mask(attention_scores, padding_mask=padding_mask)
        attention = torch.softmax(attention_scores, dim=-1)
        attention = self.dropout(attention)
        attention = attention @ v
        return attention

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, model_dim = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _apply_casual_mask(self, attention_scores: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        seq_len = attention_scores.shape[-1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=attention_scores.device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        attention_scores = attention_scores.masked_fill(causal_mask, float('-inf'))

        if padding_mask is not None:
            padding_mask_q = padding_mask.unsqueeze(1).unsqueeze(-1)
            padding_mask_k = padding_mask.unsqueeze(1).unsqueeze(-2)
            attention_scores = attention_scores.masked_fill(~padding_mask_q, float('-inf'))
            attention_scores = attention_scores.masked_fill(~padding_mask_k, float('-inf'))
        return attention_scores

    
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        split_q = self._split_heads(q)
        split_k = self._split_heads(k)
        split_v = self._split_heads(v)
        
        multi_attention = self._calculate_attention(split_q, split_k, split_v, padding_mask=padding_mask)
        attention = multi_attention.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.config.model_dim)
        attention = self.dropout(attention)
        attention = self.w_o(attention)
        return attention