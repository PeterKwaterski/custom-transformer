from torch import nn
import torch
from config import Config

class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.w_q = nn.Linear(config.model_dim, config.model_dim)
        self.w_k = nn.Linear(config.model_dim, config.model_dim)
        self.w_v = nn.Linear(config.model_dim, config.model_dim)
        self.w_o = nn.Linear(config.model_dim, config.model_dim)
        self.dropout = nn.Dropout(config.dropout)

        def _init_weights(self):
            nn.init.normal_(self.w_q.weight, std=0.02)
            nn.init.normal_(self.w_k.weight, std=0.02)
            nn.init.normal_(self.w_v.weight, std=0.02)
            nn.init.normal_(self.w_o.weight, std=0.02)

        def _calculate_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
            numerator = q @ k.transpose()
            denominator = torch.sqrt(self.head_dim)
            attention = self._apply_casual_mask(numerator / denominator, padding_mask=padding_mask)
            attention = torch.softmax(attention, dim=-1) @ v
            return attention

        def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
            x.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
            return x.permute(0, 2, 1, 3)

        def _apply_casual_mask(self, attention: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
            seq_len = attention.shape[-1]
            mask = torch.triu(torch.ones(seq_len, seq_len, device=attention.device), diagonal=1).bool()
            attention = attention.masked_fill(mask, float('-inf'))

            if padding_mask is not None:
                padding_mask_expanded = padding_mask.unsqueeze(1)
                attention = attention.masked_fill(~padding_mask_expanded, float('-inf'))
            return attention

        
        def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
            batch_size, seq_len = x.shape
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