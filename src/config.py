class Config:
    def __init__(self):
        self.vocab_size = 50000
        self.model_dim = 1024
        self.max_seq_len = 2048
        self.dropout = 0.1
        self.num_heads = 16
        self.head_dim = self.model_dim // self.num_heads
        self.hidden_dim = self.model_dim * 4