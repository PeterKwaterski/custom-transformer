class Config:
    def __init__(self):
        self.vocab_size = 20
        self.model_dim = 1024
        self.max_seq_len = 15
        self.dropout = 0.1
        self.num_heads = 16
        self.head_dim = self.model_dim // self.num_heads
        self.hidden_dim = self.model_dim * 4
        self.num_layers = 1
        self.training_range = (0, 100)
        self.testing_range = (0, 499)
        self.training_size = 10000
        self.testing_size = 1000
        self.pad_token_id = "<pad>"
        self.epochs = 20
        self.learning_rate = 0.001
        self.batch_size = 32