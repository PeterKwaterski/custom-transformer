from config import Config
from data.dataloader import TrainingDataLoader
from transformer.decoder import Decoder
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

class Trainer:
    def __init__(self, config: Config):
        self.config = config

    def train(self):
        for epoch in range(self.config.epochs):
            for i in range(len(TrainingDataLoader(self.config.training_range, self.config.training_size, self.config.batch_size))):
                X = TrainingDataLoader[i]['input_ids']
                y = TrainingDataLoader[i]['labels']
                padding_mask = TrainingDataLoader[i]['padding_mask']
                y_hat = Decoder(X, padding_mask=padding_mask)
                loss = CrossEntropyLoss()(y_hat, y)
                loss.backward()
                AdamW(Decoder.parameters(), lr=self.config.learning_rate).step()
                print(f"Epoch {epoch}, Batch {i}, Loss {loss.item()}")