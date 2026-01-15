from src.config import Config
from src.data.dataloader import TrainingDataLoader
from src.transformer.decoder import Decoder
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.model = Decoder(config)

    def train(self):
        dataloader = TrainingDataLoader(self.config.training_range, self.config.training_size, self.config.batch_size)
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        
        for epoch in range(self.config.epochs):
            for i, batch in enumerate(dataloader):
                X = batch['input_ids']
                y = batch['labels']
                padding_mask = batch['padding_mask']
                y_hat = self.model(X, padding_mask=padding_mask)
                loss = CrossEntropyLoss()(y_hat.reshape(-1, y_hat.size(-1)), y.reshape(-1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                print(f"Epoch {epoch}, Batch {i}, Loss {loss.item()}")