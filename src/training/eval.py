from src.data.dataloader import TestingDataLoader, PAD_TOKEN_ID
from src.transformer.decoder import Decoder
from src.config import Config
from torch.nn import CrossEntropyLoss
import torch
class Evaluator:
    def __init__(self, model: Decoder, model_path: str):
        self.model = model
        self.model_path = model_path
        self.config = Config()

    def evaluate(self):
        dataloader = TestingDataLoader(self.config.testing_range, self.config.testing_size, self.config.batch_size)
        self.model.load_state_dict(torch.load(self.model_path))
        total_loss = 0
        correct_samples = 0
        total_samples = 0
        for i, batch in enumerate(dataloader):
                X = batch['input_ids']
                y = batch['labels']
                padding_mask = batch['padding_mask']
                y_hat = self.model(X, padding_mask=padding_mask)
                loss = CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)(y_hat.reshape(-1, y_hat.size(-1)), y.reshape(-1))
                total_loss += loss.item()
                predictions = y_hat.argmax(dim=-1)
                correct_mask = (predictions == y) & padding_mask
                correct_samples += correct_mask.sum().item()
                total_samples += padding_mask.sum().item()
        print(f"Accuracy: {correct_samples / total_samples}")
        print(f"Loss: {total_loss / len(dataloader.dataset)}")