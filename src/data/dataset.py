from src.config import Config
from src.data.data_generator import generate_data
import torch

class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, value_range, size):
        super().__init__()
        self.training_data, self.training_labels = generate_data(value_range, size=size)

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, index):
        return self.training_data[index], self.training_labels[index]
    
    def get_data(self):
        return self.training_data, self.training_labels

class TestingDataset(torch.utils.data.Dataset):
    def __init__(self, value_range, size):
        super().__init__()
        self.testing_data, self.testing_labels = generate_data(value_range, size=size)

    def __len__(self):
        return len(self.testing_data)

    def __getitem__(self, index):
        return self.testing_data[index], self.testing_labels[index]

    def get_data(self):
        return self.testing_data, self.testing_labels
