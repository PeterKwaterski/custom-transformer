from config import Config
from data_generator import generate_data
import torch

class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, training_range, training_size):
        super().__init__()
        self.training_data, self.training_labels = generate_data(training_range, training_size=training_size)

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, index):
        return self.training_data[index], self.training_labels[index]

class TestingDataset(torch.utils.data.Dataset):
    def __init__(self, testing_range, testing_size):
        super().__init__()
        self.testing_data, self.testing_labels = generate_data(testing_range, testing_size=testing_size)

    def __len__(self):
        return len(self.testing_data)

    def __getitem__(self, index):
        return self.testing_data[index], self.testing_labels[index]
