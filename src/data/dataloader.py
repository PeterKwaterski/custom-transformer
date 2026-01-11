import torch


from dataset import TrainingDataset, TestingDataset
from utils import tokenize
from config import Config

class TrainingDataLoader(torch.utils.data.DataLoader):
    def __init__(self, training_range, training_size, batch_size, shuffle=True):
        self.training_dataset, self.training_labels = TrainingDataset(training_range, training_size)
        super().__init__(self.training_dataset, batch_size=batch_size, shuffle=shuffle)

    def __len__(self):
        return len(self.training_dataset)
    
    def __getitem__(self, index):
        return tokenize(self.training_dataset[index]), tokenize(self.training_labels[index])

    def collate_fn(self, batch):
        inputs = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        padded_inputs = []
        for seq in inputs:
            padded = seq + [Config.pad_token_id] * (Config.max_seq_len - len(seq))
            padded_inputs.append(padded)

        input_tensor = torch.tensor(padded_inputs)
        label_tensor = torch.tensor(labels)

        padding_mask = (input_tensor != Config.pad_token_id)
        return {'input_ids': input_tensor, 'labels': label_tensor, 'padding_mask': padding_mask}

class TestingDataLoader(torch.utils.data.DataLoader):
    def __init__(self, testing_range, testing_size, batch_size, shuffle=True):
        self.testing_dataset, self.testing_labels = TestingDataset(testing_range, testing_size)
        super().__init__(self.testing_dataset, batch_size=batch_size, shuffle=shuffle)

    def __len__(self):
        return len(self.testing_dataset)
    
    def __getitem__(self, index):
        return tokenize(self.testing_dataset[index]), tokenize(self.testing_labels[index])