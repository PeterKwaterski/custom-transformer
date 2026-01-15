import torch


from src.data.dataset import TrainingDataset, TestingDataset
from src.utils import tokenize
from src.config import Config

# Build vocabulary from all possible characters
def build_vocab():
    """Build vocabulary mapping characters to integer IDs."""
    # All possible characters in the data: digits, operators, parentheses, space
    chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
             '+', '-', '*', '/', '(', ')', ' ', '<pad>', '=']
    # Create mapping: char -> id (starting from 1, 0 reserved for padding)
    vocab = {char: idx + 1 for idx, char in enumerate(chars)}
    # Add padding token
    vocab['<pad>'] = 0
    return vocab

VOCAB = build_vocab()
PAD_TOKEN_ID = 0

def tokens_to_ids(tokens):
    """Convert list of token strings to list of integer IDs."""
    return [VOCAB.get(token, PAD_TOKEN_ID) for token in tokens]

class TrainingDataLoader(torch.utils.data.DataLoader):
    def __init__(self, value_range, size, batch_size, shuffle=True):
        dataset = TrainingDataset(value_range=value_range, size=size)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        config = Config()
        inputs = [tokenize(item[0]) for item in batch]
        labels = [tokenize(item[1]) for item in batch]

        input_ids_list = []
        for seq in inputs:
            ids = tokens_to_ids(seq)
            padded = ids + [PAD_TOKEN_ID] * (config.max_seq_len - len(ids))
            input_ids_list.append(padded[:config.max_seq_len])
        
        label_ids_list = []
        for label_seq in labels:
            ids = tokens_to_ids(label_seq)
            padded = ids + [PAD_TOKEN_ID] * (config.max_seq_len - len(ids))
            label_ids_list.append(padded[:config.max_seq_len])

        input_tensor = torch.tensor(input_ids_list, dtype=torch.long)
        label_tensor = torch.tensor(label_ids_list, dtype=torch.long)

        padding_mask = (input_tensor != PAD_TOKEN_ID)
        return {'input_ids': input_tensor, 'labels': label_tensor, 'padding_mask': padding_mask}

class TestingDataLoader(torch.utils.data.DataLoader):
    def __init__(self, testing_range, testing_size, batch_size, shuffle=True):
        dataset = TestingDataset(value_range=testing_range, size=testing_size)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        config = Config()
        inputs = [tokenize(item[0]) for item in batch]
        labels = [tokenize(item[1]) for item in batch]

        input_ids_list = []
        for seq in inputs:
            ids = tokens_to_ids(seq)
            padded = ids + [PAD_TOKEN_ID] * (config.max_seq_len - len(ids))
            input_ids_list.append(padded[:config.max_seq_len])
        
        label_ids_list = []
        for label_seq in labels:
            ids = tokens_to_ids(label_seq)
            padded = ids + [PAD_TOKEN_ID] * (config.max_seq_len - len(ids))
            label_ids_list.append(padded[:config.max_seq_len])

        input_tensor = torch.tensor(input_ids_list, dtype=torch.long)
        label_tensor = torch.tensor(label_ids_list, dtype=torch.long)

        padding_mask = (input_tensor != PAD_TOKEN_ID)
        return {'input_ids': input_tensor, 'labels': label_tensor, 'padding_mask': padding_mask}