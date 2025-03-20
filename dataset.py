import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenized_texts = []
        self.max_length = max_length

        for text in tqdm(texts, desc="Tokenizing texts"):
            tokens = tokenizer.encode(text)
            # Truncate to max_length - 1 to leave room for target
            if len(tokens) > max_length - 1:
                tokens = tokens[:max_length - 1]

            self.tokenized_texts.append(tokens)

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        tokens = self.tokenized_texts[idx]

        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor(tokens[1:] + [0], dtype=torch.long)  # Next tokens as targets with padding at end

        return x, y

# Collate function for DataLoader
def collate_fn(batch):
    inputs, targets = zip(*batch)

    # Pad sequences
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=-1)  # -1 will be ignored in loss

    return inputs_padded, targets_padded