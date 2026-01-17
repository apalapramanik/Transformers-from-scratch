import torch


class CharDataset:
    """
    PURPOSE:
    Character-level dataset for language modeling.

    Given a long text sequence, it produces:
    - input sequence of length T
    - target sequence shifted by 1
    """

    def __init__(self, text, seq_len):
        self.seq_len = seq_len

        # Build vocabulary
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

        # Encode entire text as integers
        self.data = torch.tensor(
            [self.stoi[c] for c in text],
            dtype=torch.long
        )

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y
