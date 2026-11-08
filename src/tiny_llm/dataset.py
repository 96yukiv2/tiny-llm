"""Build shifted input and target sequences for next-character prediction."""
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, tokens: list[int], block_size: int):
        if block_size < 1 or len(tokens) <= block_size:
            raise ValueError("Text must contain more characters than block_size")
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[index : index + self.block_size + 1]
        return chunk[:-1], chunk[1:]
