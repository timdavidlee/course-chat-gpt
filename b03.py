import torch
from typing import Tuple
from torch.utils.data import DataLoader, Dataset


class MyTorchDataset(Dataset):
    def __init__(self, window_size: int, long_seq: torch.LongTensor):
        self.seq = long_seq
        self.window_size = window_size

    def __len__(self):
        return self.seq.shape[0] - self.window_size - 1

    def __getitem__(self, idx) -> Tuple[torch.LongTensor, torch.LongTensor]: 
        x = self.seq[idx: idx + self.window_size]
        y = self.seq[idx + 1 : idx + 1 + self.window_size]
        return x, y