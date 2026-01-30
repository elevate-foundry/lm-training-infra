"""Dataset and dataloader utilities for language model training."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """Memory-mapped token dataset for efficient large-scale training.

    Expects a flat binary file of uint16 token IDs (created by prepare_data.py).
    """

    def __init__(self, data_path: str, seq_len: int):
        self.seq_len = seq_len
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        chunk = torch.from_numpy(self.data[start : start + self.seq_len + 1].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def create_dataloaders(
    train_path: str,
    val_path: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    train_ds = TextDataset(train_path, seq_len)
    val_ds = TextDataset(val_path, seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, val_loader
