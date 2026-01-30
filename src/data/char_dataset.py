"""Character-level dataset for TinyStories / any HuggingFace text dataset."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CharTokenizer:
    """Simple character-level tokenizer (byte-level, 256 vocab)."""

    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")


class CharDataset(Dataset):
    """Memory-mapped character-level dataset."""

    def __init__(self, data_path: str, seq_len: int):
        self.seq_len = seq_len
        self.data = np.memmap(data_path, dtype=np.uint8, mode="r")

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        chunk = torch.from_numpy(
            self.data[start : start + self.seq_len + 1].astype(np.int64)
        )
        return chunk[:-1], chunk[1:]


def create_char_dataloaders(
    train_path: str,
    val_path: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    train_ds = CharDataset(train_path, seq_len)
    val_ds = CharDataset(val_path, seq_len)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    return train_loader, val_loader
