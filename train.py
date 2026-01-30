#!/usr/bin/env python3
"""Main entry point for language model training."""

import argparse
import os
import random

import numpy as np
import torch
import yaml

from src.model import Transformer, TransformerConfig
from src.data import create_dataloaders
from src.training import Trainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(config: dict) -> torch.device:
    device_str = config["system"]["device"]
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def main():
    parser = argparse.ArgumentParser(description="Train a language model from scratch")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory with train.bin and val.bin")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    config = load_config(args.config)

    # Seed everything
    seed = config["system"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = get_device(config)
    print(f"Using device: {device}")

    # Build model
    mc = config["model"]
    model_config = TransformerConfig(
        vocab_size=mc["vocab_size"],
        d_model=mc["d_model"],
        n_heads=mc["n_heads"],
        n_layers=mc["n_layers"],
        d_ff=mc["d_ff"],
        max_seq_len=mc["max_seq_len"],
        dropout=mc["dropout"],
        bias=mc["bias"],
    )
    model = Transformer(model_config).to(device)
    print(f"Model parameters: {model.param_count():,}")

    if config["system"]["compile"] and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Data
    train_path = os.path.join(args.data_dir, "train.bin")
    val_path = os.path.join(args.data_dir, "val.bin")
    if not os.path.exists(train_path):
        print(f"Data not found at {train_path}. Run scripts/prepare_data.py first.")
        return

    train_loader, val_loader = create_dataloaders(
        train_path=train_path,
        val_path=val_path,
        seq_len=mc["max_seq_len"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    # Train
    trainer = Trainer(model, train_loader, val_loader, config, device)
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
