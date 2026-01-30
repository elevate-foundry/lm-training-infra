#!/usr/bin/env python3
"""Download and tokenize a dataset into binary files for training."""

import argparse
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def tokenize_and_save(dataset_name: str, output_dir: str, tokenizer_name: str = "gpt2"):
    os.makedirs(output_dir, exist_ok=True)
    enc = tiktoken.get_encoding(tokenizer_name)

    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, trust_remote_code=True)

    for split in ["train", "validation"]:
        if split not in ds:
            continue
        split_name = "val" if split == "validation" else split
        out_path = os.path.join(output_dir, f"{split_name}.bin")

        all_tokens = []
        for example in tqdm(ds[split], desc=f"Tokenizing {split}"):
            text = example.get("text", "")
            if text:
                tokens = enc.encode_ordinary(text)
                all_tokens.extend(tokens)

        arr = np.array(all_tokens, dtype=np.uint16)
        arr.tofile(out_path)
        print(f"Saved {len(arr):,} tokens to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--dataset", type=str, default="openwebtext", help="HuggingFace dataset name")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="tiktoken encoding name")
    args = parser.parse_args()
    tokenize_and_save(args.dataset, args.output_dir, args.tokenizer)
