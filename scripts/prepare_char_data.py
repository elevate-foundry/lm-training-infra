#!/usr/bin/env python3
"""Download TinyStories and write character-level binary files."""

import argparse
import os

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--val_fraction", type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.dataset}...")
    ds = load_dataset(args.dataset, trust_remote_code=True)

    # TinyStories has train/validation splits
    for split_name, out_name in [("train", "train"), ("validation", "val")]:
        if split_name not in ds:
            continue
        print(f"Processing {split_name}...")
        all_bytes = bytearray()
        for example in tqdm(ds[split_name]):
            text = example.get("text", "")
            if text:
                all_bytes.extend(text.encode("utf-8"))

        arr = np.frombuffer(bytes(all_bytes), dtype=np.uint8)
        out_path = os.path.join(args.output_dir, f"{out_name}.bin")
        arr.tofile(out_path)
        print(f"Saved {len(arr):,} bytes to {out_path}")


if __name__ == "__main__":
    main()
