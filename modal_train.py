#!/usr/bin/env python3
"""Run BrailleFormer and baseline training on Modal (remote GPU).

Usage:
    modal run modal_train.py --model brailleformer
    modal run modal_train.py --model baseline
    modal run modal_train.py --model both
"""

import modal

app = modal.App("brailleformer-poc")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "datasets>=2.14.0",
        "wandb>=0.15.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "safetensors>=0.4.0",
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_file("train_poc.py", remote_path="/root/train_poc.py")
)

volume = modal.Volume.from_name("brailleformer-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=3600 * 4,  # 4 hours max
    volumes={"/data": volume},
)
def train(model_type: str = "brailleformer", wandb_key: str | None = None):
    """Train a model on Modal GPU."""
    import subprocess
    import sys

    # Set up wandb if key provided
    if wandb_key:
        import os
        os.environ["WANDB_API_KEY"] = wandb_key

    # Prepare data if not already present
    import os
    if not os.path.exists("/data/train.bin"):
        print("Preparing data...")
        subprocess.run(
            [sys.executable, "-c", """
import os, numpy as np
from datasets import load_dataset
from tqdm import tqdm

os.makedirs("/data", exist_ok=True)
ds = load_dataset("roneneldan/TinyStories", trust_remote_code=True)
for split_name, out_name in [("train", "train"), ("validation", "val")]:
    if split_name not in ds:
        continue
    print(f"Processing {split_name}...")
    all_bytes = bytearray()
    for ex in tqdm(ds[split_name]):
        text = ex.get("text", "")
        if text:
            all_bytes.extend(text.encode("utf-8"))
    arr = np.frombuffer(bytes(all_bytes), dtype=np.uint8)
    path = f"/data/{out_name}.bin"
    arr.tofile(path)
    print(f"Saved {len(arr):,} bytes to {path}")
"""],
            check=True,
        )
        volume.commit()

    # Run training
    config_name = "brailleformer" if model_type == "brailleformer" else "baseline"
    subprocess.run(
        [
            sys.executable, "/root/train_poc.py",
            "--config", f"/root/configs/{config_name}.yaml",
            "--data_dir", "/data",
            "--output_dir", f"/data/checkpoints/{model_type}",
        ],
        check=True,
    )
    volume.commit()
    return f"{model_type} training complete"


@app.local_entrypoint()
def main(model: str = "brailleformer", wandb_key: str | None = None):
    if model == "both":
        # Run both sequentially (or use .spawn() for parallel)
        print("Training BrailleFormer...")
        result1 = train.remote(model_type="brailleformer", wandb_key=wandb_key)
        print(result1)
        print("Training Baseline...")
        result2 = train.remote(model_type="baseline", wandb_key=wandb_key)
        print(result2)
    else:
        result = train.remote(model_type=model, wandb_key=wandb_key)
        print(result)
