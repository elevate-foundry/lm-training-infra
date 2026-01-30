#!/usr/bin/env python3
"""Unified training script for BrailleFormer PoC.

Trains either the BrailleFormer or baseline transformer on character-level data.
Works both locally and on Modal.
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from safetensors.torch import save_model

from src.model.brailleformer import BrailleFormer, BrailleFormerConfig
from src.model.baseline import BaselineTransformer, BaselineConfig
from src.data.char_dataset import CharDataset, CharTokenizer, create_char_dataloaders


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(config: dict, device: torch.device) -> nn.Module:
    mc = config["model"]
    model_type = mc.get("type", "brailleformer")

    if model_type == "brailleformer":
        cfg = BrailleFormerConfig(
            vocab_size=mc["vocab_size"],
            d_model=mc["d_model"],
            n_heads=mc["n_heads"],
            n_layers=mc["n_layers"],
            d_ff=mc["d_ff"],
            max_seq_len=mc["max_seq_len"],
            cell_size=mc.get("cell_size", 6),
            tensor_rank=mc.get("tensor_rank", 16),
            dropout=mc["dropout"],
        )
        model = BrailleFormer(cfg)
    else:
        cfg = BaselineConfig(
            vocab_size=mc["vocab_size"],
            d_model=mc["d_model"],
            n_heads=mc["n_heads"],
            n_layers=mc["n_layers"],
            d_ff=mc["d_ff"],
            max_seq_len=mc["max_seq_len"],
            dropout=mc["dropout"],
        )
        model = BaselineTransformer(cfg)

    return model.to(device)


def get_lr(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    decay_ratio = (step - warmup) / (max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def evaluate(model: nn.Module, val_loader, eval_steps: int, device: torch.device, dtype) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= eval_steps:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device.type, dtype=dtype):
            _, loss = model(x, y)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


def train(config: dict, data_dir: str, output_dir: str):
    # Seed
    seed = config["system"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Device
    if config["system"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["system"]["device"])
    print(f"Device: {device}")

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        config["system"]["dtype"]
    ]

    # Model
    model = build_model(config, device)
    model_type = config["model"].get("type", "brailleformer")
    print(f"Model: {model_type} | Parameters: {model.param_count():,}")

    # Data
    seq_len = config["data"]["seq_len"]
    tc = config["training"]
    train_loader, val_loader = create_char_dataloaders(
        train_path=os.path.join(data_dir, "train.bin"),
        val_path=os.path.join(data_dir, "val.bin"),
        seq_len=seq_len,
        batch_size=tc["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    # Optimizer
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": tc["weight_decay"]},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=tc["learning_rate"],
        betas=(tc["beta1"], tc["beta2"]),
        fused=(device.type == "cuda"),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    # W&B
    wandb_run = None
    try:
        import wandb
        lc = config["logging"]
        wandb_run = wandb.init(
            project=lc["wandb_project"],
            name=lc.get("wandb_run_name") or f"{model_type}-poc",
            config=config,
        )
    except Exception:
        pass

    # Training loop
    os.makedirs(output_dir, exist_ok=True)
    lc = config["logging"]
    model.train()
    train_iter = iter(train_loader)
    best_val_loss = float("inf")
    t0 = time.time()
    step = 0

    while step < tc["max_steps"]:
        lr = get_lr(step, tc["warmup_steps"], tc["max_steps"], tc["learning_rate"], tc["min_learning_rate"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        loss_accum = 0.0
        for _ in range(tc["gradient_accumulation_steps"]):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device.type, dtype=dtype):
                _, loss = model(x, y)
                loss = loss / tc["gradient_accumulation_steps"]
            scaler.scale(loss).backward()
            loss_accum += loss.item()

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), tc["max_grad_norm"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % lc["log_interval"] == 0:
            dt = time.time() - t0
            chars_per_sec = (
                tc["batch_size"] * seq_len * tc["gradient_accumulation_steps"]
                * lc["log_interval"] / dt
            )
            print(
                f"[{model_type}] step {step:>6d} | loss {loss_accum:.4f} | "
                f"lr {lr:.2e} | {chars_per_sec:.0f} char/s"
            )
            if wandb_run:
                wandb.log({"train/loss": loss_accum, "train/lr": lr, "train/chars_per_sec": chars_per_sec}, step=step)
            t0 = time.time()

        if step % lc["eval_interval"] == 0:
            val_loss = evaluate(model, val_loader, lc["eval_steps"], device, dtype)
            print(f"[{model_type}] step {step:>6d} | val_loss {val_loss:.4f}")
            if wandb_run:
                wandb.log({"val/loss": val_loss}, step=step)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(output_dir, "best")
                os.makedirs(ckpt_path, exist_ok=True)
                save_model(model, os.path.join(ckpt_path, "model.safetensors"))
                print(f"  → saved best model (val_loss={val_loss:.4f})")

        if step % lc["save_interval"] == 0:
            ckpt_path = os.path.join(output_dir, f"step_{step}")
            os.makedirs(ckpt_path, exist_ok=True)
            save_file(model.state_dict(), os.path.join(ckpt_path, "model.safetensors"))

    # Generate sample
    tokenizer = CharTokenizer()
    prompt = "Once upon a time"
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    model.eval()
    output_ids = model.generate(input_ids, max_new_tokens=200, temperature=0.8, top_k=50)
    generated = tokenizer.decode(output_ids[0].tolist())
    print(f"\n{'='*60}")
    print(f"[{model_type}] Sample generation:")
    print(generated)
    print(f"{'='*60}")
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")

    if wandb_run:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir or config["logging"]["output_dir"]
    train(config, args.data_dir, output_dir)


if __name__ == "__main__":
    main()
