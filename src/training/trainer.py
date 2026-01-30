"""Training loop with gradient accumulation, mixed precision, and checkpointing."""

import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from safetensors.torch import save_file, load_file


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        tc = config["training"]
        self.max_steps = tc["max_steps"]
        self.grad_accum_steps = tc["gradient_accumulation_steps"]
        self.max_grad_norm = tc["max_grad_norm"]
        self.lr = tc["learning_rate"]
        self.min_lr = tc["min_learning_rate"]
        self.warmup_steps = tc["warmup_steps"]

        lc = config["logging"]
        self.log_interval = lc["log_interval"]
        self.eval_interval = lc["eval_interval"]
        self.eval_steps = lc["eval_steps"]
        self.save_interval = lc["save_interval"]
        self.output_dir = lc["output_dir"]

        # Set up dtype
        dtype_str = config["system"]["dtype"]
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype_str]
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.dtype == torch.float16))

        # Optimizer: AdamW with separate weight decay for non-bias/norm params
        decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
        nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": tc["weight_decay"]},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=self.lr,
            betas=(tc["beta1"], tc["beta2"]),
            fused=device.type == "cuda",
        )

        self.step = 0
        self.best_val_loss = float("inf")
        os.makedirs(self.output_dir, exist_ok=True)

        self._try_wandb_init(lc)

    def _try_wandb_init(self, lc: dict):
        self.wandb_run = None
        try:
            import wandb

            self.wandb_run = wandb.init(
                project=lc["wandb_project"],
                name=lc.get("wandb_run_name"),
                config=self.config,
            )
        except Exception:
            pass

    def _get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.lr * (step + 1) / self.warmup_steps
        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.lr - self.min_lr)

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        for i, (x, y) in enumerate(self.val_loader):
            if i >= self.eval_steps:
                break
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast(self.device.type, dtype=self.dtype):
                _, loss = self.model(x, y)
            total_loss += loss.item()
        self.model.train()
        return total_loss / min(self.eval_steps, len(self.val_loader))

    def _save_checkpoint(self, name: str):
        path = os.path.join(self.output_dir, name)
        os.makedirs(path, exist_ok=True)
        save_file(self.model.state_dict(), os.path.join(path, "model.safetensors"))
        torch.save(
            {
                "step": self.step,
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "config": self.config,
            },
            os.path.join(path, "training_state.pt"),
        )

    def load_checkpoint(self, path: str):
        state_dict = load_file(os.path.join(path, "model.safetensors"))
        self.model.load_state_dict(state_dict)
        training_state = torch.load(
            os.path.join(path, "training_state.pt"), map_location=self.device, weights_only=True
        )
        self.optimizer.load_state_dict(training_state["optimizer"])
        self.scaler.load_state_dict(training_state["scaler"])
        self.step = training_state["step"]
        self.best_val_loss = training_state["best_val_loss"]

    def train(self):
        self.model.train()
        train_iter = iter(self.train_loader)
        t0 = time.time()

        while self.step < self.max_steps:
            lr = self._get_lr(self.step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Gradient accumulation
            loss_accum = 0.0
            for micro_step in range(self.grad_accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)
                x, y = x.to(self.device), y.to(self.device)

                with torch.amp.autocast(self.device.type, dtype=self.dtype):
                    _, loss = self.model(x, y)
                    loss = loss / self.grad_accum_steps
                self.scaler.scale(loss).backward()
                loss_accum += loss.item()

            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1

            # Logging
            if self.step % self.log_interval == 0:
                dt = time.time() - t0
                tokens_per_sec = (
                    self.config["training"]["batch_size"]
                    * self.config["model"]["max_seq_len"]
                    * self.grad_accum_steps
                    * self.log_interval
                    / dt
                )
                print(
                    f"step {self.step:>6d} | loss {loss_accum:.4f} | "
                    f"lr {lr:.2e} | {tokens_per_sec:.0f} tok/s"
                )
                if self.wandb_run:
                    import wandb

                    wandb.log(
                        {"train/loss": loss_accum, "train/lr": lr, "train/tokens_per_sec": tokens_per_sec},
                        step=self.step,
                    )
                t0 = time.time()

            # Evaluation
            if self.step % self.eval_interval == 0:
                val_loss = self._evaluate()
                print(f"step {self.step:>6d} | val_loss {val_loss:.4f}")
                if self.wandb_run:
                    import wandb

                    wandb.log({"val/loss": val_loss}, step=self.step)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best")

            # Save periodic checkpoint
            if self.step % self.save_interval == 0:
                self._save_checkpoint(f"step_{self.step}")

        self._save_checkpoint("final")
        print(f"Training complete. Best val loss: {self.best_val_loss:.4f}")
