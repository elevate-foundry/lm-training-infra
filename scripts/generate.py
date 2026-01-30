#!/usr/bin/env python3
"""Generate text from a trained model checkpoint."""

import argparse

import tiktoken
import torch
import yaml
from safetensors.torch import load_file

from src.model import Transformer, TransformerConfig


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--prompt", type=str, default="The meaning of life is", help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    mc = config["model"]
    model_config = TransformerConfig(
        vocab_size=mc["vocab_size"],
        d_model=mc["d_model"],
        n_heads=mc["n_heads"],
        n_layers=mc["n_layers"],
        d_ff=mc["d_ff"],
        max_seq_len=mc["max_seq_len"],
        dropout=0.0,
        bias=mc["bias"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(model_config).to(device)
    state_dict = load_file(f"{args.checkpoint}/model.safetensors", device=str(device))
    model.load_state_dict(state_dict)
    model.eval()

    enc = tiktoken.get_encoding(config["data"]["tokenizer"])
    input_ids = torch.tensor([enc.encode(args.prompt)], device=device)

    print(f"Prompt: {args.prompt}")
    print("=" * 60)

    output_ids = model.generate(
        input_ids, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k
    )
    print(enc.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
