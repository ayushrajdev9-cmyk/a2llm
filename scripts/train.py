#!/usr/bin/env python3
"""Train an A2LM model. Thin CLI over a2lm.training.train()."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from a2lm.config import preset
from a2lm.training import train


def main() -> None:
    ap = argparse.ArgumentParser(description="Train A2LM")
    ap.add_argument("--preset", default="mini",
                    help="nano|micro|mini|small|base|large")
    ap.add_argument("--config", default=None,
                    help="JSON config file (overrides preset defaults)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--tokenizer", default=None, help="byte|char")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="checkpoint dir override")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--no-tie", action="store_true", help="disable weight tying")
    args = ap.parse_args()

    if args.config:
        import json
        cfg = preset(args.preset)
        overrides = json.loads(Path(args.config).read_text(encoding="utf-8"))
        # overrides may be a full config dict or {"model": {...}, "train": {...}}
        for section, values in overrides.items():
            if section in ("model", "data", "train") and isinstance(values, dict):
                getattr(cfg, section).__dict__.update(values)
            else:
                setattr(cfg, section, values)
    else:
        cfg = preset(args.preset)

    if args.data: cfg.data.data_path = args.data
    if args.steps: cfg.train.num_steps = args.steps
    if args.batch_size: cfg.train.batch_size = args.batch_size
    if args.lr: cfg.train.learning_rate = args.lr
    if args.ctx: cfg.model.context_length = args.ctx
    if args.dim: cfg.model.embedding_dim = args.dim
    if args.layers: cfg.model.num_layers = args.layers
    if args.heads: cfg.model.num_heads = args.heads
    if args.tokenizer: cfg.data.tokenizer_type = args.tokenizer
    if args.seed: cfg.train.seed = args.seed
    if args.out: cfg.checkpoint_dir = args.out
    if args.grad_accum: cfg.train.grad_accumulation = args.grad_accum
    if args.no_tie: cfg.model.tie_embeddings = False
    cfg.device = args.device

    train(cfg, args)


if __name__ == "__main__":
    main()