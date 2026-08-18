#!/usr/bin/env python3
"""Evaluate an A2LM checkpoint: loss, perplexity, sizes, speed."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from a2lm.evaluation import evaluate, print_report
from a2lm.training import pick_device


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate an A2LM checkpoint")
    ap.add_argument("checkpoint", help="path to a .pt checkpoint")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--val-steps", type=int, default=50, help="batches averaged")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = pick_device(args.device)
    report = evaluate(args.checkpoint, device=device, val_steps=args.val_steps, seed=args.seed)
    print_report(report)


if __name__ == "__main__":
    main()