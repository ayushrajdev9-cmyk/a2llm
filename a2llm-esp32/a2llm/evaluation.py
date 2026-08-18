"""Evaluation: validation loss, perplexity, sizes, speed — a single report.

Perplexity = exp(mean cross-entropy loss): the effective branching factor
the model sees at each next-token decision. Uniform random over V tokens
gives ppl = V; a perfect model gives 1.0. Generated text quality is NOT a
metric here - look at samples for that.
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .checkpoint import load_model_for_inference
from .dataset import get_batch, load_corpus
from .utils import TokensPerSecond, format_bytes, model_report


@torch.no_grad()
def evaluate(checkpoint_path: str | Path, device: str = "cpu",
             val_steps: int = 50, seed: int = 42) -> dict:
    """Full evaluation report dict for a checkpoint."""
    model, tokenizer, cfg = load_model_for_inference(checkpoint_path, device=device)
    corpus = load_corpus(
        cfg.data.data_path, tokenizer,
        train_split=cfg.data.train_split, seed=cfg.data.seed,
    )

    rng = random.Random(seed)
    total, count = 0.0, 0
    tps = TokensPerSecond()
    tps.start()
    for _ in range(val_steps):
        xs, ys = get_batch(
            corpus.val, cfg.model.context_length, cfg.train.batch_size, device, rng
        )
        _, loss = model(xs, ys)
        total += loss.item()
        count += 1
        tps.add(cfg.train.batch_size * cfg.model.context_length)
    val_loss = total / count
    ppl = math.exp(min(val_loss, 20.0))

    report = model_report(model)
    report.update(
        {
            "checkpoint": str(checkpoint_path),
            "device": device,
            "validation_loss": val_loss,
            "perplexity": ppl,
            "generation_speed_tok_s": tps.rate,
            "vocab_size": cfg.model.vocab_size,
            "context_length": cfg.model.context_length,
        }
    )
    return report


def print_report(report: dict) -> None:
    """Pretty-print an evaluation report."""
    print("A2LM evaluation report")
    print("-" * 44)
    print(f"Checkpoint:       {report['checkpoint']}")
    print(f"Parameters:       {report['total']:,} (trainable {report['trainable']:,})")
    print(f"Model size:       {report['int8_size_human']} int8 | "
          f"{format_bytes(report['fp32_size_bytes'])} fp32")
    print(f"Validation loss:  {report['validation_loss']:.4f}")
    print(f"Perplexity:       {report['perplexity']:.2f}")
    print(f"Generation speed: {report['generation_speed_tok_s']:.0f} tok/s "
          f"({report['device']})")
    print(f"Vocab / context:  {report['vocab_size']} / {report['context_length']}")
    print("-" * 44)
    print("Baseline: uniform random over the vocab has perplexity = vocab size.")