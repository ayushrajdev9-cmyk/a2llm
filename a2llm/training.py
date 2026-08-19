"""Training engine: the full loop, reusable by the CLI and scripts.

Handles device selection, data loading, AdamW + cosine schedule with warmup,
gradient clipping, optional mixed precision (CUDA), gradient accumulation,
validation, periodic sample generation, checkpointing, and experiment
tracking (experiments/exp_NNNN/).
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .checkpoint import BEST_NAME, LATEST_NAME, load_checkpoint, save_checkpoint
from .config import A2LMConfig, preset
from .dataset import get_batch, load_corpus
from .model import A2LM
from .tokenizer import build_tokenizer
from .utils import TokensPerSecond, gpu_memory_mb, write_experiment_meta


def pick_device(device: str) -> str:
    """auto -> cuda if available else cpu; explicit names validated."""
    if device == "auto":
        chosen = "cuda" if torch.cuda.is_available() else "cpu"
        return chosen
    if device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' requested but CUDA is not available")
        return device
    raise ValueError(f"unknown device {device!r}; use 'auto', 'cpu', or 'cuda'")


def learning_rate_at(step: int, cfg: A2LMConfig) -> float:
    """Linear warmup, then cosine decay to min_learning_rate."""
    t = cfg.train
    if step < t.warmup_steps:
        return t.learning_rate * (step + 1) / max(1, t.warmup_steps)
    progress = (step - t.warmup_steps) / max(1, t.num_steps - t.warmup_steps)
    progress = min(progress, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return t.min_learning_rate + cosine * (t.learning_rate - t.min_learning_rate)


@torch.no_grad()
def evaluate_loss(
    model: A2LM, stream, cfg: A2LMConfig, device: str, val_steps: int | None = None
) -> float:
    """Mean validation loss over val_steps batches. Model stays in train mode."""
    model.eval()
    rng = random.Random(cfg.train.seed + 1)
    total, count = 0.0, 0
    steps = val_steps or cfg.train.val_steps
    for _ in range(steps):
        xs, ys = get_batch(
            stream, cfg.model.context_length, cfg.train.batch_size, device, rng
        )
        _, loss = model(xs, ys)
        total += loss.item()
        count += 1
    model.train()
    return total / count


def build_experiment_dir(base: Path) -> Path:
    """Return the next free experiments/exp_NNNN directory (never overwrites)."""
    base.mkdir(parents=True, exist_ok=True)
    existing = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("exp_")]
    nums = [int(p.name.split("_")[1]) for p in existing if p.name.split("_")[1].isdigit()]
    n = (max(nums) + 1) if nums else 1
    return base / f"exp_{n:04d}"


def train(cfg: A2LMConfig, args: argparse.Namespace | None = None) -> dict:
    """Run (or resume) a training run. Returns final metrics."""
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.train.seed)
    random.seed(cfg.train.seed)

    use_amp = device == "cuda"
    grad_accum = getattr(cfg.train, "grad_accumulation", 1) or 1

    # ---- Data -------------------------------------------------------------
    corpus_text = Path(cfg.data.data_path).read_text(encoding="utf-8")
    tokenizer = build_tokenizer(cfg.data.tokenizer_type, corpus_text)
    corpus = load_corpus(
        cfg.data.data_path,
        tokenizer,
        train_split=cfg.data.train_split,
        seed=cfg.data.seed,
    )
    print(f"[train] device: {device}")
    print(f"[train] corpus: {len(corpus.tokens):,} tokens | train: {len(corpus.train):,} "
          f"| val: {len(corpus.val):,} | vocab: {tokenizer.vocab_size}")

    # ---- Model ------------------------------------------------------------
    cfg.model.vocab_size = tokenizer.vocab_size
    model = A2LM(cfg.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"[train] model: {model!r}")

    start_step, best_val = 0, float("inf")
    if args and args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        best_val = ckpt.get("best_val_loss", float("inf"))
        print(f"[train] resumed from {args.resume} at step {start_step}")

    # ---- Experiment tracking ---------------------------------------------
    exp_dir = build_experiment_dir(Path(cfg.checkpoint_dir).parent / "experiments")
    print(f"[train] experiment dir: {exp_dir}")
    write_experiment_meta(exp_dir, cfg.to_dict(), {"status": "running"})

    out_dir = Path(cfg.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.train.seed)
    tps = TokensPerSecond()
    tps.start()
    t0 = time.time()

    metrics = {"train_losses": [], "val_losses": [], "val_ppls": []}
    for step in range(start_step, cfg.train.num_steps):
        lr = learning_rate_at(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Gradient accumulation: average loss over grad_accum micro-batches.
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(grad_accum):
            xs, ys = get_batch(
                corpus.train, cfg.model.context_length,
                cfg.train.batch_size, device, rng,
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                _, loss = model(xs, ys)
            accum_loss += loss.item()
            scaler.scale(loss / grad_accum).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        tps.add(cfg.train.batch_size * cfg.model.context_length * grad_accum)

        if (step + 1) % cfg.train.log_every == 0:
            mem = gpu_memory_mb()
            mem_s = f" gpu_mem={mem:.0f}MB" if mem else ""
            print(f"[train] step {step + 1}/{cfg.train.num_steps} "
                  f"loss={accum_loss / grad_accum:.4f} lr={lr:.2e} "
                  f"{tps.rate:.0f} tok/s elapsed={time.time() - t0:.0f}s{mem_s}")

        if (step + 1) % cfg.train.eval_every == 0:
            val_loss = evaluate_loss(model, corpus.val, cfg, device)
            ppl = math.exp(min(val_loss, 20.0))
            metrics["val_losses"].append(val_loss)
            metrics["val_ppls"].append(ppl)
            print(f"[train] step {step + 1}: val_loss={val_loss:.4f} val_ppl={ppl:.2f}")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    out_dir / BEST_NAME, cfg, model, optimizer,
                    step + 1, best_val, corpus.tokenizer,
                )

        if (step + 1) % cfg.train.sample_every == 0:
            model.eval()
            prompt = "ROMEO:"
            ids = tokenizer.encode(prompt)
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            out = model.generate(idx, max_new_tokens=60, temperature=0.9, top_k=40)
            sample = tokenizer.decode(out[0].tolist())
            print(f"[train] sample @ step {step + 1}:\n{sample}\n")
            (exp_dir / f"sample_step_{step + 1}.txt").write_text(sample, encoding="utf-8")
            model.train()

        if (step + 1) % max(cfg.train.eval_every // 2, 1) == 0:
            save_checkpoint(
                out_dir / LATEST_NAME, cfg, model, optimizer,
                step + 1, best_val, corpus.tokenizer,
            )

    # ---- Final ------------------------------------------------------------
    val_loss = evaluate_loss(model, corpus.val, cfg, device)
    ppl = math.exp(min(val_loss, 20.0))
    if val_loss < best_val:
        best_val = val_loss
        save_checkpoint(out_dir / BEST_NAME, cfg, model, optimizer,
                        cfg.train.num_steps, best_val, corpus.tokenizer)
    save_checkpoint(out_dir / LATEST_NAME, cfg, model, optimizer,
                    cfg.train.num_steps, best_val, corpus.tokenizer)

    final = {
        "status": "done",
        "final_train_loss": metrics["train_losses"][-1] if metrics["train_losses"] else None,
        "val_loss": val_loss,
        "val_perplexity": ppl,
        "tokens_per_second": tps.rate,
        "elapsed_seconds": time.time() - t0,
        "steps": cfg.train.num_steps,
    }
    write_experiment_meta(exp_dir, cfg.to_dict(), final)
    print(f"[train] final val_loss={val_loss:.4f} val_ppl={ppl:.2f} "
          f"{tps.rate:.0f} tok/s in {time.time() - t0:.1f}s")
    print(f"[train] checkpoints in {out_dir}: {LATEST_NAME}, {BEST_NAME}")
    return final