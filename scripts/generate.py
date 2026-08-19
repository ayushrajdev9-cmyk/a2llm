#!/usr/bin/env python3
"""Generate text with a trained A2LM checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from a2llm.checkpoint import load_model_for_inference
from a2llm.generate import generate_text
from a2llm.training import pick_device


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate text with A2LM")
    ap.add_argument("checkpoint", help="path to a .pt checkpoint")
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.9, help=">0; 1.0 = model distribution")
    ap.add_argument("--top-k", type=int, default=50, help="sample from top-k only (None = off)")
    ap.add_argument("--top-p", type=float, default=None, help="nucleus sampling threshold (None = off)")
    ap.add_argument("--stop", default=None, help="stop generation at this text (e.g. '\\n\\n')")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, tokenizer, cfg = load_model_for_inference(args.checkpoint, device=device)
    print(f"[generate] loaded {model!r}")

    stop_tokens = None
    if args.stop:
        stop_tokens = tokenizer.encode(args.stop)
        print(f"[generate] stop at {args.stop!r} (token ids {stop_tokens})")

    continuation = generate_text(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop_tokens=stop_tokens,
        device=device,
        seed=args.seed,
    )
    print(f"[generate] temperature={args.temperature} top_k={args.top_k} "
          f"top_p={args.top_p} max_new_tokens={args.max_new_tokens}")
    print("=" * 60)
    print(args.prompt + continuation)
    print("=" * 60)


if __name__ == "__main__":
    main()