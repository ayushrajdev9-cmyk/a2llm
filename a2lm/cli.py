"""A2LM command-line interface.

Usage:
    a2lm train    [--preset NAME | --config FILE] [overrides...]
    a2lm evaluate CHECKPOINT
    a2lm generate CHECKPOINT [--prompt TEXT] [--max-new-tokens N] ...
    a2lm count-params [CHECKPOINT]
    a2lm inspect CHECKPOINT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from a2lm.checkpoint import load_checkpoint, load_model_for_inference
from a2lm.config import A2LMConfig, PRESETS, preset
from a2lm.training import train
from a2lm.utils import format_bytes, model_report


def _apply_overrides(cfg: A2LMConfig, args: argparse.Namespace) -> A2LMConfig:
    for key, value in vars(args).items():
        if value is None or key in ("command", "preset", "config", "resume", "device", "out"):
            continue
        for section in ("model", "data", "train"):
            if hasattr(getattr(cfg, section), key):
                setattr(getattr(cfg, section), key, value)
                break
        else:
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    return cfg


def cmd_train(args: argparse.Namespace) -> None:
    if args.config:
        cfg = preset(args.preset)
        overrides = json.loads(Path(args.config).read_text(encoding="utf-8"))
        for section, values in overrides.items():
            if section in ("model", "data", "train") and isinstance(values, dict):
                getattr(cfg, section).__dict__.update(values)
            else:
                setattr(cfg, section, values)
        cfg = _apply_overrides(cfg, args)
    else:
        cfg = _apply_overrides(preset(args.preset), args)
    cfg.device = args.device
    train(cfg, args)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from a2lm.evaluation import evaluate, print_report
    from a2lm.training import pick_device
    device = pick_device(args.device)
    print_report(evaluate(args.checkpoint, device=device, val_steps=args.val_steps))


def cmd_generate(args: argparse.Namespace) -> None:
    from a2lm.generate import generate_text
    from a2lm.training import pick_device
    device = pick_device(args.device)
    model, tokenizer, _ = load_model_for_inference(args.checkpoint, device=device)
    stop_tokens = tokenizer.encode(args.stop) if args.stop else None
    text = generate_text(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p,
        stop_tokens=stop_tokens, device=device, seed=args.seed,
    )
    print(args.prompt + text)


def cmd_count_params(args: argparse.Namespace) -> None:
    if args.checkpoint:
        model, _, _ = load_model_for_inference(args.checkpoint)
    else:
        from a2lm.model import A2LM
        model = A2LM(preset(args.preset).model)
    r = model_report(model)
    print(f"Total parameters:       {r['total']:,}")
    print(f"Trainable parameters:   {r['trainable']:,}")
    print(f"Non-trainable:          {r['non_trainable']:,}")
    print(f"Size (fp32):            {format_bytes(r['fp32_size_bytes'])}")
    print(f"Size (fp16):            {format_bytes(r['fp16_size_bytes'])}")
    print(f"Size (int8):            {format_bytes(r['int8_size_bytes'])}")


def cmd_inspect(args: argparse.Namespace) -> None:
    ckpt = load_checkpoint(args.checkpoint)
    cfg = A2LMConfig.from_dict(ckpt["config"])
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Run name:    {cfg.run_name}")
    print(f"Step:        {ckpt.get('step')}")
    print(f"Best val:    {ckpt.get('best_val_loss'):.4f}")
    print(f"Tokenizer:   {ckpt['tokenizer']['type']}")
    print(f"Model:       {cfg.model}")
    print(f"Train:       {cfg.train}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="a2lm", description="A2LM language model toolkit")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train a model")
    p.add_argument("--preset", default="mini", choices=sorted(PRESETS))
    p.add_argument("--config", default=None, help="JSON config file")
    p.add_argument("--data", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--ctx", type=int, default=None)
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--tokenizer", default=None, choices=["byte", "char"])
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--resume", default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="evaluate a checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("--device", default="auto")
    p.add_argument("--val-steps", type=int, default=50)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("generate", help="generate text")
    p.add_argument("checkpoint")
    p.add_argument("--prompt", default="ROMEO:")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--stop", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("count-params", help="report parameter counts and sizes")
    p.add_argument("checkpoint", nargs="?", default=None)
    p.add_argument("--preset", default="mini", choices=sorted(PRESETS))
    p.set_defaults(func=cmd_count_params)

    p = sub.add_parser("inspect", help="show checkpoint metadata")
    p.add_argument("checkpoint")
    p.set_defaults(func=cmd_inspect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()