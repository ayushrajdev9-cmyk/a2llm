#!/usr/bin/env python3
"""Export a trained A2LM checkpoint to the ESP32 C engine.

Usage:
    python scripts/export_esp32.py checkpoints/best.pt [-o esp32/main/weights.h]

Requires a byte-tokenizer checkpoint with vocab_size == 256 (that is what
the C engine encodes). Produces esp32/main/weights.h consumed by
esp32/main/a2lm.c.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from a2lm.checkpoint import load_checkpoint, restore_tokenizer
from a2lm.model import A2LM
from a2lm.quantize import export_model_to_header


def main() -> None:
    ap = argparse.ArgumentParser(description="Export checkpoint -> ESP32 weights.h")
    ap.add_argument("checkpoint", help="path to a .pt checkpoint (byte tokenizer)")
    ap.add_argument("-o", "--out", default="esp32/main/weights.h")
    args = ap.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    tok = restore_tokenizer(ckpt["tokenizer"])
    print(f"[export] tokenizer: {type(tok).__name__}")

    from a2lm.config import A2LMConfig
    cfg = A2LMConfig.from_dict(ckpt["config"])
    model = A2LM(cfg.model)
    model.load_state_dict(ckpt["model"])
    model.eval()

    meta = export_model_to_header(
        model, args.out, tokenizer_type=ckpt["tokenizer"]["type"]
    )
    print(f"[export] params: {meta['params']:,}")
    print(f"[export] int8 weights: {meta['int8_bytes']:,} bytes "
          f"({meta['int8_bytes'] / 1024:.1f} KB flash)")
    print(f"[export] wrote {meta['header']}")


if __name__ == "__main__":
    main()