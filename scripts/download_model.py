#!/usr/bin/env python3
"""Download pre-trained A2LLM checkpoints from GitHub Releases.

Usage:
    python scripts/download_model.py                # list available models
    python scripts/download_model.py nano           # download the nano checkpoint
    python scripts/download_model.py micro --out checkpoints/micro/best.pt
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO = "ayushrajdev9-cmyk/a2llm"
RELEASE = "v0.3.0"
# asset name -> friendly name (keep in sync with the GitHub release)
MODELS = {
    "a2llm-esp32-nano.pt": "nano  (A2LLM-ESP32, 34,432 params, 33 KB int8 flash)",
    "a2llm-micro.pt": "micro (120,064 params, multilingual, val_loss 1.5686, ppl 4.80)",
    "a2llm-mini.pt": "mini (1,056,096 params, default PC model, val_loss 3.3162, ppl 27.56, 4000 steps on T4)",
}

def main() -> None:
    ap = argparse.ArgumentParser(description="Download A2LLM checkpoints")
    ap.add_argument("model", nargs="?", default=None,
                    help="model to download: " + ", ".join(MODELS))
    ap.add_argument("--out", default=None, help="destination path")
    args = ap.parse_args()

    if args.model is None:
        print("Available models (GitHub release %s):" % RELEASE)
        for asset, desc in MODELS.items():
            print(f"  {asset:<22} {desc}")
        print("\nUsage: python scripts/download_model.py <model>")
        return

    if args.model not in MODELS:
        # also accept the friendly name (e.g. "micro" for "a2llm-micro.pt")
        match = [a for a in MODELS if args.model in a]
        if len(match) == 1:
            args.model = match[0]
        else:
            sys.exit(f"unknown model {args.model!r}; choose from: {', '.join(MODELS)}")

    url = f"https://github.com/{REPO}/releases/download/{RELEASE}/{args.model}"
    out = Path(args.out or f"checkpoints/{args.model}")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, out)
    print(f"saved to {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()