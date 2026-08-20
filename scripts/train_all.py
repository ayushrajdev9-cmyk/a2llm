#!/usr/bin/env python3
"""A2LLM — train ALL models and auto-publish every release.

Run this on your big GPU box (molab RTX PRO 6000 / any cloud machine).

What it does:
  1. clones the a2llm repo (fresh copy)
  2. installs deps (torch cu128 for Blackwell GPUs, datasets, etc.)
  3. builds the COMBINED corpus = multilingual (542M tokens) + gpt-oss_20b distilled QA (~92M)
  4. trains all 6 tiers: nano -> micro -> mini -> small -> base -> large
  5. publishes each release automatically:
        nano   -> v0.4.0   a2llm-nano.pt
        small  -> v0.5.0   a2llm-small.pt
        mini   -> v0.6.0   a2llm-mini.pt
        micro  -> v0.7.0   a2llm-micro.pt
        base   -> v0.8.0   a2llm-base.pt
        large  -> v0.9.0   a2llm-large.pt
  6. updates MODELS.md + README.md and pushes

Usage:
    GH_TOKEN=ghp_xxx python train_all.py
"""
import os
import subprocess
import sys

TOKEN = os.environ.get("GH_TOKEN", "")
REPO = "https://github.com/ayushrajdev9-cmyk/a2llm.git"
WORK = "/tmp/a2llm"
TIERS = ["nano (v0.4.0)", "micro (v0.7.0)", "mini (v0.6.0)", "small (v0.5.0)", "base (v0.8.0)", "large (v0.9.0)"]


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    if not TOKEN:
        sys.exit("Set your GitHub token first:\n\n    GH_TOKEN=ghp_xxx python train_all.py\n")
    print("===== A2LLM: train ALL models + publish =====")
    print("plan:", " -> ".join(TIERS), flush=True)

    # 1. fresh clone
    sh(f"rm -rf {WORK} && git clone --depth 1 {REPO} {WORK}")

    # 2. deps (cu128 = Blackwell RTX PRO 6000; on older GPUs use cu124/cu121)
    sh("pip install -q torch --index-url https://download.pytorch.org/whl/cu128")
    sh(f"pip install -q -r {WORK}/requirements.txt pandas pyarrow datasets")

    # 3. combined corpus: multilingual + distilled QA
    os.chdir(WORK)
    sh("python scripts/prepare_data_large.py")

    # 4+5+6. train all tiers, publish each, push docs
    os.environ["GH_TOKEN"] = TOKEN
    sh("python scripts/train_cloud.py --all")

    print("""
ALL 6 MODELS TRAINED + PUBLISHED
  https://github.com/ayushrajdev9-cmyk/a2llm/releases
    """)


if __name__ == "__main__":
    main()