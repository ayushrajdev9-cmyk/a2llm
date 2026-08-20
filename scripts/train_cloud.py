#!/usr/bin/env python3
"""Train any A2LLM tier on a big GPU (RTX PRO 6000 / any cloud box) and auto-publish to GitHub.

Usage:
    GH_TOKEN=ghp_xxx python scripts/train_cloud.py --preset base
    GH_TOKEN=ghp_xxx python scripts/train_cloud.py --all          # train all 6 tiers, publish each
    python scripts/train_cloud.py --preset base --no-publish      # dry run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import urllib.request

# tokens/step = batch * ctx (all ~65k); 30k steps ≈ ~3 epochs of the combined corpus
PRESETS = {
    "nano":  {"steps": 30000, "batch": 2048, "asset": "a2llm-nano.pt",  "release": "v0.4.0", "row": "a2llm-esp32-nano", "params": "34,432"},
    "micro": {"steps": 30000, "batch": 1024, "asset": "a2llm-micro.pt", "release": "v0.7.0", "row": "a2llm-micro",      "params": "120,064"},
    "mini":  {"steps": 30000, "batch": 512,  "asset": "a2llm-mini.pt",  "release": "v0.6.0", "row": "a2llm-mini",       "params": "1,056,096"},
    "small": {"steps": 30000, "batch": 256,  "asset": "a2llm-small.pt", "release": "v0.5.0", "row": "a2llm-small",      "params": "2,763,264"},
    "base":  {"steps": 30000, "batch": 256,  "asset": "a2llm-base.pt",  "release": "v0.8.0", "row": "a2llm-base",       "params": "4,864,000"},
    "large": {"steps": 30000, "batch": 128,  "asset": "a2llm-large.pt", "release": "v0.9.0", "row": "a2llm-large",      "params": "14,479,104"},
}
ORDER = ["nano", "micro", "mini", "small", "base", "large"]
OWNER, REPO = "ayushrajdev9-cmyk", "a2llm"
DATA = "data/pretrain_multilingual.txt"


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def api(url: str, token: str, method: str = "GET", body: dict | None = None,
        raw: bytes | None = None, ctype: str | None = None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github+json")
    if ctype:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def gpu_report() -> None:
    import torch
    print(f"torch {torch.__version__}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {p.total_memory/1e9:.1f} GB | CC: {p.major}.{p.minor}")
        if p.major >= 12 and int(torch.__version__.split(".")[1]) < 7:
            print("WARNING: Blackwell (CC 12.x) needs torch>=2.7:  pip install torch --index-url https://download.pytorch.org/whl/cu128")
    else:
        print("WARNING: no CUDA — training will be slow")


def ensure_data() -> None:
    if not os.path.exists(DATA):
        print("corpus missing — building (downloads cached, ~7 min first time)")
        sh("python scripts/prepare_data_large.py")
    else:
        print(f"corpus present: {DATA} ({os.path.getsize(DATA)/1e6:.0f} MB)")


def train(preset: str, steps: int, batch: int) -> None:
    print(f"\n===== TRAINING {preset}: {steps} steps x batch {batch} =====")
    sh(f"python scripts/train.py --preset {preset} --data {DATA} --steps {steps} --batch-size {batch} --out checkpoints/{preset}")


def publish(preset: str, token: str) -> None:
    cfg = PRESETS[preset]
    import torch
    ck = torch.load(f"checkpoints/{preset}/best.pt", map_location="cpu", weights_only=False)
    loss, ppl = float(ck["best_val_loss"]), math.exp(float(ck["best_val_loss"]))
    tag, asset = cfg["release"], cfg["asset"]
    notes = (f"MODEL: {preset}\n**{asset}** — {preset} tier trained on combined corpus "
             f"(multilingual en+hi+ur+bn + gpt-oss_20b distilled QA, {cfg['steps']} steps x batch {cfg['batch']}).\n"
             f"- params: {cfg['params']}\n- best val_loss {loss:.4f}, ppl {ppl:.2f}")
    base = f"https://api.github.com/repos/{OWNER}/{REPO}"
    st, r = api(f"{base}/releases", token, "POST",
                {"tag_name": tag, "name": f"A2LLM {tag} — {preset}", "body": notes})
    if st == 422:  # tag exists -> update instead of create
        _, ex = api(f"{base}/releases/tags/{tag}", token)
        st, r = api(f"{base}/releases/{ex['id']}", token, "PATCH", {"body": notes})
    assert st < 300, f"release failed: {st} {r}"
    rid = r["id"]
    with open(f"checkpoints/{preset}/best.pt", "rb") as f:
        data = f.read()
    st, u = api(f"https://uploads.github.com/repos/{OWNER}/{REPO}/releases/{rid}/assets?name={asset}",
                token, "POST", raw=data, ctype="application/octet-stream")
    assert st < 300, f"upload failed: {st} {u}"
    print(f"✅ RELEASED: https://github.com/{OWNER}/{REPO}/releases/tag/{tag}  (val_loss {loss:.4f}, ppl {ppl:.2f})")
    # docs update: MODELS.md + README.md via a fresh shallow clone
    sh(f"rm -rf /tmp/a2llm_docs && git clone -q --depth 1 https://x-access-token:{token}@github.com/{OWNER}/{REPO}.git /tmp/a2llm_docs")
    row, tag_url = cfg["row"], f"https://github.com/{OWNER}/{REPO}/releases/tag/{tag}"
    md = open("/tmp/a2llm_docs/MODELS.md").read()
    for line in md.splitlines():
        low = line.lower()
        if (row in low or (row == "a2llm-esp32-nano" and "a2llm-esp32 (nano)" in low)) and "trained" in low:
            indent = line[: len(line) - len(line.lstrip())]
            md = md.replace(line, f"{indent}- {line.strip().split(':', 1)[0]}: TRAINED ✅  (release {tag}, combined corpus, val_loss {loss:.4f}, ppl {ppl:.2f})")
            break
    open("/tmp/a2llm_docs/MODELS.md", "w").write(md)
    rd = open("/tmp/a2llm_docs/README.md").read()
    rd = re.sub(r"(\|\s*" + re.escape(row) + r"\s*\|[^|]*\|)([^|]*\|)([^|]*\|)",
                lambda m: m.group(1) + " ✅ trained | " + tag + " |", rd, count=1)
    if f"[{tag}]" not in rd:
        sep = re.search(r"(\| Release \| Assets \| Result \|\n\|[-| ]+\|\n)", rd)
        if sep:
            rd = rd.replace(sep.group(1), sep.group(1) + f"| [{tag}]({tag_url}) | `{asset}` | val_loss {loss:.4f}, ppl {ppl:.2f} |\n")
    open("/tmp/a2llm_docs/README.md", "w").write(rd)
    sh(f"cd /tmp/a2llm_docs && git config user.name 'a2llm-bot' && git config user.email 'bot@a2llm' && "
       f"git add MODELS.md README.md && git commit -q -m 'docs: {tag} {preset} (val_loss {loss:.4f}, ppl {ppl:.2f})' && git push -q origin main")
    print("📄 docs updated + pushed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS)
    ap.add_argument("--all", action="store_true", help="train all 6 tiers in order")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--token", default=os.environ.get("GH_TOKEN", ""))
    ap.add_argument("--no-publish", action="store_true")
    args = ap.parse_args()

    if not args.preset and not args.all:
        sys.exit("give --preset <tier> or --all")
    if not args.token and not args.no_publish:
        sys.exit("set GH_TOKEN=... (repo scope) or pass --no-publish")

    gpu_report()
    ensure_data()
    presets = ORDER if args.all else [args.preset]
    for p in presets:
        train(p, args.steps or PRESETS[p]["steps"], args.batch_size or PRESETS[p]["batch"])
        if not args.no_publish:
            publish(p, args.token)
    print("\nALL DONE 🎉")


if __name__ == "__main__":
    main()