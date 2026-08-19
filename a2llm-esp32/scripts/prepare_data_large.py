#!/usr/bin/env python3
"""Build a large multilingual pretraining corpus (billions of tokens).

Sources (byte tokenizer => tokens == UTF-8 bytes):

  en: wikitext-103-raw (Wikipedia articles) + TinyStories (simple stories)
  hi: Common Crawl (mc4) Hindi
  ur: Common Crawl (mc4) Urdu
  bn: Common Crawl (mc4) Bengali
  es/fr: optional via --extra-es-fr

Default corpus: ~2.2 GB of text (~2.2 B tokens). Downloads are cached in
data/raw/ and skipped on re-runs.

Usage:
    python scripts/prepare_data_large.py
    python scripts/prepare_data_large.py --extra-es-fr   # + ~1.6B more tokens
    python scripts/prepare_data_large.py --out data/pretrain.txt
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.request
from pathlib import Path

WIKITEXT = "https://huggingface.co/datasets/Salesforce/wikitext/raw/main/wikitext-103-raw/wiki.train.raw"
TINYSTORIES = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
MC4 = "https://huggingface.co/datasets/allenai/c4/resolve/main/multilingual/c4-{lang}.tfrecord-{i:05d}-of-00064.json.gz"

# (lang, n_shards) — each shard is a gzipped JSONL file with a "text" field
LANG_SHARDS = {
    "hi": 4,
    "ur": 2,
    "bn": 2,
}
EXTRA_SHARDS = {"es": 4, "fr": 4}  # only with --extra-es-fr


def download(url: str, dest: Path, label: str) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [cache] {label}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"  [get] {label} <- {url}")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as exc:  # noqa: BLE001 - a failed source must not kill the build
        print(f"  [skip] {label}: {exc}", file=sys.stderr)
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a multilingual pretraining corpus")
    ap.add_argument("--out", default="data/pretrain_multilingual.txt")
    ap.add_argument("--extra-es-fr", action="store_true",
                    help="also pull Spanish + French shards (~+1.6B tokens)")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = Path("data/raw")
    langs = dict(LANG_SHARDS)
    if args.extra_es_fr:
        langs.update(EXTRA_SHARDS)

    sources: list[tuple[str, str, str]] = []  # (label, kind, url)
    sources.append(("en-wikitext103", "plain", WIKITEXT))
    sources.append(("en-tinystories", "plain", TINYSTORIES))
    for lang, n in langs.items():
        for i in range(n):
            url = MC4.format(lang=lang, i=i)
            sources.append((f"{lang}-{i:02d}", "jsonl.gz", url))

    print(f"== building {out} from {len(sources)} source files ==")
    total = 0
    with open(out, "w", encoding="utf-8") as fh:
        for label, kind, url in sources:
            dest = raw_dir / f"{label}.{'jsonl.gz' if kind == 'jsonl.gz' else 'txt'}"
            if not download(url, dest, label):
                continue
            n = 0
            if kind == "plain":
                data = dest.read_bytes()
                fh.write(data.decode("utf-8", errors="ignore"))
                fh.write("\n\n")
                n = len(data)
            else:
                with gzip.open(dest, "rt", encoding="utf-8", errors="ignore") as gz:
                    for line in gz:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = obj.get("text")
                        if text:
                            fh.write(text)
                            fh.write("\n\n")
                            n += len(text.encode("utf-8"))
            total += n
            print(f"  + {label}: {n/1e6:,.0f} MB")
    print(f"== done: {total/1e9:.2f} GB of text (~{total/1e6:,.0f} M tokens) ==")


if __name__ == "__main__":
    main()