#!/usr/bin/env python3
"""Build a large multilingual pretraining corpus (billions of tokens).

Sources (byte tokenizer => tokens == UTF-8 bytes):

  en: wikitext-103-raw (Wikipedia, parquet) + TinyStories (simple stories)
  hi: Common Crawl (mc4 / c4) Hindi  - Devanagari
  ur: Common Crawl (mc4 / c4) Urdu
  bn: Common Crawl (mc4 / c4) Bengali
  es/fr: optional via --extra-es-fr

Default corpus: ~1.7-2 GB of text (~1.7-2 B tokens). Downloads are cached in
data/raw/ and skipped on re-runs, so re-running only fetches what is missing.

Usage:
    python scripts/prepare_data_large.py
    python scripts/prepare_data_large.py --extra-es-fr   # + more languages
    python scripts/prepare_data_large.py --out data/pretrain.txt
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.request
from pathlib import Path

WIKITEXT_PARQUET = [
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
    "wikitext-103-raw-v1/train-0000{i}-of-00002.parquet".format(i=i)
    for i in range(2)
]
TINYSTORIES = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
               "resolve/main/TinyStories-train.txt")
MC4 = ("https://huggingface.co/datasets/allenai/c4/resolve/main/"
       "multilingual/c4-{lang}.tfrecord-{i:05d}-of-{total:05d}.json.gz")

# (lang, n_shards, total_shards) - sized so total CC text is ~1.5 GB
# (measured: 1 hi shard = ~132 MB of text)
LANG_SHARDS = {
    "hi": (6, 1024),
    "ur": (4, 128),
    "bn": (4, 512),
}
EXTRA_SHARDS = {"es": (16, 1024), "fr": (16, 1024)}  # only with --extra-es-fr


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


def write_parquet_text(dest: Path, fh) -> int:
    """Extract the 'text' column of a parquet file as plain text."""
    try:
        import pandas as pd
    except ImportError:
        print("  [skip] pandas/pyarrow not installed - pip install pandas pyarrow",
              file=sys.stderr)
        return 0
    df = pd.read_parquet(dest)
    n = 0
    for text in df["text"].astype(str):
        fh.write(text)
        fh.write("\n\n")
        n += len(text.encode("utf-8"))
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a multilingual pretraining corpus")
    ap.add_argument("--out", default="data/pretrain_multilingual.txt")
    ap.add_argument("--extra-es-fr", action="store_true",
                    help="also pull Spanish + French shards")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = Path("data/raw")
    langs = dict(LANG_SHARDS)
    if args.extra_es_fr:
        langs.update(EXTRA_SHARDS)

    sources: list[tuple[str, str, str]] = []  # (label, kind, url)
    for i, url in enumerate(WIKITEXT_PARQUET):
        sources.append((f"en-wikitext103-{i}", "parquet", url))
    sources.append(("en-tinystories", "plain", TINYSTORIES))
    for lang, (n, total) in langs.items():
        for i in range(n):
            sources.append((f"{lang}-{i:02d}", "jsonl.gz",
                            MC4.format(lang=lang, i=i, total=total)))

    print(f"== building {out} from {len(sources)} source files ==")
    total = 0
    with open(out, "w", encoding="utf-8") as fh:
        for label, kind, url in sources:
            ext = {"parquet": ".parquet", "plain": ".txt", "jsonl.gz": ".jsonl.gz"}[kind]
            dest = raw_dir / f"{label}{ext}"
            if not download(url, dest, label):
                continue
            n = 0
            try:
                if kind == "parquet":
                    n = write_parquet_text(dest, fh)
                elif kind == "plain":
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
            except Exception as exc:  # one bad source must not kill the build
                print(f"  [skip] {label}: {exc}", file=sys.stderr)
                continue
            total += n
            print(f"  + {label}: {n/1e6:,.0f} MB")
    print(f"== done: {total/1e9:.2f} GB of text (~{total/1e6:,.0f} M tokens) ==")
    append_distilled(out)


def append_distilled(out: str) -> None:
    """Append uran1um1/gpt-oss_20b_distilled QA pairs (MIT license) to the corpus.

    Idempotent: skips if the marker line is already present, so re-runs
    never double-append the distilled data.
    """
    marker = "# === gpt-oss_20b_distilled (MIT) ==="
    if marker in open(out, encoding="utf-8").read():
        print("  [skip] distilled QA: already in corpus")
        return
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [skip] distilled QA: `pip install datasets` to enable")
        return
    try:
        ds = load_dataset("uran1um1/gpt-oss_20b_distilled", split="train")
        n = 0
        with open(out, "a", encoding="utf-8") as fh:
            fh.write("\n\n" + marker + "\n\n")
            for row in ds:
                for key in ("user", "assistant"):
                    t = row.get(key)
                    if isinstance(t, dict):  # dataset stores {"content": "..."}
                        t = t.get("content")
                    if isinstance(t, str) and t.strip():
                        fh.write(t)
                        fh.write("\n\n")
                        n += len(t.encode("utf-8"))
        print(f"  + distilled-qa (gpt-oss_20b, MIT): {n/1e6:,.0f} MB (~{n/1e6:.0f} M tokens)")
    except Exception as exc:  # never break the corpus build
        print(f"  [skip] distilled QA: {exc}")
    append_distilled(Path(out))


def append_distilled(out: Path) -> None:
    """Append gpt-oss_20b_distilled QA pairs (MIT licensed) to the corpus."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [skip] distilled QA: pip install datasets", file=sys.stderr)
        return
    try:
        ds = load_dataset("uran1um1/gpt-oss_20b_distilled", split="train")
        n = 0
        with open(out, "a", encoding="utf-8") as fh:
            for row in ds:
                for key in ("user", "assistant"):
                    t = row.get(key)
                    if isinstance(t, str) and t.strip():
                        fh.write(t)
                        fh.write("\n\n")
                        n += len(t.encode("utf-8"))
        print(f"  + distilled-qa (gpt-oss_20b, MIT): {n/1e6:,.0f} MB (~{n/1e6:.0f} M tokens)")
    except Exception as exc:  # never break the corpus build
        print(f"  [skip] distilled QA: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()