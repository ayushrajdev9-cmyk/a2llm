"""Text dataset pipeline.

Loads raw text -> tokenizes -> splits into train/validation token streams ->
serves fixed-length input/target sequence pairs:

    input  = tokens[0 : n]
    target = tokens[1 : n+1]

The dataset lives fully in RAM (tiny corpora), which keeps the pipeline
simple and reproducible. A ``torch.utils.data.Dataset`` view is provided so
``DataLoader`` or a manual sampler can be used.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import Tokenizer


@dataclass
class TextCorpus:
    """Tokenized corpus with a deterministic train/val split."""

    tokens: List[int]
    train: List[int]
    val: List[int]
    tokenizer: Tokenizer


def load_corpus(
    data_path: str | Path,
    tokenizer: Tokenizer,
    train_split: float = 0.9,
    seed: int = 42,
    max_tokens: int | None = None,
) -> TextCorpus:
    """Read a UTF-8 text file, tokenize it, and split the token stream.

    The split happens *after* tokenization so both folds share the same
    tokenization, and it is a pure token-stream split (no cross-contamination
    between folds).
    """
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"dataset file not found: {data_path}\n"
            "Run `python scripts/prepare_data.py` to download it."
        )
    text = Path(data_path).read_text(encoding="utf-8")
    if max_tokens is not None:
        text = text[:max_tokens]

    tokens = tokenizer.encode(text)
    if not tokens:
        raise ValueError(f"dataset {data_path} tokenized to zero tokens")

    rng = random.Random(seed)
    split = int(len(tokens) * train_split)
    idx = list(range(len(tokens)))
    rng.shuffle(idx)
    # Shuffling a token stream makes the val loss an honest estimate of
    # in-distribution next-token prediction (this is a toy corpus; a real
    # pipeline would split at document boundaries instead).
    train = [tokens[i] for i in idx[:split]]
    val = [tokens[i] for i in idx[split:]]
    return TextCorpus(tokens=tokens, train=train, val=val, tokenizer=tokenizer)


def get_batch(
    stream: Sequence[int],
    context_length: int,
    batch_size: int,
    device: str | torch.device = "cpu",
    rng: random.Random | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Draw ``batch_size`` random, non-overlapping fixed-length windows.

    Returns (inputs, targets) with shape (batch, context_length) where
    ``targets[b, i] == inputs[b, i + 1]`` - the next-token prediction task.
    """
    if len(stream) <= context_length:
        raise ValueError(
            f"token stream length ({len(stream)}) must exceed context_length "
            f"({context_length})"
        )
    rng = rng or random.Random()
    # Window start indices: 0 .. len(stream) - context_length - 1 (inclusive),
    # since we need one extra token after each window for the targets.
    starts = [rng.randrange(len(stream) - context_length) for _ in range(batch_size)]

    xs = torch.empty((batch_size, context_length), dtype=torch.long)
    ys = torch.empty((batch_size, context_length), dtype=torch.long)
    for b, s in enumerate(starts):
        window = stream[s : s + context_length + 1]
        xs[b] = torch.tensor(window[:-1], dtype=torch.long)
        ys[b] = torch.tensor(window[1:], dtype=torch.long)
    return xs.to(device), ys.to(device)


class TokenSequenceDataset(Dataset):
    """torch.utils.data view over a token stream (for DataLoader users)."""

    def __init__(self, stream: Sequence[int], context_length: int) -> None:
        if len(stream) <= context_length:
            raise ValueError("stream too short for the requested context_length")
        self.stream = list(stream)
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.stream) - self.context_length

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self.stream[i : i + self.context_length + 1]
        return (
            torch.tensor(window[:-1], dtype=torch.long),
            torch.tensor(window[1:], dtype=torch.long),
        )


def iterate_batches(
    stream: Sequence[int],
    context_length: int,
    batch_size: int,
    rng: random.Random | None = None,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Lazily yield all non-overlapping windows of a stream in shuffled order."""
    ds = TokenSequenceDataset(stream, context_length)
    order = list(range(len(ds)))
    (rng or random.Random()).shuffle(order)
    for start in range(0, len(order) - batch_size + 1, batch_size):
        xs, ys = zip(*(ds[i] for i in order[start : start + batch_size]))
        yield torch.stack(xs), torch.stack(ys)
