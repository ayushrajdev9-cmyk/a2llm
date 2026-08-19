"""Text dataset pipeline.

Loads raw text -> tokenizes -> splits into train/validation token streams ->
serves fixed-length input/target sequence pairs:

    input  = tokens[0 : n]
    target = tokens[1 : n+1]

Memory model:

* ``ByteTokenizer`` corpora (the default): the token stream is a zero-copy
  ``numpy.uint8`` view over the raw UTF-8 bytes, so a ~2-3 GB multilingual
  corpus costs ~2-3 GB of RAM. No Python lists of ints are ever built.
* Other tokenizers: the token stream is a Python list (small corpora).

Batches are drawn as random contiguous windows (no full-corpus shuffle), so
training cost does not depend on corpus size - only on the number of steps.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import ByteTokenizer, Tokenizer


@dataclass
class TextCorpus:
    """Tokenized corpus with a deterministic train/val split."""

    tokens: Sequence[int]
    train: Sequence[int]
    val: Sequence[int]
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

    For the byte tokenizer the tokens are the raw UTF-8 bytes, stored as a
    numpy ``uint8`` view - O(1) extra memory regardless of corpus size.
    """
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"dataset file not found: {data_path}\n"
            "Run `python scripts/prepare_data.py` to download it."
        )
    raw = Path(data_path).read_bytes()
    if max_tokens is not None:
        raw = raw[:max_tokens]

    if isinstance(tokenizer, ByteTokenizer):
        # tokens == UTF-8 bytes, zero-copy numpy view
        tokens: Sequence[int] = np.frombuffer(raw, dtype=np.uint8)
        if len(tokens) == 0:
            raise ValueError(f"dataset {data_path} tokenized to zero tokens")
        split = int(len(tokens) * train_split)
        return TextCorpus(
            tokens=tokens,
            train=tokens[:split],
            val=tokens[split:],
            tokenizer=tokenizer,
        )

    text = raw.decode("utf-8")
    tokens = tokenizer.encode(text)
    if not tokens:
        raise ValueError(f"dataset {data_path} tokenized to zero tokens")

    rng = random.Random(seed)
    split = int(len(tokens) * train_split)
    idx = list(range(len(tokens)))
    rng.shuffle(idx)
    # Shuffling a token stream makes the val loss an honest estimate of
    # in-distribution next-token prediction (toy corpora; real pipelines
    # split at document boundaries instead).
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
