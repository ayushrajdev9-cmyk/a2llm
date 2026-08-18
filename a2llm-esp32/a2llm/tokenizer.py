"""Tokenizers for A2LM.

A tokenizer maps raw text <-> token IDs.

Two implementations are provided:

* ``ByteTokenizer``  - one token per UTF-8 byte (vocab size 256). Tiny,
  deterministic, works on any text, and is the natural fit for the ESP32
  export (no vocab table has to be stored on-device).
* ``CharTokenizer``  - one token per Unicode character, vocabulary built
  from the training text and persisted to a JSON vocab file.

Both follow the same ``Tokenizer`` interface so they are drop-in
replaceable (e.g. by a BPE / SentencePiece tokenizer later).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Sequence


class Tokenizer(ABC):
    """Common interface every A2LM tokenizer must implement."""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert text into a list of token IDs."""

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        """Convert token IDs back into text."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Number of distinct token IDs."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist vocabulary metadata so the tokenizer can be reloaded."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        """Rebuild a tokenizer from its saved metadata."""


class ByteTokenizer(Tokenizer):
    """One token per UTF-8 byte: vocab is fixed to the 256 byte values.

    Encoding uses UTF-8, so any Unicode text round-trips losslessly.
    There are no unknown tokens by construction - every byte is a token.
    """

    def __init__(self) -> None:
        self._vocab_size = 256

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        for i in ids:
            if not 0 <= i < 256:
                raise ValueError(f"token id {i} out of byte range [0, 255]")
        return bytes(ids).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"type": "byte"}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        meta = json.loads(Path(path).read_text(encoding="utf-8"))
        if meta.get("type") != "byte":
            raise ValueError(f"not a byte tokenizer file: {path}")
        return cls()

    def __repr__(self) -> str:  # pragma: no cover
        return "ByteTokenizer(vocab_size=256)"


class CharTokenizer(Tokenizer):
    """One token per character, vocab learned from the training corpus.

    The vocabulary is stored with the checkpoint so decoding always matches
    the tokenization that was used during training. Characters never seen
    during fitting are mapped to an UNK token.
    """

    UNK = "<unk>"

    def __init__(self, chars: Sequence[str] | None = None) -> None:
        # "<unk>" always gets id 0; training must never see it.
        self._stoi: dict[str, int] = {self.UNK: 0}
        self._itos: list[str] = [self.UNK]
        if chars:
            self._add_chars(chars)

    def _add_chars(self, chars: Sequence[str]) -> None:
        for c in chars:
            if c not in self._stoi:
                self._stoi[c] = len(self._itos)
                self._itos.append(c)

    @classmethod
    def fit(cls, text: str) -> "CharTokenizer":
        """Build a tokenizer whose vocab covers every char in ``text``."""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self._itos)

    def encode(self, text: str) -> List[int]:
        return [self._stoi.get(c, 0) for c in text]

    def decode(self, ids: Sequence[int]) -> str:
        out = []
        for i in ids:
            if not 0 <= i < len(self._itos):
                raise ValueError(f"token id {i} out of range [0, {len(self._itos) - 1}]")
            out.append(self._itos[i])
        return "".join(out)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"type": "char", "itos": self._itos}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        meta = json.loads(Path(path).read_text(encoding="utf-8"))
        if meta.get("type") != "char":
            raise ValueError(f"not a char tokenizer file: {path}")
        t = cls()
        t._itos = list(meta["itos"])
        t._stoi = {c: i for i, c in enumerate(t._itos)}
        return t

    def __repr__(self) -> str:  # pragma: no cover
        return f"CharTokenizer(vocab_size={self.vocab_size})"


def build_tokenizer(
    tokenizer_type: str, text: str | None = None
) -> Tokenizer:
    """Factory: create the tokenizer named in the config.

    ``text`` is only needed by the ``char`` tokenizer (vocab fitting).
    """
    if tokenizer_type == "byte":
        return ByteTokenizer()
    if tokenizer_type == "char":
        if text is None:
            raise ValueError("CharTokenizer requires the corpus text to fit its vocab")
        return CharTokenizer.fit(text)
    raise ValueError(
        f"unknown tokenizer_type {tokenizer_type!r}; choose 'byte' or 'char'"
    )
