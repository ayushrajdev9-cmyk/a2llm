"""Checkpoint save / load / resume helpers.

A checkpoint is a single .pt file containing everything needed to resume
training or run inference:

    {
        "config":     full A2LMConfig dict,
        "model":      model state_dict,
        "optimizer":  optimizer state_dict (None for inference-only saves),
        "step":       training step count,
        "best_val_loss": lowest validation loss seen so far,
        "rng":        torch + python RNG states (for exact resume),
        "tokenizer":  tokenizer metadata,
    }
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .config import A2LMConfig
from .model import A2LM
from .tokenizer import ByteTokenizer, CharTokenizer, Tokenizer

LATEST_NAME = "latest.pt"
BEST_NAME = "best.pt"


def save_checkpoint(
    path: str | Path,
    cfg: A2LMConfig,
    model: A2LM,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    best_val_loss: float,
    tokenizer: Tokenizer,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Serialize a full training state to ``path`` (atomically)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt: Dict[str, Any] = {
        "config": cfg.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "best_val_loss": best_val_loss,
        "rng": {
            "torch": torch.random.get_rng_state().tolist(),
            "python": random.getstate(),
        },
        "tokenizer": _tokenizer_metadata(tokenizer),
    }
    if extra:
        ckpt.update(extra)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)  # atomic on POSIX: never leave a half-written checkpoint


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    """Load a checkpoint dict, with a clear error if it is missing/corrupt."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as e:  # corrupt file / wrong format
        raise RuntimeError(f"failed to load checkpoint {path}: {e}") from e


def load_model_for_inference(
    path: str | Path, device: str = "cpu"
) -> tuple[A2LM, Tokenizer, A2LMConfig]:
    """Load a checkpoint and return (model in eval mode, tokenizer, config)."""
    ckpt = load_checkpoint(path, map_location=device)
    cfg = A2LMConfig.from_dict(ckpt["config"])
    model = A2LM(cfg.model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tokenizer = restore_tokenizer(ckpt["tokenizer"])
    return model, tokenizer, cfg


def _tokenizer_metadata(tokenizer: Tokenizer) -> Dict[str, Any]:
    if isinstance(tokenizer, ByteTokenizer):
        return {"type": "byte"}
    if isinstance(tokenizer, CharTokenizer):
        return {"type": "char", "itos": list(tokenizer._itos)}  # noqa: SLF001
    raise TypeError(f"unsupported tokenizer type: {type(tokenizer).__name__}")


def restore_tokenizer(meta: Dict[str, Any]) -> Tokenizer:
    ttype = meta.get("type")
    if ttype == "byte":
        return ByteTokenizer()
    if ttype == "char":
        t = CharTokenizer()
        t._itos = list(meta["itos"])  # noqa: SLF001
        t._stoi = {c: i for i, c in enumerate(t._itos)}
        return t
    raise ValueError(f"unknown tokenizer metadata type: {ttype!r}")
