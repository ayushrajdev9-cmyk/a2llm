"""Small utilities: parameter counting, model-size reporting, metrics.

All numbers here are computed from the actual model/run - never fabricated.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch

from .model import A2LM


def count_parameters(model: A2LM) -> Dict[str, int]:
    """Total / trainable / non-trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }


def size_bytes(num_params: int, bytes_per_param: int) -> int:
    """On-disk/in-memory size estimate for a given precision."""
    return num_params * bytes_per_param


def format_bytes(n: int) -> str:
    """Human-readable byte count."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


def model_report(model: A2LM) -> Dict[str, Any]:
    """Full size report for a model."""
    counts = count_parameters(model)
    n = counts["total"]
    return {
        **counts,
        "fp32_size_bytes": size_bytes(n, 4),
        "fp16_size_bytes": size_bytes(n, 2),
        "int8_size_bytes": size_bytes(n, 1),
        "int8_size_human": format_bytes(size_bytes(n, 1)),
        "fp32_size_human": format_bytes(size_bytes(n, 4)),
    }


class TokensPerSecond:
    """Simple elapsed-time / token-throughput tracker."""

    def __init__(self) -> None:
        self._t0: float | None = None
        self._tokens = 0

    def start(self) -> None:
        self._t0 = time.time()

    def add(self, tokens: int) -> None:
        self._tokens += tokens

    @property
    def rate(self) -> float:
        if self._t0 is None:
            return 0.0
        dt = time.time() - self._t0
        return self._tokens / dt if dt > 0 else 0.0


def gpu_memory_mb() -> float | None:
    """Allocated CUDA memory in MB, or None when no CUDA is in use."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return None


def write_experiment_meta(
    exp_dir: Path, config_dict: Dict[str, Any], metrics: Dict[str, Any]
) -> None:
    """Persist config + metrics for an experiment run."""
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.json").write_text(
        json.dumps(config_dict, indent=2), encoding="utf-8"
    )
    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )