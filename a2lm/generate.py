"""Autoregressive text generation (sampling + CLI helpers).

Sampling strategy:

    1. Model logits for the last position.
    2. Divide by temperature (higher = more random, lower = greedier).
    3. top-k: keep only the k most probable tokens.
    4. top-p (nucleus): keep the smallest set of tokens whose cumulative
       probability exceeds p (applied after top-k, if both are set).
    5. Sample one token, append, repeat until max_new_tokens or a stop token.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import A2LM
from .tokenizer import Tokenizer


@torch.no_grad()
def generate_text(
    model: A2LM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    stop_tokens: list[int] | None = None,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> str:
    """Generate a continuation for ``prompt`` and return it as a string.

    The prompt itself is not included in the returned text. ``stop_tokens``
    are token IDs that halt generation early (e.g. a newline byte 10).
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    if seed is not None:
        torch.manual_seed(seed)

    ids = tokenizer.encode(prompt)
    if len(ids) > model.cfg.context_length:
        ids = ids[-model.cfg.context_length :]
    if not ids:
        raise ValueError("prompt encoded to zero tokens")

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    model.eval()
    generated: list[int] = []
    stop = set(stop_tokens or [])

    for _ in range(max_new_tokens):
        window = idx[:, -model.cfg.context_length :]
        logits, _ = model(window)                            # (1, T, V)
        logits = logits[:, -1, :] / temperature              # (1, V)

        if top_k is not None:
            if top_k < 1:
                raise ValueError(f"top_k must be >= 1, got {top_k}")
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)                    # (1, V)
        if top_p is not None:
            sorted_p, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_p, dim=-1)
            # Keep the smallest prefix covering >= top_p of the mass.
            keep = cum <= top_p
            keep[..., 0] = True                              # always keep the top token
            masked = torch.zeros_like(probs)
            masked.scatter_(-1, sorted_idx, sorted_p * keep)
            probs = masked
            if probs.sum() == 0:
                probs = F.softmax(logits, dim=-1)

        next_id = torch.multinomial(probs, num_samples=1)    # (1, 1)
        tok = int(next_id.item())
        generated.append(tok)
        if tok in stop:
            break
        idx = torch.cat([idx, next_id], dim=1)

    return tokenizer.decode(generated)