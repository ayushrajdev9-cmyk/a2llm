"""A2LM-0.01: a small decoder-only Transformer.

Block diagram (per input token):

    token_id -> token_embedding (+ positional embedding)
             -> [ LayerNorm -> CausalSelfAttention -> +residual ]   x num_layers
             -> [ LayerNorm -> FeedForward (GELU)   -> +residual ]
             -> LayerNorm
             -> LM head (linear) -> logits over the vocabulary

Standard choices, all explicitly coded:
  * learned token + positional embeddings
  * pre-norm residual blocks (LayerNorm before each sublayer)
  * GELU activation in the feed-forward network
  * weight tying between token embeddings and the LM head (optional)
  * dropout throughout (except the final norm / head)

The model is pure next-token prediction: forward() returns logits for the
token after every position, and training loss is cross-entropy against the
shifted target sequence.
"""

from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalMultiHeadSelfAttention
from .config import ModelConfig


class FeedForward(nn.Module):
    """Position-wise MLP: Linear -> GELU -> Linear.

    Applied identically to every position; the only source of
    cross-position interaction is attention.
    """

    def __init__(self, embedding_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(approximate="tanh"),  # tanh approx = what the ESP32 C port uses
            nn.Linear(ffn_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """One decoder block: attention sublayer + feed-forward sublayer.

    Pre-norm residual design:

        h = x + attention(LayerNorm(x))
        x = h + feedforward(LayerNorm(h))
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.embedding_dim)
        self.attn = CausalMultiHeadSelfAttention(
            cfg.embedding_dim, cfg.num_heads, cfg.dropout
        )
        self.ln2 = nn.LayerNorm(cfg.embedding_dim)
        self.ffn = FeedForward(cfg.embedding_dim, cfg.ffn_dim, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class A2LM(nn.Module):
    """Decoder-only next-token language model."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # 1. Token embeddings: vocab_size x embedding_dim lookup table.
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        # 2. Positional information: learned embedding per position index.
        self.position_embedding = nn.Embedding(cfg.context_length, cfg.embedding_dim)
        self.drop = nn.Dropout(cfg.dropout)

        # 3..9. Stack of transformer blocks.
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.num_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.embedding_dim)

        # 10. Language-model head: embedding_dim -> vocab_size logits.
        if cfg.tie_embeddings:
            # Shared weights: the head is the transposed embedding table.
            self.lm_head = nn.Linear(
                cfg.embedding_dim, cfg.vocab_size, bias=False
            )
            self.lm_head.weight = self.token_embedding.weight
        else:
            self.lm_head = nn.Linear(cfg.embedding_dim, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        # Scale residual projections (GPT-style init for stable training).
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("ffn.net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.num_layers) ** 0.5)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the model.

        idx:     (B, T) token IDs (must satisfy T <= context_length)
        targets: (B, T) next-token IDs; when given, also returns the
                 mean cross-entropy loss over all B*T positions.

        Returns (logits, loss). logits shape: (B, T, vocab_size).
        """
        B, T = idx.shape
        if T > self.cfg.context_length:
            raise ValueError(
                f"sequence length {T} exceeds context_length "
                f"{self.cfg.context_length}"
            )

        # 1+2. Token + positional embeddings, broadcast-summed.
        tok = self.token_embedding(idx)                 # (B, T, C)
        pos = self.position_embedding(
            torch.arange(T, device=idx.device)
        )                                               # (T, C)
        x = self.drop(tok + pos)

        # 3..9. Transformer blocks.
        for block in self.blocks:
            x = block(x)

        # Final normalization, then head -> logits.
        x = self.ln_f(x)
        logits = self.lm_head(x)                        # (B, T, V)

        loss = None
        if targets is not None:
            # 12. Cross-entropy: logits (B*T, V) vs targets (B*T).
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation: predict one token, append, repeat.

        idx: (B, T) prompt token IDs. Returns (B, T + max_new_tokens).
        The last context_length tokens are kept as the conditioning window.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Keep only the last context_length tokens (window slides).
            window = idx[:, -self.cfg.context_length :]
            logits, _ = self(window)                     # (B, T, V)
            logits = logits[:, -1, :] / max(temperature, 1e-8)  # last position

            if top_k is not None:
                if top_k < 1:
                    raise ValueError(f"top_k must be >= 1, got {top_k}")
                # Zero out everything outside the top-k most likely tokens.
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"A2LM(vocab={self.cfg.vocab_size}, ctx={self.cfg.context_length}, "
            f"dim={self.cfg.embedding_dim}, layers={self.cfg.num_layers}, "
            f"heads={self.cfg.num_heads}, params={self.num_parameters():,})"
        )


def build_model(cfg: ModelConfig) -> A2LM:
    """Construct an A2LM model from a ModelConfig."""
    return A2LM(cfg)
