"""Overfit test: a tiny model MUST memorize a tiny dataset.

This is the single most important correctness test: if the architecture or
the training loop is broken, the model cannot drive cross-entropy loss
towards ~0 on a small fixed batch.
"""

import torch

from a2llm.config import ModelConfig
from a2llm.model import A2LM


def test_model_overfits_tiny_batch():
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=32, context_length=16, embedding_dim=64,
        num_layers=2, num_heads=4, dropout=0.0,
    )
    model = A2LM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)

    xs = torch.randint(0, 32, (8, 16))
    ys = torch.randint(0, 32, (8, 16))

    first_loss = None
    for step in range(400):
        opt.zero_grad()
        _, loss = model(xs, ys)
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = loss.item()

    assert first_loss > 1.0, "first loss should be well above zero"
    assert loss.item() < 0.05, (
        f"model failed to overfit: final loss {loss.item():.4f} "
        "(expect < 0.05 after 400 steps on a fixed batch)"
    )