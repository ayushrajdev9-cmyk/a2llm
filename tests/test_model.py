"""Model tests: forward pass, output dimensions, causality, ties, CPU."""

import torch

from a2llm.config import ModelConfig
from a2llm.model import A2LM


def make_cfg(**kw) -> ModelConfig:
    base = dict(
        vocab_size=64, context_length=16, embedding_dim=32,
        num_layers=2, num_heads=4, dropout=0.0,
    )
    base.update(kw)
    return ModelConfig(**base)


class TestForward:
    def test_output_dimensions(self):
        m = A2LM(make_cfg())
        idx = torch.randint(0, 64, (3, 12))
        logits, loss = m(idx, idx)  # targets = same ids is fine for shape checks
        assert logits.shape == (3, 12, 64)
        assert loss.shape == torch.Size([])

    def test_forward_without_targets(self):
        m = A2LM(make_cfg())
        logits, loss = m(torch.randint(0, 64, (2, 8)))
        assert loss is None
        assert logits.shape == (2, 8, 64)

    def test_context_overflow_raises(self):
        m = A2LM(make_cfg())
        with pytest_raises(ValueError):
            m(torch.randint(0, 64, (1, 17)))

    def test_parameter_count_matches_formula(self):
        cfg = make_cfg()
        m = A2LM(cfg)
        assert m.num_parameters() == cfg.num_parameters()

    def test_embedding_tied_head(self):
        m = A2LM(make_cfg(tie_embeddings=True))
        assert m.lm_head.weight is m.token_embedding.weight
        m2 = A2LM(make_cfg(tie_embeddings=False))
        assert m2.lm_head.weight is not m2.token_embedding.weight

    def test_cpu_inference_and_loss_decreases_after_step(self):
        torch.manual_seed(0)
        cfg = make_cfg()
        m = A2LM(cfg)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        idx = torch.randint(0, 64, (4, 16))
        _, l1 = m(idx, idx)
        opt.zero_grad(); l1.backward(); opt.step()
        _, l2 = m(idx, idx)
        assert l2.item() < l1.item()


def pytest_raises(exc):
    import pytest
    return pytest.raises(exc)