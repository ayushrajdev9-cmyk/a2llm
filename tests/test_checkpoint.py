"""Checkpoint tests: save/load, identical outputs, metadata, resume fields."""

import torch

from a2llm.checkpoint import (
    BEST_NAME, LATEST_NAME, load_checkpoint, load_model_for_inference,
    save_checkpoint,
)
from a2llm.config import A2LMConfig
from a2llm.model import A2LM
from a2llm.tokenizer import ByteTokenizer
from a2llm.utils import count_parameters


def make_cfg() -> A2LMConfig:
    cfg = A2LMConfig()
    cfg.model.vocab_size = 256
    cfg.model.context_length = 32
    cfg.model.embedding_dim = 32
    cfg.model.num_layers = 2
    cfg.model.num_heads = 2
    cfg.model.dropout = 0.0  # determinism: no dropout noise
    cfg.run_name = "test-run"
    return cfg


class TestCheckpoint:
    def test_save_and_load_roundtrip(self, tmp_path):
        torch.manual_seed(0)
        cfg = make_cfg()
        model = A2LM(cfg.model)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        tok = ByteTokenizer()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, cfg, model, opt, step=123, best_val_loss=3.5, tokenizer=tok)

        ckpt = load_checkpoint(path)
        assert ckpt["step"] == 123
        assert ckpt["best_val_loss"] == 3.5
        assert ckpt["tokenizer"]["type"] == "byte"
        assert ckpt["config"]["run_name"] == "test-run"

    def test_identical_outputs_after_reload(self, tmp_path):
        torch.manual_seed(1)
        cfg = make_cfg()
        model = A2LM(cfg.model)
        tok = ByteTokenizer()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, cfg, model, None, step=0, best_val_loss=9.9, tokenizer=tok)

        m2, tok2, cfg2 = load_model_for_inference(path)
        idx = torch.randint(0, 256, (2, 16))
        with torch.no_grad():
            model.eval()  # match eval mode of the reloaded model
            logits1, _ = model(idx)
            logits2, _ = m2(idx)
        assert torch.equal(logits1, logits2)
        assert isinstance(tok2, ByteTokenizer)
        assert cfg2.model.embedding_dim == 32

    def test_missing_checkpoint_raises(self, tmp_path):
        with self._expect(FileNotFoundError):
            load_checkpoint(tmp_path / "ghost.pt")

    def test_garbage_checkpoint_raises(self, tmp_path):
        p = tmp_path / "bad.pt"
        p.write_bytes(b"this is not a torch file")
        with self._expect(RuntimeError):
            load_checkpoint(p)

    def test_optimizer_state_roundtrip(self, tmp_path):
        torch.manual_seed(2)
        cfg = make_cfg()
        model = A2LM(cfg.model)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # take one real step so the optimizer has momentum state
        x = torch.randint(0, 256, (2, 16))
        _, loss = model(x, x)
        opt.zero_grad(); loss.backward(); opt.step()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, cfg, model, opt, step=1, best_val_loss=8.8, tokenizer=ByteTokenizer())
        ckpt = load_checkpoint(path)
        assert ckpt["optimizer"] is not None
        assert len(ckpt["optimizer"]["state"]) > 0

    def test_param_count_utility(self):
        cfg = make_cfg()
        r = count_parameters(A2LM(cfg.model))
        assert r["total"] == r["trainable"] + r["non_trainable"]
        assert r["trainable"] > 0

    def test_best_latest_names_are_distinct(self):
        assert LATEST_NAME != BEST_NAME

    class _expect:
        def __init__(self, exc):
            import pytest
            self._ctx = pytest.raises(exc)
        def __enter__(self):
            return self._ctx.__enter__()
        def __exit__(self, *a):
            return self._ctx.__exit__(*a)