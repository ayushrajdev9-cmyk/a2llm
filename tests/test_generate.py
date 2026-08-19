"""Generation tests: determinism, limits, temperature, top-k, top-p, stop."""

import torch

from a2llm.config import ModelConfig
from a2llm.generate import generate_text
from a2llm.model import A2LM
from a2llm.tokenizer import ByteTokenizer


def make_model() -> A2LM:
    torch.manual_seed(0)
    return A2LM(
        ModelConfig(
            vocab_size=256, context_length=32, embedding_dim=32,
            num_layers=2, num_heads=2, dropout=0.0,
        )
    )


class TestGeneration:
    def test_max_token_limit(self):
        m = make_model()
        tok = ByteTokenizer()
        ids = torch.tensor([tok.encode("ROMEO:")], dtype=torch.long)
        out = m.generate(ids, max_new_tokens=25, temperature=0.9, top_k=40)
        assert out.shape[1] == ids.shape[1] + 25

    def test_deterministic_with_seed(self):
        m = make_model()
        tok = ByteTokenizer()
        a = generate_text(m, tok, "hello", max_new_tokens=30, seed=99)
        b = generate_text(m, tok, "hello", max_new_tokens=30, seed=99)
        assert a == b

    def test_low_temperature_reproducible_same_seed(self):
        m = make_model()
        tok = ByteTokenizer()
        a = generate_text(m, tok, "the", max_new_tokens=20, temperature=0.01, seed=5)
        b = generate_text(m, tok, "the", max_new_tokens=20, temperature=0.01, seed=5)
        assert a == b  # same seed + same temperature = same output

    def test_top_k_and_top_p_accepted(self):
        m = make_model()
        tok = ByteTokenizer()
        out = generate_text(
            m, tok, "ROMEO:", max_new_tokens=15, temperature=0.9,
            top_k=20, top_p=0.9, seed=7,
        )
        assert len(out) > 0

    def test_stop_tokens_early_exit(self):
        m = make_model()
        tok = ByteTokenizer()
        # stop at newline (byte 10): output must not contain a newline
        out = generate_text(
            m, tok, "abc", max_new_tokens=50, temperature=0.5,
            top_k=5, stop_tokens=[10], seed=3,
        )
        assert "\n" not in out

    def test_invalid_temperature_raises(self):
        m = make_model()
        tok = ByteTokenizer()
        try:
            generate_text(m, tok, "x", temperature=0.0)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    def test_long_prompt_is_truncated_to_context(self):
        m = make_model()
        tok = ByteTokenizer()
        long_prompt = "x" * 100
        out = generate_text(m, tok, long_prompt, max_new_tokens=5, seed=1)
        assert len(out) == 5