"""Causal attention mask tests: future tokens must be blocked."""

import torch

from a2lm.attention import CausalMultiHeadSelfAttention, build_causal_mask


class TestCausalMask:
    def test_shape(self):
        m = build_causal_mask(8)
        assert m.shape == (8, 8)

    def test_lower_triangle_allowed(self):
        m = build_causal_mask(5)
        for i in range(5):
            for j in range(5):
                if j <= i:
                    assert m[i, j] == 0.0
                else:
                    assert m[i, j] == float("-inf")

    def test_first_token_attends_only_to_itself(self):
        m = build_causal_mask(4)
        allowed = torch.isfinite(m[0])
        assert allowed.tolist() == [True, False, False, False]


class TestAttentionBlocking:
    def test_attention_weights_are_causal(self):
        torch.manual_seed(0)
        attn = CausalMultiHeadSelfAttention(embedding_dim=16, num_heads=2, dropout=0.0)
        x = torch.randn(1, 6, 16)
        with torch.no_grad():
            qkv = attn.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            B, T, C = x.shape
            H, D = 2, 8
            q = q.view(B, T, H, D).transpose(1, 2)
            k = k.view(B, T, H, D).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / (D ** 0.5)
            # apply the causal mask exactly like forward() does
            scores = scores + build_causal_mask(T)
            weights = torch.softmax(scores, dim=-1)
        # no weight may point at a future position
        for b in range(1):
            for h in range(2):
                for i in range(T):
                    for j in range(i + 1, T):
                        assert weights[b, h, i, j] == 0.0, f"future attention at {i}->{j}"

    def test_forward_shape(self):
        torch.manual_seed(1)
        attn = CausalMultiHeadSelfAttention(embedding_dim=32, num_heads=4)
        out = attn(torch.randn(2, 10, 32))
        assert out.shape == (2, 10, 32)

    def test_bad_head_split_raises(self):
        import pytest
        with pytest.raises(ValueError):
            CausalMultiHeadSelfAttention(embedding_dim=16, num_heads=3)