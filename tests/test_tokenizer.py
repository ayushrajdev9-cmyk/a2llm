"""Tokenizer tests: round-trip, unknowns, save/load, modularity."""

import pytest

from a2lm.tokenizer import ByteTokenizer, CharTokenizer


TEXT = "Hello, A2LM! Héllo wörld 🌍 — \n\t\"quotes\""


class TestByteTokenizer:
    def test_roundtrip(self):
        t = ByteTokenizer()
        ids = t.encode(TEXT)
        assert t.decode(ids) == TEXT

    def test_vocab_size(self):
        assert ByteTokenizer().vocab_size == 256

    def test_ascii_bytes_are_identity(self):
        t = ByteTokenizer()
        assert t.encode("abc") == [97, 98, 99]

    def test_out_of_range_decode_raises(self):
        t = ByteTokenizer()
        with pytest.raises(ValueError):
            t.decode([300])

    def test_save_load(self, tmp_path):
        p = tmp_path / "byte.json"
        t = ByteTokenizer()
        t.save(p)
        t2 = ByteTokenizer.load(p)
        assert isinstance(t2, ByteTokenizer)
        assert t2.vocab_size == 256
        assert t2.encode("x") == [120]


class TestCharTokenizer:
    def test_roundtrip(self):
        t = CharTokenizer.fit(TEXT)
        assert t.decode(t.encode(TEXT)) == TEXT

    def test_vocab_covers_corpus(self):
        t = CharTokenizer.fit("abcd")
        assert t.vocab_size == 4 + 1  # + <unk>
        assert t.encode("a") == [1]

    def test_unknown_token_is_unk(self):
        t = CharTokenizer.fit("abc")
        assert t.encode("xyz") == [0, 0, 0]  # <unk> id is 0

    def test_save_load(self, tmp_path):
        p = tmp_path / "char.json"
        t = CharTokenizer.fit("hello world")
        t.save(p)
        t2 = CharTokenizer.load(p)
        assert t2._itos == t._itos  # noqa: SLF001
        assert t2.decode(t2.encode("hello world")) == "hello world"

    def test_load_rejects_wrong_type(self, tmp_path):
        p = tmp_path / "wrong.json"
        ByteTokenizer().save(p)
        with pytest.raises(ValueError):
            CharTokenizer.load(p)