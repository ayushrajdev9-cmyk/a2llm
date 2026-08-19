"""Dataset tests: sequence creation, input/target shift, split."""

import random

import pytest

from a2llm.dataset import TokenSequenceDataset, get_batch, iterate_batches, load_corpus
from a2llm.tokenizer import ByteTokenizer


def make_corpus(tmp_path, text="abcdefghijklmnopqrstuvwxyz"):
    p = tmp_path / "corpus.txt"
    p.write_text(text, encoding="utf-8")
    return p


class TestSequenceCreation:
    def test_target_is_input_shifted_by_one(self):
        stream = [0, 1, 2, 3, 4, 5, 6, 7]
        xs, ys = get_batch(stream, context_length=4, batch_size=2, rng=random.Random(0))
        # every sample must satisfy ys[b, i] == xs[b, i+1]
        for b in range(2):
            assert (ys[b, :-1] == xs[b, 1:]).all()

    def test_window_lengths(self):
        stream = [0] * 20
        xs, ys = get_batch(stream, context_length=6, batch_size=3, rng=random.Random(1))
        assert xs.shape == (3, 6)
        assert ys.shape == (3, 6)

    def test_dataset_len_and_getitem(self):
        ds = TokenSequenceDataset(list(range(10)), context_length=4)
        assert len(ds) == 6  # 10 - 4 windows
        x, y = ds[0]
        assert x.tolist() == [0, 1, 2, 3]
        assert y.tolist() == [1, 2, 3, 4]

    def test_too_short_stream_raises(self):
        with pytest.raises(ValueError):
            get_batch([1, 2, 3], context_length=4, batch_size=1)

    def test_iterate_batches_covers_all_windows(self):
        ds = TokenSequenceDataset(list(range(10)), context_length=4)
        n = 0
        for xs, ys in iterate_batches(list(range(10)), 4, 3, rng=random.Random(0)):
            n += xs.shape[0]
        assert n == len(ds)


class TestSplit:
    def test_train_val_partition(self, tmp_path):
        p = make_corpus(tmp_path)
        t = ByteTokenizer()
        corpus = load_corpus(p, t, train_split=0.8, seed=7)
        assert len(corpus.train) + len(corpus.val) == len(corpus.tokens)
        assert len(corpus.train) == 20  # 26 tokens * 0.8
        assert len(corpus.val) == 6

    def test_deterministic_split(self, tmp_path):
        p = make_corpus(tmp_path)
        t = ByteTokenizer()
        a = load_corpus(p, t, seed=3)
        b = load_corpus(p, t, seed=3)
        # numpy-backed streams for the byte tokenizer
        assert list(a.train) == list(b.train) and list(a.val) == list(b.val)

    def test_missing_file_raises(self, tmp_path):
        t = ByteTokenizer()
        with pytest.raises(FileNotFoundError):
            load_corpus(tmp_path / "nope.txt", t)