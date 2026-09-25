import numpy as np

from docrag.embeddings.cache import CachedEmbedder
from tests.conftest import FakeEmbedder


def test_second_run_makes_no_embedding_calls(tmp_path):
    inner = FakeEmbedder()
    cached = CachedEmbedder(inner, tmp_path / "c.sqlite")
    texts = ["alpha beta", "gamma delta", "alpha beta"]
    first = cached.embed_documents(texts)
    assert inner.calls == 2  # duplicate text embedded once
    second = cached.embed_documents(texts)
    assert inner.calls == 2
    assert np.array_equal(first, second)


def test_only_new_texts_are_embedded_and_order_is_kept(tmp_path):
    inner = FakeEmbedder()
    cached = CachedEmbedder(inner, tmp_path / "c.sqlite")
    cached.embed_documents(["one", "two"])
    assert cached.count_uncached(["one", "three"]) == 1
    out = cached.embed_documents(["three", "one"])
    assert inner.calls == 3
    assert np.array_equal(out, inner.embed_documents(["three", "one"]))


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "c.sqlite"
    CachedEmbedder(FakeEmbedder(), path).embed_documents(["persist me"])
    assert CachedEmbedder(FakeEmbedder(), path).count_uncached(["persist me"]) == 0
