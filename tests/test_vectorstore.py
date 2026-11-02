from docrag.models import Chunk
from docrag.vectorstore.numpy_store import NumpyStore
from tests.conftest import FakeEmbedder


def _chunks(texts):
    return [Chunk(id=str(i), doc="d.pdf", page=i + 1, text=t) for i, t in enumerate(texts)]


def test_search_ranks_relevant_chunk_first():
    texts = ["paint coating thickness", "compressor screw speed", "network cable pinout"]
    emb = FakeEmbedder()
    store = NumpyStore()
    store.add(_chunks(texts), emb.embed_documents(texts))
    hits = store.search(emb.embed_query("screw speed"), 2)
    assert hits[0].chunk.text == "compressor screw speed"
    assert hits[0].score >= hits[1].score


def test_save_and_load_roundtrip(tmp_path):
    texts = ["alpha", "beta gamma"]
    emb = FakeEmbedder()
    store = NumpyStore()
    store.add(_chunks(texts), emb.embed_documents(texts))
    store.save(tmp_path)
    loaded = NumpyStore.load(tmp_path)
    assert len(loaded) == 2
    assert loaded.search(emb.embed_query("beta"), 1)[0].chunk.page == 2


def test_empty_store_returns_no_hits():
    assert NumpyStore().search(FakeEmbedder().embed_query("x"), 3) == []
