"""Store contract tests, run against NumpyStore always and PgVectorStore when DATABASE_URL is set.

    docker compose up -d
    DATABASE_URL=postgresql://docrag:docrag@localhost:5433/docrag python -m pytest -m integration

Without DATABASE_URL the Postgres cases are skipped, so the default suite needs no database.
"""

import os
import uuid
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from docrag import cli
from docrag.config import Settings
from docrag.models import Chunk, Page
from docrag.vectorstore.numpy_store import NumpyStore
from tests.conftest import FakeEmbedder

DATABASE_URL = os.environ.get("DATABASE_URL")
needs_db = pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set (needs Postgres with pgvector)")
integration = [pytest.mark.integration, needs_db]


@pytest.fixture
def pg_factory():
    from docrag.vectorstore.pgvector_store import PgVectorStore

    opened = []

    def make(name: str | None = None) -> "PgVectorStore":
        store = PgVectorStore(DATABASE_URL, name or f"test_{uuid.uuid4().hex}")
        opened.append(store)
        return store

    yield make
    for store in opened:
        store.reset()
        store.close()


@pytest.fixture(params=["numpy", pytest.param("pgvector", marks=integration)])
def new_store(request):
    if request.param == "numpy":
        return NumpyStore
    return request.getfixturevalue("pg_factory")


def _chunks(specs):
    return [Chunk(id=f"c{i}", doc=doc, page=i + 1, text=text) for i, (doc, text) in enumerate(specs)]


def _filled(new_store, specs):
    emb = FakeEmbedder()
    store = new_store()
    chunks = _chunks(specs)
    store.add(chunks, emb.embed_documents([c.text for c in chunks]))
    return store, emb


SPECS = [
    ("a.pdf", "cable insulation thickness"),
    ("a.pdf", "extruder screw speed"),
    ("b.pdf", "network cable pinout"),
    ("b.pdf", "screw terminal torque"),
]


def test_search_ranks_relevant_chunk_first_and_scores_descend(new_store):
    store, emb = _filled(new_store, SPECS)
    hits = store.search(emb.embed_query("extruder screw speed"), 3)
    assert hits[0].chunk.text == "extruder screw speed"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_score_is_cosine_similarity(new_store):
    store = new_store()
    store.add(_chunks([("a.pdf", "x"), ("a.pdf", "y")]), np.array([[3.0, 0.0], [0.0, 2.0]]))
    hits = store.search(np.array([5.0, 0.0]), 2)  # lengths must not matter, only direction
    assert hits[0].chunk.id == "c0" and hits[0].score == pytest.approx(1.0, abs=1e-6)
    assert hits[1].score == pytest.approx(0.0, abs=1e-6)


def test_device_filter_restricts_and_returns_fewer_than_k(new_store):
    store, emb = _filled(new_store, SPECS)
    query = emb.embed_query("screw")
    only_b = store.search(query, 4, {"b.pdf"})
    assert {h.chunk.doc for h in only_b} == {"b.pdf"} and len(only_b) == 2
    assert store.search(query, 4, {"nothing.pdf"}) == []
    assert store.search(query, 4, set()) == []


def test_k_larger_than_the_store_returns_everything_and_zero_returns_nothing(new_store):
    store, emb = _filled(new_store, SPECS)
    assert len(store.search(emb.embed_query("cable"), 50)) == 4
    assert store.search(emb.embed_query("cable"), 0) == []


def test_empty_store(new_store):
    store = new_store()
    assert store.search(FakeEmbedder().embed_query("x"), 3) == []
    assert len(store) == 0 and store.docs() == set() and store.chunks() == []


def test_docs_chunks_and_len_keep_insertion_order(new_store):
    store, _ = _filled(new_store, SPECS)
    assert len(store) == 4 and store.docs() == {"a.pdf", "b.pdf"}
    assert [c.id for c in store.chunks()] == ["c0", "c1", "c2", "c3"]


def test_chunk_fields_survive_including_section_and_unicode(new_store):
    store = new_store()
    chunk = Chunk(id="x", doc="d.pdf", page=7, text="Level ‣ 0…1 °C", section="1. Intro > 1.2 Scope")
    bare = Chunk(id="y", doc="d.pdf", page=8, text="plain")
    store.add([chunk, bare], np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert store.chunks() == [chunk, bare]
    assert store.search(np.array([1.0, 0.0]), 1)[0].chunk == chunk


def test_add_rejects_mismatched_lengths(new_store):
    with pytest.raises(ValueError):
        new_store().add(_chunks([("a.pdf", "x")]), np.zeros((2, 3)))


def test_add_can_be_called_more_than_once(new_store):
    store = new_store()
    store.add(_chunks([("a.pdf", "x")]), np.array([[1.0, 0.0]]))
    store.add([Chunk(id="z", doc="b.pdf", page=1, text="z")], np.array([[0.0, 1.0]]))
    assert len(store) == 2 and store.docs() == {"a.pdf", "b.pdf"}


# --- Postgres only -------------------------------------------------------------------------


@pytest.mark.integration
@needs_db
def test_ties_come_back_in_insertion_order(pg_factory):
    store = pg_factory()
    same = np.array([[1.0, 0.0]] * 4)
    store.add(_chunks([("a.pdf", str(i)) for i in range(4)]), same)
    assert [h.chunk.id for h in store.search(np.array([1.0, 0.0]), 4)] == ["c0", "c1", "c2", "c3"]


@pytest.mark.integration
@needs_db
def test_indexes_with_different_names_do_not_see_each_other(pg_factory):
    one, two = pg_factory(), pg_factory()
    one.add(_chunks([("a.pdf", "x")]), np.array([[1.0, 0.0]]))
    two.add([Chunk(id="c0", doc="b.pdf", page=1, text="y")], np.array([[1.0, 0.0, 0.0]]))  # same id, other dimension
    assert one.docs() == {"a.pdf"} and two.docs() == {"b.pdf"}
    assert len(one) == 1 and len(two) == 1
    assert one.search(np.array([1.0, 0.0]), 5)[0].chunk.doc == "a.pdf"


@pytest.mark.integration
@needs_db
def test_reset_replaces_an_index_and_reopening_finds_the_rows(pg_factory):
    from docrag.vectorstore.pgvector_store import PgVectorStore

    store = pg_factory()
    store.add(_chunks([("a.pdf", "x"), ("a.pdf", "y")]), np.eye(2))
    reopened = PgVectorStore(DATABASE_URL, store._index)  # a second connection sees committed rows
    try:
        assert len(reopened) == 2
    finally:
        reopened.close()
    store.reset()
    assert len(store) == 0
    store.add(_chunks([("a.pdf", "x")]), np.eye(2)[:1])  # ids can be reused after a reset
    assert len(store) == 1


REAL_INDEX = Path("data/index/openai_text-embedding-3-small_c800o150")


@pytest.mark.integration
@needs_db
@pytest.mark.skipif(not REAL_INDEX.exists(), reason="no local index (data/ is not in the repository)")
def test_same_top_k_as_numpy_on_the_real_index(pg_factory):
    """The point of the flag: switching stores must not change what is retrieved."""
    numpy_store = NumpyStore.load(REAL_INDEX)
    vectors = np.load(REAL_INDEX / "vectors.npy")
    pg = pg_factory()
    pg.add(numpy_store.chunks(), vectors)
    assert [c.id for c in pg.chunks()] == [c.id for c in numpy_store.chunks()]

    rng = np.random.default_rng(0)
    # Queries: stored vectors plus noise, so scores are realistic and ranks are not trivial.
    picks = rng.choice(len(vectors), size=60, replace=False)
    queries = vectors[picks] + rng.normal(scale=0.02, size=(60, vectors.shape[1])).astype(np.float32)
    filters = [None] + [{doc} for doc in sorted(numpy_store.docs())]
    checked = 0
    for query in queries:
        for docs in filters:
            for k in (1, 4, 8):
                want, got = numpy_store.search(query, k, docs), pg.search(query, k, docs)
                assert [h.chunk.id for h in got] == [h.chunk.id for h in want]
                assert [h.score for h in got] == pytest.approx([h.score for h in want], abs=1e-5)
                checked += 1
    assert checked == 60 * len(filters) * 3


# --- the flag and the fallback, no database needed -----------------------------------------


def test_numpy_is_the_default_store():
    assert Settings.model_fields["store"].default == "numpy"
    assert cli.store_kind(Namespace(store=None), Settings(store="numpy")) == "numpy"
    assert cli.store_kind(Namespace(), Settings(store="numpy")) == "numpy"  # older callers without the flag


def test_flag_beats_the_setting_and_unknown_values_are_rejected():
    assert cli.store_kind(Namespace(store="pgvector"), Settings(store="numpy")) == "pgvector"
    assert cli.store_kind(Namespace(store=None), Settings(store="pgvector")) == "pgvector"
    with pytest.raises(SystemExit, match="Unknown store 'sqlite'"):
        cli.store_kind(Namespace(store=None), Settings(store="sqlite"))


def test_pgvector_without_database_url_explains_what_to_do():
    with pytest.raises(SystemExit, match="DATABASE_URL is not set"):
        cli.open_pgvector(Settings(database_url=None), "x")


def test_pgvector_store_cannot_be_loaded_from_a_directory():
    pytest.importorskip("pgvector")
    from docrag.vectorstore.pgvector_store import PgVectorStore

    with pytest.raises(NotImplementedError):
        PgVectorStore.load(Path("anywhere"))


def _index_settings(tmp_path):
    return Settings(
        index_dir=tmp_path / "index", embedding_cache_path=tmp_path / "cache.sqlite",
        pdf_dir=tmp_path, store="numpy", database_url=None,
    )


def _patch_cli(monkeypatch):
    emb = FakeEmbedder()
    monkeypatch.setattr(cli, "make_embedder", lambda name, s: emb)
    monkeypatch.setattr(
        cli, "load_corpus", lambda d: [Page(doc="a.pdf", page=1, text="alpha beta gamma. " * 30)]
    )
    return emb


def _index_args(**kw):
    return Namespace(embedder="openai", chunk_size=None, chunk_overlap=None, store=None, yes=True, **kw)


def test_default_index_and_load_still_use_files(tmp_path, monkeypatch):
    _patch_cli(monkeypatch)
    monkeypatch.setattr(cli, "open_pgvector", lambda *a: pytest.fail("the default path must not touch Postgres"))
    s = _index_settings(tmp_path)
    cli.cmd_index(_index_args(), s)
    assert list((tmp_path / "index").rglob("vectors.npy"))
    store, _ = cli.load_store(_index_args(), s)
    assert isinstance(store, NumpyStore) and len(store) > 0


def test_pgvector_dry_run_does_not_connect(tmp_path, monkeypatch, capsys):
    _patch_cli(monkeypatch)
    monkeypatch.setattr(cli, "open_pgvector", lambda *a: pytest.fail("a dry run must not connect"))
    args = _index_args()
    args.yes, args.store = False, "pgvector"
    cli.cmd_index(args, _index_settings(tmp_path))
    out = capsys.readouterr().out
    assert "Postgres, index 'fake_hash_c800o150'" in out and "Dry run only" in out
    assert not (tmp_path / "index").exists()  # and nothing was written to files


def test_pgvector_index_writes_to_the_store_and_not_to_files(tmp_path, monkeypatch):
    _patch_cli(monkeypatch)
    fake = NumpyStore()
    fake.reset = lambda: setattr(fake, "was_reset", True)
    fake.save = lambda directory: None  # like PgVectorStore.save
    monkeypatch.setattr(cli, "open_pgvector", lambda s, name: fake)
    args = _index_args()
    args.store = "pgvector"
    cli.cmd_index(args, _index_settings(tmp_path))
    assert getattr(fake, "was_reset", False) and len(fake) > 0
    assert not (tmp_path / "index").exists()


def test_pgvector_load_with_an_empty_database_says_to_build_the_index(tmp_path, monkeypatch):
    _patch_cli(monkeypatch)
    monkeypatch.setattr(cli, "open_pgvector", lambda s, name: NumpyStore())
    args = _index_args()
    args.store = "pgvector"
    with pytest.raises(SystemExit, match="No index in Postgres.*--store pgvector"):
        cli.load_store(args, _index_settings(tmp_path))
