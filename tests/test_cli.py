from argparse import Namespace

from docrag import cli
from docrag.config import Settings
from tests.test_eval import GOOD, _pipeline
from tests.test_trace import UsageLLM
from tests.test_pipeline import answer


def _run_ask(monkeypatch, capsys, verbose):
    pipeline = _pipeline(UsageLLM(answer(citations=(GOOD,))))
    monkeypatch.setattr(cli, "load_pipeline", lambda args, s, llm: (pipeline, "fake"))
    monkeypatch.setattr(cli, "make_llm", lambda name, s: None)
    args = Namespace(llm="openai", question="operator panel screw speed", verbose=verbose, device="m")
    cli.cmd_ask(args, Settings())
    return capsys.readouterr().out


def test_ask_prints_time_tokens_and_cost(monkeypatch, capsys):
    out = _run_ask(monkeypatch, capsys, verbose=False)
    assert "Time " in out and "Tokens 1,000 in / 100 out" in out and "Cost ~$0.00021" in out
    assert "--- 1." not in out  # chunk text only with --verbose


def test_verbose_prints_retrieved_chunk_text(monkeypatch, capsys):
    out = _run_ask(monkeypatch, capsys, verbose=True)
    assert "--- 1. m.pdf p.7" in out and "operator panel" in out


# ---- search: dense, keyword and fused side by side -----------------------------------


def _search(monkeypatch, capsys, device):
    from docrag.models import Chunk
    from docrag.vectorstore.numpy_store import NumpyStore
    from tests.conftest import FakeEmbedder

    emb, store = FakeEmbedder(), NumpyStore()
    chunks = [
        Chunk(id="a", doc="A user guide.pdf", page=3, text="Internal 1 means an internal transmitter problem.", section="Errors"),
        Chunk(id="b", doc="A user guide.pdf", page=4, text="Configuration registers and addresses."),
        Chunk(id="c", doc="B user guide.pdf", page=9, text="Internal 1 in the other manual."),
    ]
    store.add(chunks, emb.embed_documents([c.text for c in chunks]))
    monkeypatch.setattr(cli, "load_store", lambda args, s: (store, emb))
    cli.cmd_search(Namespace(device=device, question="Internal 1", top_k=2), Settings())
    return capsys.readouterr().out


def test_search_shows_three_rankings_with_scores_and_matched_terms(monkeypatch, capsys):
    out = _search(monkeypatch, capsys, device="a")
    assert out.index("DENSE") < out.index("KEYWORD") < out.index("HYBRID")
    assert "cosine " in out and "bm25 " in out and "matched: internal, 1" in out
    assert "rrf " in out and "dense #1" in out and "keyword #1" in out
    assert "A user guide.pdf p.3" in out and "[Errors]" in out
    assert "B user guide.pdf" not in out  # the device filter applies here too


def test_search_without_a_device_only_asks(monkeypatch, capsys):
    out = _search(monkeypatch, capsys, device=None)
    assert "Which device is this about?" in out and "DENSE" not in out


# ---- console encoding ----------------------------------------------------------------


def test_output_is_utf8_even_when_the_default_encoding_cannot_represent_the_text(monkeypatch):
    import io

    raw = io.BytesIO()
    fake_stdout = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")  # what a redirect gets on Windows
    monkeypatch.setattr("sys.stdout", fake_stdout)
    monkeypatch.setattr("sys.stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    cli.use_utf8_output()
    print("bullet \u2023 ellipsis \u2026 en-space\u2002")  # would raise UnicodeEncodeError before the fix
    fake_stdout.flush()
    assert "\u2023 ellipsis \u2026".encode("utf-8") in raw.getvalue()


def test_use_utf8_output_tolerates_streams_that_cannot_be_reconfigured(monkeypatch):
    class Plain:  # e.g. a captured stream with no reconfigure()
        pass

    monkeypatch.setattr("sys.stdout", Plain())
    monkeypatch.setattr("sys.stderr", Plain())
    cli.use_utf8_output()  # must not raise
