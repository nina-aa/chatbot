"""The `eval` command end to end with fakes: with and without the reranker."""

import json
from argparse import Namespace

import pytest

from docrag import cli
from docrag.config import Settings
from docrag.models import Chunk
from docrag.retrieval.bm25 import Bm25Retriever
from docrag.retrieval.dense import DenseRetriever
from docrag.retrieval.hybrid import HybridRetriever
from docrag.retrieval.rerank import RerankingRetriever
from docrag.vectorstore.numpy_store import NumpyStore
from tests.conftest import FakeEmbedder
from tests.test_pipeline import ScriptedLLM, statements

DOC = "A user guide.pdf"
TEXT = "To clear the error, restart the transmitter and check the wiring."


def _store():
    emb, store = FakeEmbedder(), NumpyStore()
    chunks = [
        Chunk(id="a1", doc=DOC, page=1, text=TEXT, section="Errors"),
        Chunk(id="a2", doc=DOC, page=2, text="Configuration registers and addresses."),
        Chunk(id="b1", doc="B user guide.pdf", page=1, text=TEXT),
    ]
    store.add(chunks, emb.embed_documents([c.text for c in chunks]))
    return store, emb


def _golden(tmp_path):
    rows = [
        {"id": "q1", "question": "how do I clear the error", "device": "A", "reference_answer": "restart",
         "relevant": [{"doc": DOC, "pages": [1]}]},
        {"id": "q2", "question": "what is the warranty", "device": "A", "answerable": False},
    ]
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def _run(tmp_path, monkeypatch, llm, made=None, **overrides):
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir(exist_ok=True)
    (pdfs / DOC).touch()
    store, emb = _store()
    monkeypatch.setattr(cli, "load_store", lambda args, s: (store, emb))
    monkeypatch.setattr(cli, "make_llm", lambda name, s: (made.append(name) if made is not None else None) or llm)
    args = dict(
        golden=_golden(tmp_path), limit=None, retrieval_only=False, llm="openai", judge="none",
        embedder="openai", chunk_size=None, chunk_overlap=None, top_k=2, min_score=None,
        retriever=None, rerank=False, rerank_llm="openai", no_device_filter=False,
    )
    args.update(overrides)
    settings = Settings(pdf_dir=pdfs, eval_dir=tmp_path / "runs")
    cli.cmd_eval(Namespace(**args), settings)
    (run,) = (tmp_path / "runs").iterdir()
    return run, json.loads((run / "summary.json").read_text())


REFUSE = json.dumps({"sufficient": False, "statements": []})
ANSWER = statements(("Restart it.", DOC, 1, "restart the transmitter and check the wiring"))
RATINGS = json.dumps({"ratings": [{"id": 1, "score": 3}, {"id": 2, "score": 0}]})


def test_full_eval_without_the_reranker(tmp_path, monkeypatch, capsys):
    llm = ScriptedLLM(ANSWER, REFUSE)  # exactly one answer call per question: no rerank calls
    run, saved = _run(tmp_path, monkeypatch, llm)
    out = capsys.readouterr().out
    assert len(llm.prompts) == 2
    assert saved["config"]["rerank"] is None and saved["config"]["retriever"] == "dense"
    assert saved["summary"]["answered_rate"] == 1.0 and saved["summary"]["correct_refusal_rate"] == 1.0
    assert saved["summary"]["rerank_input_tokens"] == 0
    assert "rerank" not in run.name and "Reranking" not in out
    assert {p.name for p in run.iterdir()} == {"results.jsonl", "summary.json"}


def test_retrieval_only_eval_without_the_reranker_calls_no_llm(tmp_path, monkeypatch, capsys):
    llm = ScriptedLLM()  # would raise if called
    run, saved = _run(tmp_path, monkeypatch, llm, retrieval_only=True)
    assert llm.prompts == [] and saved["config"]["llm"] is None
    assert saved["summary"]["hit_at_k"]["1"] == 1.0 and run.name.endswith("_retrieval")
    assert "hit@k" in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["dense", "keyword", "hybrid"])
def test_each_retriever_runs_a_full_eval(tmp_path, monkeypatch, kind):
    _, saved = _run(tmp_path, monkeypatch, ScriptedLLM(ANSWER, REFUSE), retriever=kind)
    assert saved["config"]["retriever"] == kind and saved["summary"]["answered_rate"] == 1.0


def test_eval_with_the_reranker_records_and_prices_it(tmp_path, monkeypatch, capsys):
    llm = ScriptedLLM(RATINGS, ANSWER, RATINGS, REFUSE)  # per question: rerank call, then answer call
    run, saved = _run(tmp_path, monkeypatch, llm, rerank=True)
    assert len(llm.prompts) == 4
    assert saved["config"]["rerank"] == "openai" and run.name.endswith("_rerank")
    assert "Reranking: openai" in capsys.readouterr().out


def test_no_device_filter_flag_runs_without_the_device_check(tmp_path, monkeypatch):
    _, saved = _run(tmp_path, monkeypatch, ScriptedLLM(ANSWER, REFUSE), no_device_filter=True)
    assert saved["config"]["device_filter"] is False


# ---- what build_retriever assembles --------------------------------------------------


def test_reranking_is_off_unless_asked_for(monkeypatch):
    store, emb = _store()
    s = Settings()
    monkeypatch.setattr(cli, "make_llm", lambda name, s: ScriptedLLM())
    assert isinstance(cli.build_retriever("dense", store, emb, s), DenseRetriever)
    assert isinstance(cli.build_retriever("keyword", store, emb, s), Bm25Retriever)
    assert isinstance(cli.build_retriever("hybrid", store, emb, s), HybridRetriever)
    wrapped = cli.build_retriever("hybrid", store, emb, s, rerank_llm="openai")
    assert isinstance(wrapped, RerankingRetriever)


def test_the_default_setting_is_plain_dense_retrieval():
    assert Settings().retriever == "dense"


# ---- the reranker is strictly opt-in -------------------------------------------------


def test_no_llm_is_created_for_reranking_unless_the_flag_is_used(tmp_path, monkeypatch):
    made = []
    _run(tmp_path, monkeypatch, ScriptedLLM(ANSWER, REFUSE), made=made)
    assert made == ["openai"]  # only the answering model; nothing was built for reranking


def test_the_flag_builds_a_second_llm_for_reranking(tmp_path, monkeypatch):
    made = []
    _run(tmp_path, monkeypatch, ScriptedLLM(RATINGS, ANSWER, RATINGS, REFUSE), made=made, rerank=True)
    assert made == ["openai", "openai"]  # answering model + reranking model


def test_naming_a_rerank_model_alone_does_not_turn_reranking_on(tmp_path, monkeypatch):
    made = []
    _run(tmp_path, monkeypatch, ScriptedLLM(ANSWER, REFUSE), made=made, rerank=False, rerank_llm="gemini")
    assert made == ["openai"]


def _parse(monkeypatch, argv):
    """Runs the real argument parser and returns what the command would receive."""
    import sys

    seen = {}
    monkeypatch.setattr(cli, "cmd_eval", lambda args, s: seen.update(args=args))
    monkeypatch.setattr(cli, "cmd_ask", lambda args, s: seen.update(args=args))
    monkeypatch.setattr(sys, "argv", ["docrag", *argv])
    cli.main()
    return seen["args"]


def test_rerank_defaults_to_off_in_the_real_argument_parser(monkeypatch):
    assert _parse(monkeypatch, ["eval", "g.jsonl"]).rerank is False
    assert _parse(monkeypatch, ["ask", "--device", "A", "question"]).rerank is False
    assert _parse(monkeypatch, ["eval", "g.jsonl", "--rerank-llm", "gemini"]).rerank is False
    args = _parse(monkeypatch, ["eval", "g.jsonl", "--rerank"])
    assert args.rerank is True and args.rerank_llm == "openai"
