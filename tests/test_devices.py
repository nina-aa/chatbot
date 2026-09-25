import json
from argparse import Namespace

import pytest

from docrag import cli
from docrag.config import Settings
from docrag.devices import DeviceRegistry, derive_names, load_overrides
from docrag.eval.golden import GoldenItem, RelevantPages, check_devices
from docrag.eval.runner import run_eval
from docrag.models import Chunk
from docrag.rag.pipeline import RagPipeline
from docrag.retrieval.dense import DenseRetriever
from docrag.vectorstore.numpy_store import NumpyStore
from tests.conftest import FakeEmbedder
from tests.test_pipeline import ScriptedLLM, statements

# ---- names ---------------------------------------------------------------------------


def test_derive_names_from_file_names():
    assert derive_names("HMW90 user guide.pdf") == ["HMW90"]
    assert derive_names("WAA151 User Guide.pdf") == ["WAA151"]
    assert derive_names("DKP202-DKP203-DKP204-DKP204W-CAMPOLE-User-Guide-M210373EN-E (1).pdf") == [
        "DKP202", "DKP203", "DKP204", "DKP204W", "CAMPOLE",
    ]
    assert derive_names("notes.pdf") == ["notes"]  # no "user guide" in the name: whole stem


def test_registry_matches_case_insensitively_and_rejects_unknown():
    reg = DeviceRegistry({"HMW90 user guide.pdf", "HPP271 user guide.pdf"})
    assert reg.names == ["HMW90", "HPP271"]
    assert reg.resolve("hpp271") == {"HPP271 user guide.pdf"}
    assert reg.resolve("  HMW90 ") == {"HMW90 user guide.pdf"}
    assert reg.resolve("WAA151") is None and reg.resolve("") is None and reg.resolve(None) is None


def test_overrides_replace_derived_names(tmp_path):
    (tmp_path / "devices.json").write_text(json.dumps({"a.pdf": ["Alpha", "Beta"]}))
    assert load_overrides(tmp_path) == {"a.pdf": ["Alpha", "Beta"]}
    assert load_overrides(tmp_path / "nowhere") == {}
    reg = DeviceRegistry({"a.pdf", "b.pdf"}, load_overrides(tmp_path))
    assert reg.names == ["Alpha", "Beta", "b"] and reg.resolve("beta") == {"a.pdf"}


# ---- filtered search -----------------------------------------------------------------

TEXT = {
    "A user guide.pdf": "restart the transmitter to clear the error",
    "B user guide.pdf": "restart the transmitter to clear the error",  # identical wording, other device
}


def _store():
    emb = FakeEmbedder()
    store = NumpyStore()
    chunks = [Chunk(id=doc, doc=doc, page=1, text=t) for doc, t in TEXT.items()]
    store.add(chunks, emb.embed_documents([c.text for c in chunks]))
    return store, emb


def test_store_search_can_be_limited_to_documents():
    store, emb = _store()
    q = emb.embed_query("restart the transmitter")
    assert {h.chunk.doc for h in store.search(q, 5)} == set(TEXT)
    only_b = store.search(q, 5, docs={"B user guide.pdf"})
    assert [h.chunk.doc for h in only_b] == ["B user guide.pdf"]  # A never sneaks in
    assert store.search(q, 5, docs={"nothing.pdf"}) == []
    assert store.docs() == set(TEXT)


# ---- pipeline ------------------------------------------------------------------------


def _pipeline(llm, **kw):
    store, emb = _store()
    return RagPipeline(DenseRetriever(store, emb), llm, top_k=4, **kw), emb


def test_no_device_means_nothing_is_processed():
    llm = ScriptedLLM()
    pipe, emb = _pipeline(llm)
    result = pipe.ask("restart the transmitter")
    assert result.status == "needs_device" and result.reason == "device_required"
    assert result.devices == ["A", "B"]
    assert emb.calls == 2  # only the two indexed chunks; the question was not embedded
    assert llm.prompts == [] and result.trace.llm_calls == []


def test_unknown_device_is_refused_with_candidates():
    llm = ScriptedLLM()
    pipe, emb = _pipeline(llm)
    result = pipe.ask("restart", device="Z9")
    assert result.status == "needs_device" and "Z9" in result.reason and result.devices == ["A", "B"]
    assert llm.prompts == []


def test_device_limits_retrieval_to_its_manual():
    reply = statements(("Restart it.", "B user guide.pdf", 1, "restart the transmitter to clear the error"))
    llm = ScriptedLLM(reply)
    pipe, _ = _pipeline(llm)
    result = pipe.ask("restart the transmitter", device="b")
    assert result.status == "answered"
    assert {h.chunk.doc for h in result.hits} == {"B user guide.pdf"}
    assert "A user guide.pdf" not in llm.prompts[0]  # the other manual never reaches the LLM


def test_retrieve_with_device_filters_and_without_does_not():
    pipe, _ = _pipeline(ScriptedLLM())
    assert {h.chunk.doc for h in pipe.retrieve("restart", device="A")} == {"A user guide.pdf"}
    assert len(pipe.retrieve("restart")) == 2


# ---- golden set and eval -------------------------------------------------------------


def _item(id_, device):
    return GoldenItem(
        id=id_, question="restart the transmitter", device=device, reference_answer="restart",
        relevant=[RelevantPages(doc="B user guide.pdf", pages=[1])],
    )


def test_check_devices_requires_a_known_device_on_every_item():
    reg = DeviceRegistry(set(TEXT))
    check_devices([_item("a", "A"), _item("b", "b")], reg)
    with pytest.raises(ValueError, match="without a device"):
        check_devices([_item("a", "A"), _item("x", None)], reg)
    with pytest.raises(ValueError, match="Z9"):
        check_devices([_item("z", "Z9")], reg)


def test_eval_uses_the_golden_items_device_unless_told_not_to():
    pipe, _ = _pipeline(ScriptedLLM(), require_device=False)
    with_filter = run_eval(pipe, None, [_item("q", "A")], retrieval_only=True)[0]
    assert {h.chunk.doc for h in with_filter.hits} == {"A user guide.pdf"}
    assert with_filter.retrieval_rank is None  # the golden page is in B, and the filter says A
    without = run_eval(pipe, None, [_item("q", "A")], retrieval_only=True, use_device=False)[0]
    assert len(without.hits) == 2 and without.retrieval_rank is not None


# ---- CLI -----------------------------------------------------------------------------


def _ask(monkeypatch, capsys, device):
    llm = ScriptedLLM()
    pipe, emb = _pipeline(llm)
    monkeypatch.setattr(cli, "load_pipeline", lambda args, s, l: (pipe, "fake"))
    monkeypatch.setattr(cli, "make_llm", lambda name, s: None)
    cli.cmd_ask(Namespace(llm="openai", question="restart the transmitter", verbose=False, device=device), Settings())
    return capsys.readouterr().out, llm


def test_cli_without_device_lists_candidates_and_does_not_query(monkeypatch, capsys):
    out, llm = _ask(monkeypatch, capsys, device=None)
    assert "Which device is this about?" in out and "A, B" not in out  # it asks, it doesn't list
    assert llm.prompts == []


def test_cli_with_unknown_device_says_so(monkeypatch, capsys):
    out, _ = _ask(monkeypatch, capsys, device="Z9")
    assert "Unknown device 'Z9'" in out and "Which device is this about?" in out and "A, B" not in out
