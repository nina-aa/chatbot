import json

from docrag.llm.base import LLM, LLMResponse
from docrag.models import Chunk
from docrag.rag.pipeline import RagPipeline
from docrag.retrieval.dense import DenseRetriever
from docrag.vectorstore.numpy_store import NumpyStore
from tests.conftest import FakeEmbedder

PANEL = "The screw speed is set on the operator panel under the Compressor menu."
PINOUT = "Pin 1 carries the positive supply and pin 2 the ground."


class ScriptedLLM(LLM):
    """Replies with each scripted string in turn. Records every prompt it was given."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt, system=None, json_mode=False):
        self.prompts.append(prompt)
        return LLMResponse(text=self.replies.pop(0), model="scripted")


def statements(*items):
    """items are (text, doc, page, quote). Builds the JSON the LLM is asked to return."""
    return json.dumps(
        {
            "sufficient": True,
            "statements": [{"text": t, "doc": d, "page": p, "quote": q} for t, d, p, q in items],
        }
    )


def answer(text="Use the panel.", citations=(("m.pdf", 7, "screw speed is set on the operator panel"),)):
    """One statement per citation, all with the same text. For tests that don't care about wording."""
    return statements(*[(text, d, p, q) for d, p, q in citations])


def pipeline(llm, min_score=0.0, max_attempts=3):
    emb = FakeEmbedder()
    store = NumpyStore()
    chunks = [
        Chunk(id="0", doc="m.pdf", page=7, text=PANEL, section="3 Operation > 3.1 Panel"),
        Chunk(id="1", doc="m.pdf", page=9, text=PINOUT),
    ]
    store.add(chunks, emb.embed_documents([c.text for c in chunks]))
    return RagPipeline(
        DenseRetriever(store, emb), llm, top_k=2, min_score=min_score, max_attempts=max_attempts, require_device=False
    )


def test_grounded_answer_is_returned_with_verified_citation_and_section():
    result = pipeline(ScriptedLLM(answer())).ask("where is screw speed set")
    assert result.status == "answered" and result.answer == "Use the panel."
    assert result.statements[0].citation.page == 7 and result.citations[0].page == 7
    assert result.citations[0].section == "3 Operation > 3.1 Panel"
    assert result.attempts == 1 and result.model == "scripted"


def test_prompt_contains_context_with_doc_page_and_section():
    llm = ScriptedLLM(answer())
    pipeline(llm).ask("screw speed")
    assert "[doc: m.pdf | page: 7 | section: 3 Operation > 3.1 Panel]" in llm.prompts[0]


def test_low_retrieval_score_skips_the_llm():
    llm = ScriptedLLM()  # would raise IndexError if called
    result = pipeline(llm, min_score=0.99).ask("something unrelated entirely")
    assert result.status == "insufficient" and result.reason == "low_retrieval_score"
    assert llm.prompts == [] and result.model is None


def test_model_reporting_insufficient_is_passed_through():
    reply = json.dumps({"sufficient": False, "statements": []})
    result = pipeline(ScriptedLLM(reply)).ask("q")
    assert result.status == "insufficient" and result.reason == "model_reported_insufficient"
    assert result.answer is None


def test_fabricated_quote_downgrades_to_insufficient():
    fake = answer(citations=(("m.pdf", 7, "the machine must be rebooted twice daily"),))
    result = pipeline(ScriptedLLM(fake)).ask("q")
    assert result.status == "insufficient" and result.reason == "no_verifiable_citations"
    assert result.answer is None and len(result.rejected_statements) == 1


def test_quote_from_wrong_page_is_rejected():
    wrong_page = answer(citations=(("m.pdf", 9, "screw speed is set on the operator panel"),))
    assert pipeline(ScriptedLLM(wrong_page)).ask("q").status == "insufficient"


def test_each_statement_gets_its_own_verified_source():
    reply = statements(
        ("Set it on the panel.", "m.pdf", 7, "screw speed is set on the operator panel"),
        ("Use the Compressor menu.", "m.pdf", 7, "under the Compressor menu"),
        ("Pin 1 is the supply.", "m.pdf", 9, "Pin 1 carries the positive supply"),
    )
    result = pipeline(ScriptedLLM(reply)).ask("q")
    assert result.status == "answered"
    assert [(s.text, s.citation.page, s.citation.quote) for s in result.statements] == [
        ("Set it on the panel.", 7, "screw speed is set on the operator panel"),
        ("Use the Compressor menu.", 7, "under the Compressor menu"),
        ("Pin 1 is the supply.", 9, "Pin 1 carries the positive supply"),
    ]
    assert result.answer == "Set it on the panel. Use the Compressor menu. Pin 1 is the supply."
    assert len(result.citations) == 3


def test_unsupported_statement_is_dropped_from_the_answer_but_reported():
    reply = statements(
        ("Set it on the panel.", "m.pdf", 7, "screw speed is set on the operator panel"),
        ("Reboot twice daily.", "m.pdf", 7, "completely invented sentence here"),
    )
    result = pipeline(ScriptedLLM(reply)).ask("q")
    assert result.status == "answered"
    assert result.answer == "Set it on the panel."  # the invented claim is not shown
    assert [s.text for s in result.rejected_statements] == ["Reboot twice daily."]


def test_statement_quoting_the_wrong_page_is_dropped():
    reply = statements(
        ("Set it on the panel.", "m.pdf", 7, "screw speed is set on the operator panel"),
        ("Pin 1 is the supply.", "m.pdf", 7, "Pin 1 carries the positive supply"),  # that text is on p.9
    )
    result = pipeline(ScriptedLLM(reply)).ask("q")
    assert [s.text for s in result.statements] == ["Set it on the panel."]
    assert len(result.rejected_statements) == 1


def test_invalid_output_is_repaired_on_retry():
    llm = ScriptedLLM("Sure! here is the answer", answer())
    result = pipeline(llm).ask("q")
    assert result.status == "answered" and result.attempts == 2
    assert "invalid" in llm.prompts[1] and "Sure! here is the answer" in llm.prompts[1]


def test_sufficient_without_statements_fails_schema_and_is_retried():
    no_statements = json.dumps({"sufficient": True, "statements": []})
    result = pipeline(ScriptedLLM(no_statements, answer())).ask("q")
    assert result.status == "answered" and result.attempts == 2


def test_gives_up_with_error_status_after_max_attempts():
    result = pipeline(ScriptedLLM("nope", "still nope"), max_attempts=2).ask("q")
    assert result.status == "error" and result.attempts == 2 and result.answer is None
