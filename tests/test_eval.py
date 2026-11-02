import json

import pytest

from docrag.eval.golden import GoldenItem, RelevantPages, check_docs_exist, load_golden
from docrag.eval.report import format_summary, write_run
from docrag.eval.runner import NoLLM, first_relevant_rank, run_eval, summarize
from docrag.factory import default_judge, same_family
from docrag.models import Chunk, Hit
from docrag.rag.pipeline import RagPipeline
from docrag.retrieval.dense import DenseRetriever
from docrag.vectorstore.numpy_store import NumpyStore
from tests.conftest import FakeEmbedder
from tests.test_pipeline import ScriptedLLM, answer

# ---- golden set ----------------------------------------------------------------------


def _line(**kw):
    base = {"id": "q", "question": "?", "reference_answer": "a", "relevant": [{"doc": "d.pdf", "pages": [1]}]}
    return json.dumps({**base, **kw})


def test_load_golden_reads_jsonl_and_skips_blank_lines(tmp_path):
    f = tmp_path / "g.jsonl"
    f.write_text(_line(id="a") + "\n\n" + json.dumps({"id": "b", "question": "?", "answerable": False}) + "\n")
    items = load_golden(f)
    assert [i.id for i in items] == ["a", "b"] and not items[1].answerable


def test_answerable_needs_reference_and_pages():
    with pytest.raises(ValueError):
        GoldenItem(id="x", question="?")


def test_unanswerable_must_not_list_pages():
    with pytest.raises(ValueError):
        GoldenItem(id="x", question="?", answerable=False, relevant=[RelevantPages(doc="d", pages=[1])])


def test_load_error_names_file_and_line(tmp_path):
    f = tmp_path / "g.jsonl"
    f.write_text(_line() + "\n" + '{"id": "bad"}\n')
    with pytest.raises(ValueError, match=r"g\.jsonl:2"):
        load_golden(f)


def test_duplicate_ids_rejected(tmp_path):
    f = tmp_path / "g.jsonl"
    f.write_text(_line() + "\n" + _line() + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_golden(f)


def test_unknown_doc_name_fails_loudly(tmp_path):
    f = tmp_path / "g.jsonl"
    f.write_text(_line())
    with pytest.raises(ValueError, match="d.pdf"):
        check_docs_exist(load_golden(f), {"other.pdf"})
    check_docs_exist(load_golden(f), {"d.pdf"})


def test_shipped_golden_file_is_valid_and_refers_to_real_pdfs():
    from pathlib import Path

    items = load_golden(Path("eval/golden.jsonl"))
    assert items
    if Path("pdfs").exists():
        check_docs_exist(items, {p.name for p in Path("pdfs").glob("*.pdf")})


# ---- judge family --------------------------------------------------------------------


def test_default_judge_is_other_family():
    assert default_judge("openai") == "gemini"
    assert default_judge("gemini") == "openai" and default_judge("gemma") == "openai"
    assert same_family("gemini", "gemma") and not same_family("openai", "gemini")


# ---- retrieval rank ------------------------------------------------------------------


def _hit(doc, page):
    return Hit(chunk=Chunk(id=f"{doc}{page}", doc=doc, page=page, text="t"), score=0.5)


def test_first_relevant_rank():
    hits = [_hit("a.pdf", 1), _hit("a.pdf", 5), _hit("b.pdf", 5)]
    assert first_relevant_rank(hits, [RelevantPages(doc="b.pdf", pages=[5])]) == 3
    assert first_relevant_rank(hits, [RelevantPages(doc="a.pdf", pages=[5, 6])]) == 2
    assert first_relevant_rank(hits, [RelevantPages(doc="c.pdf", pages=[1])]) is None


# ---- end to end with fakes -----------------------------------------------------------

PANEL = "The screw speed is set on the operator panel under the Compressor menu."
PINOUT = "Pin 1 carries the positive supply and pin 2 the ground."

ITEMS = [
    GoldenItem(id="hit", question="where is the screw speed set on the operator panel",
               reference_answer="On the operator panel", relevant=[RelevantPages(doc="m.pdf", pages=[7])]),
    GoldenItem(id="miss", question="what does pin 1 carry",
               reference_answer="positive supply", relevant=[RelevantPages(doc="m.pdf", pages=[99])]),
    GoldenItem(id="nothing", question="what is the warranty period", answerable=False),
]
GOOD = ("m.pdf", 7, "screw speed is set on the operator panel")


def _pipeline(llm, top_k=2):
    emb = FakeEmbedder()
    store = NumpyStore()
    chunks = [Chunk(id="0", doc="m.pdf", page=7, text=PANEL), Chunk(id="1", doc="m.pdf", page=9, text=PINOUT)]
    store.add(chunks, emb.embed_documents([c.text for c in chunks]))
    return RagPipeline(DenseRetriever(store, emb), llm, top_k=top_k, require_device=False)


def _verdict(correct=True, faithful=True):
    return json.dumps({"correct": correct, "faithful": faithful, "reason": "r"})


def test_retrieval_only_run_makes_no_llm_calls_and_reports_retrieval():
    results = run_eval(_pipeline(NoLLM()), None, ITEMS, retrieval_only=True)
    s = summarize(results, ks=[1, 2])
    assert s.hit_at_k == {1: 0.5, 2: 0.5}  # "hit" found at rank 1, "miss" never
    assert s.mrr == 0.5
    assert s.correct_rate is None and s.hallucination_rate is None


def test_full_run_scores_answers_refusals_and_hallucinations():
    refuse = json.dumps({"sufficient": False, "answer": "", "citations": []})
    llm = ScriptedLLM(
        answer(citations=(GOOD,)),  # "hit": answered with a real quote
        refuse,  # "miss": false refusal
        answer(citations=(GOOD,)),  # "nothing": hallucinated an answer
    )
    judge = ScriptedLLM(_verdict(correct=True, faithful=True))  # only "hit" reaches the judge
    results = run_eval(_pipeline(llm), judge, ITEMS)
    s = summarize(results, ks=[1, 2], judge_used=True)

    assert s.answered_rate == 0.5 and s.false_refusal_rate == 0.5
    assert s.correct_rate == 0.5  # 1 of 2 answerable; the refusal counts as wrong
    assert s.faithful_rate == 1.0
    assert s.citation_on_relevant_page_rate == 1.0
    assert s.hallucination_rate == 1.0 and s.correct_refusal_rate == 0.0
    assert s.models_used == {"scripted": 3} and s.errors == 0
    assert len(judge.prompts) == 1 and "Reference answer" in judge.prompts[0]


def test_unfaithful_or_wrong_verdicts_are_counted():
    llm = ScriptedLLM(answer(citations=(GOOD,)), answer(citations=(GOOD,)))
    judge = ScriptedLLM(_verdict(True, False), _verdict(False, True))
    results = run_eval(_pipeline(llm), judge, ITEMS[:2])
    s = summarize(results, ks=[1], judge_used=True)
    assert s.correct_rate == 0.5 and s.faithful_rate == 0.5
    assert s.citation_on_relevant_page_rate == 0.5  # 2nd item's relevant page is 99


def test_citation_on_wrong_page_is_flagged():
    res = run_eval(_pipeline(ScriptedLLM(answer(citations=(GOOD,)))), None, [ITEMS[1]])
    assert res[0].citation_on_relevant_page is False


def test_judge_failure_is_recorded_not_fatal():
    llm = ScriptedLLM(answer(citations=(GOOD,)))
    judge = ScriptedLLM("garbage", "garbage", "garbage")
    results = run_eval(_pipeline(llm), judge, ITEMS[:1])
    assert results[0].verdict is None and results[0].judge_error
    assert summarize(results, [1], judge_used=True).judge_errors == 1


def test_no_judge_leaves_judge_metrics_unmeasured():
    results = run_eval(_pipeline(ScriptedLLM(answer(citations=(GOOD,)))), None, ITEMS[:1])
    s = summarize(results, [1], judge_used=False)
    assert s.correct_rate is None and s.faithful_rate is None and s.answered_rate == 1.0


def test_errors_are_counted(tmp_path):
    results = run_eval(_pipeline(ScriptedLLM("x", "x", "x")), None, ITEMS[:1])
    s = summarize(results, [1])
    assert s.errors == 1
    # and a run can be written and printed
    write_run(tmp_path, results, s, {"embedder": "fake", "top_k": 2, "min_score": 0, "llm": "openai", "judge": None})
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 1
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["summary"]["errors"] == 1
    text = format_summary(s, saved["config"])
    assert "hit@k" in text and "precision@k" in text and "page recall@k" in text


# ---- precision and recall ------------------------------------------------------------


def test_precision_and_page_recall_at_k():
    from docrag.eval.runner import page_recall_at_k, precision_at_k

    relevant = [RelevantPages(doc="a.pdf", pages=[5, 6]), RelevantPages(doc="b.pdf", pages=[1])]
    hits = [_hit("a.pdf", 5), _hit("a.pdf", 5), _hit("c.pdf", 9), _hit("b.pdf", 1)]
    assert precision_at_k(hits, relevant, 1) == 1.0
    assert precision_at_k(hits, relevant, 4) == 0.75  # 3 of 4 chunks are on a golden page
    assert page_recall_at_k(hits, relevant, 1) == 1 / 3  # only a.pdf p.5 of {a5, a6, b1}
    assert page_recall_at_k(hits, relevant, 4) == 2 / 3  # a5 and b1 found, a6 missed
    assert precision_at_k([], relevant, 3) == 0.0


def test_summary_averages_precision_and_recall_over_answerable_questions_only():
    results = run_eval(_pipeline(NoLLM(), top_k=2), None, ITEMS, retrieval_only=True)
    s = summarize(results, ks=[1, 2])
    # "hit": golden m.pdf p.7, retrieved [p.7, p.9] -> precision@1 1.0, @2 0.5, recall 1.0
    # "miss": golden m.pdf p.99, never retrieved -> 0. "nothing" is unanswerable and ignored.
    assert s.precision_at_k == {1: 0.5, 2: 0.25}
    assert s.recall_at_k == {1: 0.5, 2: 0.5}


def test_smoke_set_is_a_valid_three_question_subset_of_the_golden_set():
    from pathlib import Path

    golden = {i.id: i for i in load_golden(Path("eval/golden.jsonl"))}
    smoke = load_golden(Path("eval/golden_smoke.jsonl"))
    assert len(smoke) == 3
    assert all(i == golden[i.id] for i in smoke)  # copied, not edited: it can't drift from the golden set
    assert {i.answerable for i in smoke} == {True, False}  # one unanswerable, so refusal is exercised


def test_sample10_is_ten_unedited_golden_questions():
    from pathlib import Path

    golden = {i.id: i for i in load_golden(Path("eval/golden.jsonl"))}
    sample = load_golden(Path("eval/golden_sample10.jsonl"))
    assert len(sample) == 10 and all(i == golden[i.id] for i in sample)
    assert sum(not i.answerable for i in sample) == 2


def test_dense_misses_set_is_unedited_golden_questions():
    from pathlib import Path

    golden = {i.id: i for i in load_golden(Path("eval/golden.jsonl"))}
    misses = load_golden(Path("eval/golden_dense_misses.jsonl"))
    assert len(misses) == 3 and all(i == golden[i.id] for i in misses)
