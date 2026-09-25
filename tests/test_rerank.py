import json
import logging

from docrag.eval.golden import GoldenItem, RelevantPages
from docrag.eval.runner import run_eval, summarize
from docrag.llm.base import LLM, AllModelsFailed
from docrag.models import Chunk, Hit
from docrag.rag.pipeline import RagPipeline
from docrag.retrieval.base import Retrieval, Retriever
from docrag.retrieval.rerank import LLMReranker, RerankingRetriever, format_excerpts
from tests.test_pipeline import answer
from tests.test_trace import UsageLLM


def hit(id_, page=1, text=None, section=None):
    return Hit(chunk=Chunk(id=id_, doc="m.pdf", page=page, text=text or f"text of {id_}", section=section), score=0.5)


def ratings(*pairs):
    return json.dumps({"ratings": [{"id": i, "score": s} for i, s in pairs]})


class StubBase(Retriever):
    """Returns a fixed candidate list and remembers what it was asked for."""

    def __init__(self, hits):
        self.hits, self.asked = hits, []

    def retrieve(self, question, k, docs=None):
        self.asked.append((k, docs))
        return Retrieval(self.hits[:k], embed_s=0.1, search_s=0.2)

    def docs(self):
        return {"m.pdf"}


class DownLLM(LLM):
    def generate(self, prompt, system=None, json_mode=False):
        raise AllModelsFailed({"gpt": "HTTP 429"})


# ---- the reranker --------------------------------------------------------------------


def test_hits_are_ordered_by_score_and_ties_keep_the_original_order():
    hits = [hit("a"), hit("b"), hit("c"), hit("d")]
    llm = UsageLLM(ratings((1, 1), (2, 3), (3, 1), (4, 3)))
    ordered = LLMReranker(llm).rerank("q", hits, [])
    assert [h.chunk.id for h in ordered] == ["b", "d", "a", "c"]  # 3,3 then 1,1, each in original order
    assert [h.rerank_score for h in ordered] == [3, 3, 1, 1]


def test_unrated_excerpts_score_zero_and_unknown_ids_are_ignored():
    hits = [hit("a"), hit("b"), hit("c")]
    ordered = LLMReranker(UsageLLM(ratings((3, 2), (99, 3)))).rerank("q", hits, [])
    assert [(h.chunk.id, h.rerank_score) for h in ordered] == [("c", 2), ("a", 0), ("b", 0)]


def test_prompt_numbers_the_excerpts_with_their_source():
    seen = []

    class Spy(UsageLLM):
        def generate(self, prompt, system=None, json_mode=False):
            seen.append((prompt, system))
            return super().generate(prompt, system, json_mode)

    LLMReranker(Spy(ratings((1, 2), (2, 2)))).rerank(
        "why is the reading frozen", [hit("a", page=7, section="10 Troubleshooting"), hit("b", page=9)], []
    )
    prompt, system = seen[0]
    assert "Question: why is the reading frozen" in prompt
    assert "[1] (m.pdf, page 7, section 10 Troubleshooting)\ntext of a" in prompt
    assert "[2] (m.pdf, page 9)\ntext of b" in prompt
    assert "0 to 3" in system and '"ratings"' in system


def test_invalid_ratings_are_repaired_by_asking_again():
    calls = []
    llm = UsageLLM(ratings((1, 3), (1, 0)), ratings((1, 3), (2, 0)))  # first reply rates id 1 twice
    ordered = LLMReranker(llm, max_attempts=2).rerank("q", [hit("a"), hit("b")], calls)
    assert [h.rerank_score for h in ordered] == [3, 0] and len(calls) == 2


def test_out_of_range_scores_are_rejected():
    calls = []
    llm = UsageLLM(ratings((1, 9), (2, 0)), ratings((1, 3), (2, 0)))
    LLMReranker(llm).rerank("q", [hit("a"), hit("b")], calls)
    assert len(calls) == 2  # score 9 is not on the 0-3 scale, so it was retried


def test_calls_are_tagged_as_rerank_with_tokens():
    calls = []
    LLMReranker(UsageLLM(ratings((1, 1), (2, 2)), tokens_in=800, tokens_out=30)).rerank("q", [hit("a"), hit("b")], calls)
    assert [(c.purpose, c.input_tokens, c.output_tokens) for c in calls] == [("rerank", 800, 30)]


def test_format_excerpts_numbers_from_one():
    assert format_excerpts([hit("a"), hit("b")]).startswith("[1] (m.pdf, page 1)")


# ---- the reranking retriever ---------------------------------------------------------


def test_asks_the_base_for_a_wider_pool_and_returns_the_top_k():
    base = StubBase([hit(c) for c in "abcdefgh"])
    r = RerankingRetriever(base, LLMReranker(UsageLLM(ratings(*[(i, 3 if i == 6 else 0) for i in range(1, 9)]))), pool=8)
    result = r.retrieve("q", 3, docs={"m.pdf"})
    assert base.asked == [(8, {"m.pdf"})]  # pool of 8, and the device filter is passed through
    assert [h.chunk.id for h in result.hits] == ["f", "a", "b"]  # the pick from rank 6 leads
    assert result.embed_s == 0.1 and result.search_s == 0.2
    assert [c.purpose for c in result.llm_calls] == ["rerank"]


def test_pool_is_never_smaller_than_k():
    base = StubBase([hit(c) for c in "abcdef"])
    RerankingRetriever(base, LLMReranker(UsageLLM(ratings(*[(i, 1) for i in range(1, 7)]))), pool=2).retrieve("q", 6)
    assert base.asked[0][0] == 6


def test_one_candidate_or_none_needs_no_llm_call():
    for hits in ([], [hit("a")]):
        result = RerankingRetriever(StubBase(hits), LLMReranker(DownLLM()), pool=8).retrieve("q", 4)
        assert result.hits == hits and result.llm_calls == []


def test_unusable_output_keeps_the_original_order_and_says_so(caplog):
    base = StubBase([hit("a"), hit("b"), hit("c")])
    r = RerankingRetriever(base, LLMReranker(UsageLLM("nope", "still nope"), max_attempts=2), pool=8)
    with caplog.at_level(logging.WARNING, logger="docrag"):
        result = r.retrieve("q", 2)
    assert [h.chunk.id for h in result.hits] == ["a", "b"]
    assert all(h.rerank_score is None for h in result.hits)
    assert "reranking failed" in caplog.text
    assert len(result.llm_calls) == 2  # the wasted attempts are still counted


def test_every_model_down_also_falls_back(caplog):
    r = RerankingRetriever(StubBase([hit("a"), hit("b")]), LLMReranker(DownLLM()), pool=8)
    with caplog.at_level(logging.WARNING, logger="docrag"):
        result = r.retrieve("q", 1)
    assert [h.chunk.id for h in result.hits] == ["a"] and "reranking failed" in caplog.text


# ---- in the pipeline and the eval ----------------------------------------------------

GOOD = ("m.pdf", 7, "screw speed is set on the operator panel")


def _reranked_pipeline(answer_reply):
    base = StubBase([
        Hit(chunk=Chunk(id="x", doc="m.pdf", page=3, text="Unrelated wiring text."), score=0.9),
        Hit(chunk=Chunk(id="y", doc="m.pdf", page=7, text="The screw speed is set on the operator panel."), score=0.8),
    ])
    rerank_llm = UsageLLM(ratings((1, 0), (2, 3)), tokens_in=500, tokens_out=20)
    answer_llm = UsageLLM(answer_reply, tokens_in=1000, tokens_out=100)
    retriever = RerankingRetriever(base, LLMReranker(rerank_llm), pool=8)
    return RagPipeline(retriever, answer_llm, top_k=1, require_device=False)


def test_the_answer_prompt_gets_the_reranked_chunk_and_the_trace_counts_both_calls():
    result = _reranked_pipeline(answer(citations=(GOOD,))).ask("where is the screw speed set")
    assert result.status == "answered" and [h.chunk.id for h in result.hits] == ["y"]
    t = result.trace
    assert [c.purpose for c in t.llm_calls] == ["rerank", "answer"]
    assert (t.input_tokens, t.output_tokens) == (1500, 120)
    assert t.rerank_s >= 0 and t.cost_usd == (1500 * 0.15 + 120 * 0.60) / 1e6


def test_eval_reports_rerank_cost_separately_from_answering():
    item = GoldenItem(id="q", question="screw speed", reference_answer="panel",
                      relevant=[RelevantPages(doc="m.pdf", pages=[7])])
    results = run_eval(_reranked_pipeline(answer(citations=(GOOD,))), None, [item])
    s = summarize(results, [1])
    assert (s.rerank_input_tokens, s.rerank_output_tokens) == (500, 20)
    assert (s.answer_input_tokens, s.answer_output_tokens) == (1000, 100)  # rerank not mixed in
    assert s.rerank_cost_usd == (500 * 0.15 + 20 * 0.60) / 1e6
    assert s.hit_at_k[1] == 1.0  # the golden page reached the top slot thanks to the rerank


def test_retrieval_only_eval_still_counts_the_rerank_tokens():
    item = GoldenItem(id="q", question="screw speed", reference_answer="panel",
                      relevant=[RelevantPages(doc="m.pdf", pages=[7])])
    results = run_eval(_reranked_pipeline("unused"), None, [item], retrieval_only=True)
    s = summarize(results, [1])
    assert s.rerank_input_tokens == 500 and s.hit_at_k[1] == 1.0 and s.answer_input_tokens == 0
