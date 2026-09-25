"""Runs the golden set through the pipeline and scores it."""

import math
from collections import Counter

from pydantic import BaseModel

from docrag.eval.golden import GoldenItem, RelevantPages
from docrag.eval.judge import JudgeVerdict, judge_answer
from docrag.llm.base import LLM, AllModelsFailed, LLMCall
from docrag.models import Chunk, Hit
from docrag.rag.pipeline import RagPipeline
from docrag.rag.schema import RagResult
from docrag.rag.structured import StructuredOutputError
from docrag.rag.trace import total_cost


class NoLLM(LLM):
    """Stands in for the LLM in retrieval-only runs, which must never call one."""

    def generate(self, prompt, system=None, json_mode=False):
        raise RuntimeError("LLM called in a retrieval-only run")


def is_relevant(chunk: Chunk, relevant: list[RelevantPages]) -> bool:
    return any(chunk.doc == r.doc and chunk.page in r.pages for r in relevant)


def first_relevant_rank(hits: list[Hit], relevant: list[RelevantPages]) -> int | None:
    """1-based rank of the first retrieved chunk that lies on a relevant page."""
    return next((i for i, h in enumerate(hits, 1) if is_relevant(h.chunk, relevant)), None)


def precision_at_k(hits: list[Hit], relevant: list[RelevantPages], k: int) -> float:
    """Share of the top-k retrieved chunks that lie on a relevant page."""
    top = hits[:k]
    return sum(is_relevant(h.chunk, relevant) for h in top) / len(top) if top else 0.0


def page_recall_at_k(hits: list[Hit], relevant: list[RelevantPages], k: int) -> float:
    """Share of the golden pages that at least one of the top-k chunks comes from."""
    golden = {(r.doc, page) for r in relevant for page in r.pages}
    found = {(h.chunk.doc, h.chunk.page) for h in hits[:k]} & golden
    return len(found) / len(golden)


class ItemResult(BaseModel):
    item: GoldenItem
    result: RagResult | None = None  # None in retrieval-only runs
    hits: list[Hit]
    retrieval_rank: int | None = None  # answerable questions only
    citation_on_relevant_page: bool | None = None  # answered, answerable questions only
    verdict: JudgeVerdict | None = None
    judge_model: str | None = None
    judge_error: str | None = None
    judge_calls: list[LLMCall] = []
    retrieval_calls: list[LLMCall] = []  # reranker calls in a retrieval-only run


class Summary(BaseModel):
    # counts
    n_answerable: int
    n_unanswerable: int
    # retrieval, over answerable questions
    hit_at_k: dict[int, float | None]  # share of questions with a relevant page in the top k
    precision_at_k: dict[int, float | None]  # mean share of the top-k chunks on a relevant page
    recall_at_k: dict[int, float | None]  # mean share of the golden pages found in the top k
    mrr: float | None
    # answers, over answerable questions
    answered_rate: float | None
    false_refusal_rate: float | None
    correct_rate: float | None  # judge-correct / all answerable (refusals count as wrong)
    faithful_rate: float | None  # judge-faithful / answered
    citation_on_relevant_page_rate: float | None  # over answered
    answers_with_unsupported_statement_rate: float | None  # over answered: a statement was dropped
    # unanswerable questions
    correct_refusal_rate: float | None
    hallucination_rate: float | None  # answered when it should have refused
    # bookkeeping
    errors: int
    judge_errors: int
    mean_attempts: float | None
    models_used: dict[str, int]
    # cost of the run. Cost is None when any call used a model with no known price.
    latency_p50_s: float | None  # per question, end to end (answering side only)
    latency_p95_s: float | None
    answer_input_tokens: int
    answer_output_tokens: int
    answer_cost_usd: float | None
    rerank_input_tokens: int
    rerank_output_tokens: int
    rerank_cost_usd: float | None
    judge_input_tokens: int
    judge_output_tokens: int
    judge_cost_usd: float | None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _rate_of(total: float, count: int) -> float | None:
    return total / count if count else None


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile, q in [0, 1]."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered), max(1, math.ceil(q * len(ordered)))) - 1]


def summarize(results: list[ItemResult], ks: list[int], judge_used: bool = False) -> Summary:
    answerable = [r for r in results if r.item.answerable]
    unanswerable = [r for r in results if not r.item.answerable]

    def rank_le(r: ItemResult, k: int) -> bool:
        return r.retrieval_rank is not None and r.retrieval_rank <= k

    mrr = _rate(sum(1 / r.retrieval_rank for r in answerable if r.retrieval_rank), len(answerable))

    with_llm = [r for r in results if r.result is not None]
    answered = [r for r in with_llm if r.result.status == "answered"]
    answered_ans = [r for r in answered if r.item.answerable]
    judged = [r for r in answered_ans if r.verdict]
    models = Counter(r.result.model for r in with_llm if r.result.model)
    attempts = [r.result.attempts for r in with_llm if r.result.attempts]
    have_llm = bool(with_llm)
    answer_calls = [c for r in with_llm for c in r.result.trace.llm_calls if c.purpose != "rerank"]
    rerank_calls = [c for r in with_llm for c in r.result.trace.llm_calls if c.purpose == "rerank"]
    rerank_calls += [c for r in results for c in r.retrieval_calls]
    judge_calls = [c for r in results for c in r.judge_calls]
    latencies = [r.result.trace.total_s for r in with_llm]

    return Summary(
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
        hit_at_k={k: _rate(sum(rank_le(r, k) for r in answerable), len(answerable)) for k in ks},
        precision_at_k={
            k: _rate_of(sum(precision_at_k(r.hits, r.item.relevant, k) for r in answerable), len(answerable))
            for k in ks
        },
        recall_at_k={
            k: _rate_of(sum(page_recall_at_k(r.hits, r.item.relevant, k) for r in answerable), len(answerable))
            for k in ks
        },
        mrr=mrr,
        answered_rate=_rate(len(answered_ans), len(answerable)) if have_llm else None,
        false_refusal_rate=(
            _rate(sum(r.result.status == "insufficient" for r in answerable if r.result), len(answerable))
            if have_llm else None
        ),
        correct_rate=(
            _rate(sum(r.verdict.correct for r in judged), len(answerable))
            if have_llm and judge_used else None
        ),
        faithful_rate=_rate(sum(r.verdict.faithful for r in judged), len(judged)) if judge_used else None,
        citation_on_relevant_page_rate=_rate(
            sum(bool(r.citation_on_relevant_page) for r in answered_ans), len(answered_ans)
        ),
        answers_with_unsupported_statement_rate=_rate(
            sum(bool(r.result.rejected_statements) for r in answered), len(answered)
        ),
        correct_refusal_rate=(
            _rate(sum(r.result.status == "insufficient" for r in unanswerable if r.result), len(unanswerable))
            if have_llm else None
        ),
        hallucination_rate=(
            _rate(sum(r.result.status == "answered" for r in unanswerable if r.result), len(unanswerable))
            if have_llm else None
        ),
        errors=sum(r.result.status == "error" for r in with_llm),
        judge_errors=sum(bool(r.judge_error) for r in results),
        mean_attempts=_rate(sum(attempts), len(attempts)),
        models_used=dict(models),
        latency_p50_s=percentile(latencies, 0.5),
        latency_p95_s=percentile(latencies, 0.95),
        answer_input_tokens=sum(c.input_tokens or 0 for c in answer_calls),
        answer_output_tokens=sum(c.output_tokens or 0 for c in answer_calls),
        answer_cost_usd=total_cost(answer_calls),
        rerank_input_tokens=sum(c.input_tokens or 0 for c in rerank_calls),
        rerank_output_tokens=sum(c.output_tokens or 0 for c in rerank_calls),
        rerank_cost_usd=total_cost(rerank_calls) if rerank_calls else None,
        judge_input_tokens=sum(c.input_tokens or 0 for c in judge_calls),
        judge_output_tokens=sum(c.output_tokens or 0 for c in judge_calls),
        judge_cost_usd=total_cost(judge_calls) if judge_calls else None,
    )


def run_item(
    pipeline: RagPipeline,
    judge: LLM | None,
    item: GoldenItem,
    retrieval_only: bool,
    use_device: bool = True,
) -> ItemResult:
    device = item.device if use_device else None
    retrieval_calls: list[LLMCall] = []
    if retrieval_only:
        retrieval = pipeline.retrieve_full(item.question, device=device)
        hits, result, retrieval_calls = retrieval.hits, None, retrieval.llm_calls
    else:
        result = pipeline.ask(item.question, device)
        hits = result.hits
    out = ItemResult(
        item=item,
        result=result,
        hits=hits,
        retrieval_calls=retrieval_calls,
        retrieval_rank=first_relevant_rank(hits, item.relevant) if item.answerable else None,
    )
    if result is None or result.status != "answered" or not item.answerable:
        return out

    out.citation_on_relevant_page = any(
        any(r.doc == c.doc and c.page in r.pages for r in item.relevant) for c in result.citations
    )
    if judge is not None:
        try:
            out.verdict, out.judge_model = judge_answer(
                judge, item, result, calls=out.judge_calls
            )
        except (StructuredOutputError, AllModelsFailed) as e:
            out.judge_error = str(e)  # the item stays in the run, just unjudged
    return out


def run_eval(
    pipeline: RagPipeline,
    judge: LLM | None,
    items: list[GoldenItem],
    retrieval_only: bool = False,
    on_progress=None,
    use_device: bool = True,
) -> list[ItemResult]:
    results = []
    for i, item in enumerate(items, 1):
        results.append(run_item(pipeline, judge, item, retrieval_only, use_device))
        if on_progress:
            on_progress(i, len(items), results[-1])
    return results
