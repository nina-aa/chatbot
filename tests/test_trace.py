import json

from docrag.eval.golden import GoldenItem, RelevantPages
from docrag.eval.runner import percentile, run_eval, summarize
from docrag.llm.base import LLM, LLMCall, LLMResponse
from docrag.pricing import call_cost
from docrag.rag.trace import Trace, total_cost
from tests.test_eval import GOOD, _pipeline, _verdict
from tests.test_pipeline import answer


class UsageLLM(LLM):
    """Scripted replies that also report token usage."""

    def __init__(self, *replies, model="gpt-4o-mini", tokens_in=1000, tokens_out=100):
        self.replies, self.model, self.tin, self.tout = list(replies), model, tokens_in, tokens_out

    def generate(self, prompt, system=None, json_mode=False):
        return LLMResponse(
            text=self.replies.pop(0), model=self.model,
            input_tokens=self.tin, output_tokens=self.tout,
        )


def test_call_cost_uses_price_table_and_is_unknown_otherwise():
    assert call_cost("gpt-4o-mini", 1_000_000, 1_000_000) == 0.15 + 0.60
    assert call_cost("gemini-3.5-flash", 10, 10) is None
    assert call_cost("gpt-4o-mini", None, 10) is None


def test_total_cost_is_none_if_any_call_is_unpriced():
    priced = LLMCall(model="gpt-4o-mini", input_tokens=1000, output_tokens=100, latency_s=1)
    free = LLMCall(model="gemma-3-12b-it", input_tokens=1000, output_tokens=100, latency_s=1)
    assert total_cost([priced]) is not None
    assert total_cost([priced, free]) is None
    assert total_cost([]) is None


def test_ask_records_time_tokens_and_cost():
    result = _pipeline(UsageLLM(answer(citations=(GOOD,)))).ask("screw speed operator panel")
    t = result.trace
    assert len(t.llm_calls) == 1 and t.input_tokens == 1000 and t.output_tokens == 100
    assert t.cost_usd == (1000 * 0.15 + 100 * 0.60) / 1e6
    assert t.total_s >= t.llm_s >= 0 and t.embed_s >= 0 and t.search_s >= 0


def test_tokens_from_failed_attempts_are_still_counted():
    result = _pipeline(UsageLLM("garbage", answer(citations=(GOOD,)))).ask("q")
    assert result.attempts == 2 and len(result.trace.llm_calls) == 2
    assert result.trace.input_tokens == 2000


def test_tokens_are_counted_even_when_every_attempt_fails():
    result = _pipeline(UsageLLM("x", "x", "x")).ask("q")
    assert result.status == "error" and result.trace.input_tokens == 3000


def test_low_score_gate_uses_no_tokens():
    p = _pipeline(UsageLLM())
    p._min_score = 0.99
    t = p.ask("nothing related at all").trace
    assert t.llm_calls == [] and t.cost_usd is None and t.input_tokens == 0


def test_trace_is_serialised_with_totals():
    dumped = json.loads(Trace(llm_calls=[LLMCall(model="m", input_tokens=5, output_tokens=7, latency_s=1.5)]).model_dump_json())
    assert dumped["input_tokens"] == 5 and dumped["output_tokens"] == 7 and dumped["llm_s"] == 1.5


def test_percentile_nearest_rank():
    assert percentile([], 0.5) is None
    assert percentile([5.0], 0.95) == 5.0
    values = [float(i) for i in range(1, 101)]
    assert percentile(values, 0.5) == 50.0 and percentile(values, 0.95) == 95.0


def test_eval_summary_separates_answering_and_judging_cost():
    item = GoldenItem(id="hit", question="operator panel screw speed", reference_answer="panel",
                      relevant=[RelevantPages(doc="m.pdf", pages=[7])])
    llm = UsageLLM(answer(citations=(GOOD,)), tokens_in=2000, tokens_out=200)
    judge = UsageLLM(_verdict(), model="gemini-x", tokens_in=3000, tokens_out=50)
    s = summarize(run_eval(_pipeline(llm), judge, [item]), [1], judge_used=True)
    assert (s.answer_input_tokens, s.answer_output_tokens) == (2000, 200)
    assert s.answer_cost_usd is not None
    assert (s.judge_input_tokens, s.judge_output_tokens) == (3000, 50)
    assert s.judge_cost_usd is None  # gemini has no price
    assert s.latency_p50_s is not None and s.latency_p95_s is not None
