"""Saving eval runs to disk and printing them."""

import json
import subprocess
from datetime import datetime
from pathlib import Path

from docrag.eval.runner import ItemResult, Summary


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def new_run_dir(base: Path, label: str) -> Path:
    path = base / f"{datetime.now():%Y%m%d-%H%M%S}_{label}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_run(directory: Path, results: list[ItemResult], summary: Summary, config: dict) -> None:
    """results.jsonl has everything per question; summary.json has config + metrics."""
    with open(directory / "results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")
    payload = {"config": config, "summary": summary.model_dump(mode="json")}
    (directory / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _secs(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}s"


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _usd(x: float | None) -> str:
    return "n/a (no price for model)" if x is None else f"${x:.4f}"


def format_summary(s: Summary, config: dict) -> str:
    lines = [
        f"Questions: {s.n_answerable} answerable, {s.n_unanswerable} unanswerable",
        f"Embedder: {config['embedder']}  top_k={config['top_k']}  min_score={config['min_score']}",
    ]
    if config.get("llm"):
        lines.append(f"Answering: {config['llm']}  Judge: {config.get('judge') or 'none'}")
    if config.get("rerank"):
        lines.append(f"Reranking: {config['rerank']}")
    lines += ["", "Retrieval (answerable questions)"]
    lines.append("  k     hit@k   precision@k   page recall@k")
    lines += [
        f"  {k:<5} {_pct(v):<7} {_pct(s.precision_at_k.get(k)):<13} {_pct(s.recall_at_k.get(k))}"
        for k, v in s.hit_at_k.items()
    ]
    lines.append(f"  MRR   {'n/a' if s.mrr is None else f'{s.mrr:.3f}'}")
    lines.append("  (hit@k: a relevant page is in the top k; precision: share of the top k on a relevant page;")
    lines.append("   page recall: share of the golden pages found)")
    if config.get("llm"):
        lines += [
            "",
            "Answers (answerable questions)",
            f"  answered              {_pct(s.answered_rate)}",
            f"  false refusals        {_pct(s.false_refusal_rate)}",
            f"  correct (judge)       {_pct(s.correct_rate)}",
            f"  faithful (judge)      {_pct(s.faithful_rate)}",
            f"  cited a relevant page {_pct(s.citation_on_relevant_page_rate)}",
            "",
            "Unanswerable questions",
            f"  correctly refused     {_pct(s.correct_refusal_rate)}",
            f"  hallucinated an answer {_pct(s.hallucination_rate)}",
            "",
            "Grounding / reliability",
            f"  answers with a dropped statement {_pct(s.answers_with_unsupported_statement_rate)}",
            f"  errors {s.errors}, judge errors {s.judge_errors}, "
            f"mean attempts {'n/a' if s.mean_attempts is None else f'{s.mean_attempts:.2f}'}",
            f"  models used: {s.models_used}",
            "",
            "Latency and cost",
            f"  latency per question  p50 {_secs(s.latency_p50_s)}  p95 {_secs(s.latency_p95_s)}",
            f"  answering  {s.answer_input_tokens:,} in / {s.answer_output_tokens:,} out tokens, cost {_usd(s.answer_cost_usd)}",
        ]
        if config.get("judge"):
            lines.append(
                f"  judging    {s.judge_input_tokens:,} in / {s.judge_output_tokens:,} out tokens, cost {_usd(s.judge_cost_usd)}"
            )
    if s.rerank_input_tokens or s.rerank_output_tokens:
        if not config.get("llm"):
            lines += ["", "Reranking cost"]
        lines.append(
            f"  reranking  {s.rerank_input_tokens:,} in / {s.rerank_output_tokens:,} out tokens, cost {_usd(s.rerank_cost_usd)}"
        )
    return "\n".join(lines)
