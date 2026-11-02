# docrag

Modular RAG over technical equipment documentation, demonstrated on three public sensor manuals (PDFs in `pdfs/`). Answers are grounded in the source material, cited with document, page and section, and the system says so when the documentation does not contain the answer.

Work in progress.  The original single-file prototype is kept as `rag_system7.py` 

Docs: [architecture](docs/architecture.md) (diagrams of indexing, answering and evaluation), [setup guide](docs/setup.md), [evaluation explained](docs/evaluation-explained.md) (what the metrics and results mean, in plain language), [results](docs/results.md), [known limitations](docs/limitations.md), [Postgres/pgvector store](docs/pgvector.md), [decision log](docs/decision-log.md).

## This is a demo 

**It does not have to be perfect.** What matters is that for anything that does not work we can say:

1. **what** happens, with evidence from a saved run,
2. **why** it happens (and whether that is measured or only a hypothesis),
3. **what might fix it**, and
4. **the tradeoff** of each fix (cost, latency, risk, complexity).

[Known limitations](#known-limitations) does this for everything found so far, in full in [docs/limitations.md](docs/limitations.md). Reporting a bad number honestly is better than a tuned good one, and "we chose refusing over guessing" is a stated design decision, not a bug.

## Highlights

- **Cited, verified answers.** Every statement in an answer carries document, page, section and a quote, and is kept only if its quote is found verbatim in the retrieved text.
- **Refuses rather than guesses.** All 10 unanswerable test questions were refused in every full run, with no invented answer. The price is that 20 to 23% of answerable questions were refused too.
- **Device-scoped retrieval.** The user names the device (manual); retrieval is limited to it. On an earlier index this raised hit@4 from 80% to 93%.
- **Swappable parts.** Embedders, vector stores, retrievers and LLMs sit behind small interfaces. The vector store is in-memory by default, with an optional Postgres/pgvector store that returns the same results.
- **Retrieval options that were measured, not assumed.** Dense (default), BM25 keyword, hybrid, and an optional LLM reranker, compared on a golden set of test questions.
- **Evaluation harness.** hit@k, precision, recall, MRR, refusal rate, and an LLM judge from a different model family; every run is saved so it can be re-scored offline.
- **Cost-aware.** Embeddings are cached, indexing is a dry run unless `--yes`, and each answer records time, tokens and cost.
- **Honest about limits.** Known problems are written up with evidence, cause, possible fixes and tradeoffs, and the decision log includes the decisions that turned out wrong.

## Layout

| Package | Role |
|---|---|
| `docrag/corpus.py` | Read PDFs from `pdfs/`, take section headings from the PDF outline, cut pages at headings, split into chunks that keep doc, page and section |
| `docrag/devices.py` | Device names from file names; the registry that maps a device to its manual |
| `docrag/embeddings/` | `Embedder` interface; OpenAI and Gemini implementations; on-disk cache so text is embedded once |
| `docrag/vectorstore/` | `VectorStore` interface; in-memory cosine store (default) and a Postgres/pgvector store (`--store pgvector`) |
| `docrag/retrieval/` | `Retriever` interface: dense, BM25 keyword, hybrid (RRF fusion), and an optional LLM reranker |
| `docrag/llm/` | `LLM` interface; OpenAI, Gemini (model fallback chain), Gemma via Gemini API |
| `docrag/rag/` | The pipeline, schema-validated answers, citation verification, trace (time, tokens, cost) |
| `docrag/eval/` | Golden-set loading, metrics, LLM judge, run reports |
| `docrag/factory.py` | The only place that maps provider names to implementations |

## Setup

Python 3.12+.

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in the keys
python -m pytest       # no API calls, no embeddings, no database
```

## Usage

```bash
# 1. Dry run: chunk the PDFs, show how many embeddings and what it would cost. Embeds nothing.
python -m docrag index --embedder openai

# 2. Create the embeddings (cached, so re-chunking only embeds changed text)
python -m docrag index --embedder openai --yes

# 3. Ask. A device is required: retrieval is limited to that device's manual.
python -m docrag ask --device HPP271 --llm gemini "How do I ...?"
python -m docrag ask --llm gemma          # interactive: asks which device first
python -m docrag ask "How do I ...?"      # no device: nothing is processed, it asks which device
```

**Devices.** A device name comes from the PDF file name (`HMW90 user guide.pdf` is device `HMW90`). A manual covering several devices yields one name each. To name them yourself, put a `devices.json` in `pdfs/`: `{"file.pdf": ["NAME", "OTHER"]}`. Without a known device the question is not embedded, retrieved or sent to an LLM: the answer is just "Which device is this about?".

`--llm` is one of `openai`, `gemini`, `gemma`. With `gemini`, models in `GEMINI_MODELS` are tried in order; a 429, 404, 500, 502, 503 or 504 (or a request timeout, `GEMINI_TIMEOUT_S`) moves to the next model, and the answering model is printed. Full setup, including Postgres: [docs/setup.md](docs/setup.md).

## Retrieval

Three interchangeable retrievers, chosen with `--retriever` on `ask` and `eval`:

| | |
|---|---|
| `dense` (default) | embed the question, nearest chunks by cosine similarity |
| `keyword` | Okapi BM25 over the chunk text, section and document name. English filler words are ignored, numbers and codes such as `HTM10` or `0706hex` are kept. Term statistics come from the chosen device's manual only |
| `hybrid` | both, fused with reciprocal rank fusion (only ranks matter, so no score scaling) |

`dense` stays the default because on the golden set it ordered the top 4 best (see [Results](#results)).

**Reranking.** `--rerank` (with any retriever) fetches `rerank_pool` candidates (default 8), has an LLM rate each 0 to 3 for how well it helps answer the question, and keeps the best `top_k`. `--rerank-llm` picks the model (default `openai`). It costs one extra LLM call per question, reported separately in the trace and in eval. If the reranker fails, the original order is kept and a warning is logged. It is off by default and not shown to help enough to adopt (see [limitation 6](docs/limitations.md#6-hybrid-and-the-reranker-have-not-beaten-dense-as-a-final-top-4)).

`python -m docrag search --device HPP271 "question"` prints the three rankings side by side (scores, matched keywords, and where each fused hit stood in the other two lists). It calls no LLM, only embeds the query.

## How an answer is made trustworthy

1. **Retrieval gate.** If the best chunk scores below `--min-score` (default off), the LLM is not called and the answer is "insufficient".
2. **Schema.** The LLM must return JSON `{sufficient, statements[{text, doc, page, quote}]}`: the answer as short statements, each with its own source quote. Invalid output is repaired by re-asking with the error (up to 3 attempts). An answer with no statements is invalid.
3. **Model refusal.** If the model says the context does not answer the question, that is passed through.
4. **Citation check, per statement.** Each statement's quote must appear verbatim in a retrieved chunk of the same document and page. A statement whose quote cannot be found is dropped from the answer and reported. If no statement survives, the answer becomes "insufficient". This proves quotes are real, not that they support the statement; the eval judge measures that.

Every result carries a status (`answered`, `insufficient`, `error`, `needs_device`), the reason, the verified statements each with its citation and section, any dropped statements, the model that answered, and a trace (time per stage, tokens, cost estimate).

## Evaluation

The golden set was mostly written by Claude, the AI assistant that helped build the project (Claude Sonnet 5, through Claude Code. This is inferred: the log of the session that wrote the questions (on or before 2026-09-24) was not kept, and the project's Claude Code sessions from late September 2026 all used Sonnet 5), so it is not an independent test ([limitation 9](docs/limitations.md#9-the-evaluation-itself-is-small-and-noisy)). It is JSONL, one question per line. See `eval/golden.jsonl` (30 answerable and 10 unanswerable questions over HMW90, HPP271 and WAA151). Page numbers are **PDF page indexes** (what a PDF viewer's page box shows), not printed page numbers; for HMW90 the index is the printed number plus 2.

```json
{"id": "q1", "question": "...", "device": "HMW90", "answerable": true, "reference_answer": "...", "relevant": [{"doc": "HMW90 user guide.pdf", "pages": [38, 39]}]}
{"id": "q2", "question": "a question the manuals do not answer", "device": "HMW90", "answerable": false}
```

```bash
# Retrieval only: no answer LLM, just query embeddings. hit@k, precision, recall, MRR.
python -m docrag eval eval/golden.jsonl --retrieval-only --retriever hybrid

# Full run. The judge defaults to a different model family than --llm.
python -m docrag eval eval/golden.jsonl --llm openai            # judged by gemini
python -m docrag eval eval/golden.jsonl --llm gemma --judge openai
python -m docrag eval eval/golden.jsonl --llm openai --judge none

# Small sets, copied unedited from the golden set (tests keep them in sync, pages included):
# 3 questions (smoke), and 10 (a seeded random sample stratified by device: 8 answerable, 2 unanswerable).
python -m docrag eval eval/golden_smoke.jsonl --llm openai
python -m docrag eval eval/golden_sample10.jsonl --llm openai --retriever hybrid --rerank
# The questions dense retrieval misses (used to test the reranker where it could help)
python -m docrag eval eval/golden_dense_misses.jsonl --llm openai --retriever hybrid --rerank
```

Every item needs a `device` (it is what the user would type), and the eval refuses to start if one is missing or unknown. `--no-device-filter` searches all manuals instead, to measure what the filter adds.

Measured: hit@k (a relevant page is in the top k), precision@k (share of the top k on a relevant page), page recall@k (share of the golden pages found), MRR; share answered, false refusals, judged correctness and faithfulness; whether a citation lands on a relevant page; share of unanswerable questions correctly refused (and the hallucination rate); dropped statements, errors, models used; latency and tokens/cost split into answering, reranking and judging. Each run is saved to `data/eval/<timestamp>_<label>/` (or `$EVAL_DIR`; runs saved during development were kept locally in `eval/runs/` and are not published) as `results.jsonl` (everything per question: hits with their dense/keyword/rerank ranks, status and reason, verified and dropped statements, judge verdict and reason, trace) and `summary.json` (config, git commit and metrics), so runs can be compared and re-scored offline.

## Results

Golden set of 40 questions (30 answerable, 10 unanswerable) over three manuals; `text-embedding-3-small`, answers by `gpt-4o-mini` at temperature 0, judged by a Gemini model. **Small sample: one question moves a rate by about 3 points**, so read these as direction, not benchmark. The judge is an LLM and is not perfectly stable.

**Retrieval** (current index of 562 chunks, device filter on, top 4, 30 answerable questions). hit@4 and page recall@4 are the numbers to read. Precision@4 is weak here, because four chunks are always fetched while some questions have only two relevant ones, so it is useful only to compare methods:

| retriever | hit@1 | hit@3 | hit@4 | precision@4 | page recall@4 | MRR |
|---|---|---|---|---|---|---|
| dense (default) | 70% | 93% | 93% | 42% | 89% | 0.81 |
| hybrid (RRF) | 67% | 83% | 87% | 38% | 81% | 0.75 |
| hybrid + rerank | 83% | 93% | 93% | 42% | 87% | 0.88 |

Reranking gives the best hit@1 and MRR, ties dense at hit@4, and costs about as much again per question ($0.0112 for 40 questions in reranking calls). That gain is ordering only: both find 28 of 30, because reranking rescues two questions dense misses and loses two it finds. The model reads all four chunks anyway, answers did not improve on the 10-question sample (5 of 8 correct to 4 of 8), and no full answer run with reranking exists, so it stays opt-in.

**Answers** (full run of all 40 questions, dense retrieval):

| | |
|---|---|
| answered | 80% (24 of 30) |
| judged correct (recall of right answers) | 47% (14 of 30) |
| precision of right answers | 58% (14 of 24) |
| accuracy | 60% (24 of 40: 14 right answers, 10 right refusals) |
| faithful (judge found every claim supported) | 100% |
| citation on a relevant page | 88% |
| unanswerable questions refused | 10 of 10 |
| answering cost / p50 / p95 latency | $0.0089 / 1.74 s / 3.58 s |

Faithful does not mean right, and this score is generous: the judge passed answers that quoted real text but applied it to the wrong fault, so on a strict reading about 21 of 24 were faithful. No answer contained made-up text. Correct is out of all 30 answerable questions, with refusals counted as wrong; faithful and citations are out of the 24 answered. Cost covers the answering calls only, at prices in `docrag/pricing.py` that have not been verified.

An earlier full run on an older index scored 57% correct (17 of 30). The difference is within run-to-run spread (two earlier full runs on that index scored 47% and 57%), and the judge model also differed between runs. What the numbers do show: answers are often incomplete on multi-cause questions, and refusing costs some answerable questions. An answer counts as answered if one quote verifies, and nothing checks completeness, which is why 80% answered became 47% correct. Of the 16 failures, 10 came from answer writing, 2 were correct answers thrown away by the quote check, 2 were search misses and 2 were strict grading; the judge also graded the same behaviour differently in different questions, so usefully answered is better read as 14 to 20 of 30. Every table, the earlier index, the ablations and the per-question analysis are in [docs/results.md](docs/results.md). What each metric means is in [docs/evaluation-explained.md](docs/evaluation-explained.md).

## Known limitations

Each is written up with evidence, cause, possible fixes and tradeoffs in [docs/limitations.md](docs/limitations.md). The numbering follows that file; items 8 (a stale index, since fixed) and 13 (repository housekeeping) are left out here.

1. **Answers are often incomplete on multi-cause questions.** Details are left out, and nothing warns the user. In the last full run 10 of 24 answered questions were judged not correct. On review, 2 of them are acceptable, 4 partial but usable, 2 not usable and 2 about the wrong fault.
2. **False refusals on vague questions.** The price of never guessing; a clarifying-question flow is not built.
3. **Tables are extracted as flattened text.** Pairing a cause with its remedy is left to the LLM.
4. **The citation check proves a quote is real, not that it supports the statement,** and it is strict about quotes that straddle a chunk boundary.
5. **Dense retrieval misses exact codes and vocabulary gaps.** BM25 helps for codes only.
6. **Hybrid and the reranker have not beaten dense as a final top 4.** The reranker improves hit@1 and MRR but costs about double per question and can lose questions dense finds.
7. **Without a device, other manuals contaminate results,** which is why a device is required.
9. **The evaluation is small, noisy and not independent:** 40 questions, an LLM judge, and test questions mostly written by Claude.
10. **Provider details are not fully verified.** Gemma through the Gemini API was slow and unreliable when tested; the Gemini fallback chain was only partly exercised.
11. **Scale.** Exact in-memory search by default; the pgvector option was not tested at large sizes.
12. **The `min_score` gate is uncalibrated** and off by default. Unanswerable questions match the manuals as strongly as answerable ones, so a score threshold is unlikely to separate them.
14. **Chunking side effects.** Tiny heading-only chunks outranked real content after heading-aware splitting; merging short chunks recovered part of it (dense hit@1 is 70% against 80% on the earlier index).

## Optional: Postgres with pgvector

`--store pgvector` keeps the vectors in Postgres instead of files. Without the flag nothing imports the Postgres packages or connects to a database, so the default path is unchanged.

```bash
docker compose up -d                                   # Postgres 16 + pgvector on localhost:5433
export DATABASE_URL=postgresql://docrag:docrag@localhost:5433/docrag   # or put it in .env
python -m docrag index --embedder openai --store pgvector --yes        # loads the vectors (cached ones cost nothing)
python -m docrag ask --store pgvector --device HPP271 "How do I ...?"  # also: search, eval
```

Checked against a real Postgres: 720 searches on the real index and full evaluation runs returned the same chunks in the same order as the in-memory store. Search takes about 3 to 7 ms against 0.2 to 0.3 ms in memory. Measurements, design and gaps: [docs/pgvector.md](docs/pgvector.md).

## Not built

Web UI and document upload/management; multimodal retrieval (drawings, figures, scans, tables as images); a local LLM (Gemma is reached through the Gemini API only, to show what a local-class model can do); a clarifying-question flow.

## Tests

```bash
python -m pytest          # 172 tests, all with fakes: no API calls, no PDFs, index or database needed (13 more are skipped without a database)

# The Postgres store tests (skipped unless DATABASE_URL is set):
docker compose up -d
DATABASE_URL=postgresql://docrag:docrag@localhost:5433/docrag python -m pytest -m integration
```

GitHub Actions (`.github/workflows/tests.yml`) runs them on every push to `main` and on pull requests, in a clean Python 3.12 environment with the API keys set to blank, so a test that tried to call OpenAI or Gemini would fail instead of spending tokens.

## How the project went

The [decision log](docs/decision-log.md) records what was decided, why, what happened, and which decisions turned out to be wrong.
