# docrag

Modular RAG over technical equipment documentation (PDFs in `pdfs/`). Answers are grounded in the source material, cited with document, page and section, and the system says so when the documentation does not contain the answer.

Work in progress. See `brief.txt` for the target. The original single-file prototype is kept as `rag_system7.py` (dependencies in `requirements-legacy.txt`).

## Project stance (read this first)

This is a demonstration that some of the brief can be done well, not a finished product. **It does not have to be perfect.** What matters is that for anything that does not work we can say:

1. **what** happens, with evidence from a saved run,
2. **why** it happens (and whether that is measured or only a hypothesis),
3. **what might fix it**, and
4. **the tradeoff** of each fix (cost, latency, risk, complexity).

The section [Known limitations](#known-limitations) does this for everything found so far. When you find a new problem, add it there in the same shape instead of hiding it. Reporting a bad number honestly is better than a tuned good one, and "we chose refusing over guessing" is a stated design decision, not a bug.

### Working agreements for any assistant (human or AI) in this repo

- **Never spend tokens without asking first.** That covers `ask`, `eval` (even `--retrieval-only`, which embeds queries), reranker tests and one-off API checks. Build and test with fakes; re-score saved runs offline (`results.jsonl` has every hit, quote and verdict).
- **Run the full golden set only when the owner explicitly says so.** Use `eval/golden_smoke.jsonl` (3 questions) or `eval/golden_sample10.jsonl` (10) for quick checks, and still ask first. Say roughly what it will cost.
- **Do not re-embed** (`index --yes`) or **commit/push** without being asked. The DKP manual must not be embedded.
- **Do not print the list of devices** to the user: the CLI just asks "Which device is this about?".
- **Say what was and was not verified.** Unit tests use fakes, so "tests pass" does not mean "works against the real API". If a scripted file edit is used, assert the replacement matched; several silently did not.
- Keep CI green: `python -m pytest` must pass with no `.env`, no PDFs and no index.

## Status and handoff (as of 2026-09-25)

**Done and tested** (149 unit tests, all with fakes; real-API checks noted in [Results so far](#results-so-far) and [Known limitations](#known-limitations)):
PDF ingest with outline sections and heading-aware chunking; embedding cache; dense, BM25 and hybrid retrieval; optional LLM reranker (`--rerank`, off by default); required device and device filter; schema-validated answers with per-statement verified citations and "insufficient information" handling; time/token/cost trace; evaluation harness (hit@k, precision, recall, MRR, refusal, LLM judge from another family); golden set plus three small copies of it; GitHub Actions running the unit tests.

**Not committed.** The repo still has one commit (the original prototype). Everything else is untracked or modified in the working tree. Do not commit unasked. When the owner agrees, split it into a few logical commits on a branch. Open questions for that: `pdfs/` (public manuals, 4.8 MB, the CI tests skip cleanly without them), `eval/runs/` (3.1 MB of saved runs, they contain chunk text), and the deleted top-level `*.pdf` files from the prototype.

**Open decisions for the owner**
1. ~~Quote check ignoring table labels~~ **Dropped** by the owner: too tuned to a single dropped answer, and it loosens the trust check. Stays documented in limitation 3.
2. ~~Chunking~~ **Done:** running headers, footers and page numbers are stripped generically (`strip_running_lines` in `docrag/corpus.py`: a line repeated, digits ignored, at the top or bottom of at least 25% of a document's pages). Tiny-chunk merging was **skipped** and is documented as a tradeoff.
3. ~~Re-embed~~ **Done**, then adjusted: 637 chunks first, then short chunks merged (562 chunks, 69 more embedded, DKP gone, old section labels replaced). `data/index` is variant A (limitation 14).
4. ~~Reranker full-set retrieval comparison~~ **Done twice** (pre-merge $0.0109, current index $0.0112): see [Results so far](#results-so-far) and limitation 6. Best hit@1 and MRR, ties dense at hit@4, and loses two questions dense finds (candidate pool and reranker judgment, both documented). An answer-quality run on the full set has not been done. Verdict: **evaluated, not adopted by default**; opt-in with `--rerank`.
5. Docs still to write: an architecture document with a diagram, and final results after the re-embed.
6. Test the Gemma path against the real API (owner asked to be reminded).

### Next task: pgvector (start here in a new session)

**Goal.** A `PgVectorStore` behind the existing `VectorStore` interface (`docrag/vectorstore/base.py`), run with Docker (available). **Non-goal: changing retrieval results.** It stores the same vectors, so top-k must match `NumpyStore` on the same data. Its value is architectural, not quality or speed at this scale (numpy search takes about 1 ms): add and delete documents without rebuilding the index (the upload/management UI in the brief needs this), an indexed device filter instead of an in-memory scan, shared persistent storage for several users, and fit with the brief, which names Postgres/pgvector. Costs: a Docker dependency, connection config, and a Postgres service in CI.

**The contract to implement** (`VectorStore`): `add(chunks, vectors)`, `search(query_vector, k, docs=None) -> list[Hit]` (best first; `docs` restricts to those document file names; return fewer than k if fewer match; empty list for an empty or non-matching store), `docs() -> set[str]`, `chunks() -> list[Chunk]` (BM25 is built in memory from this and stays that way), `__len__`. `Hit.score` is **cosine similarity, higher is better** (`NumpyStore` L2-normalizes and takes a dot product); pgvector's `<=>` is cosine *distance*, so use `score = 1 - distance` and keep the ordering identical. Ties in `NumpyStore` follow insertion order; do the same (order by score, then by insertion id).

**Design point to decide.** The interface has `save(directory)` and classmethod `load(directory)`, which fit file-based stores and do not fit a database. Either make them no-ops for Postgres and construct the store from a connection setting, or refactor the interface (for example a factory that opens a store from config). Callers: `docrag/cli.py` (`cmd_index` builds and saves, `load_store` loads, `build_retriever` and `cmd_search` call `store.chunks()`), `docrag/retrieval/dense.py` (`search`, `docs`).

**Suggested shape.** Settings: `DATABASE_URL` (in `docrag/config.py` and `.env.example`) and a store choice (`--store numpy|pgvector`, default `numpy`). One table with an `index_name` column (mirror `index_dir()` in `cli.py`: embedder name plus chunk size and overlap) so several indexes can coexist; columns for the `Chunk` fields plus `embedding vector(dim)`; dimension from the embedder (1536 for `text-embedding-3-small`). The device filter becomes `WHERE doc = ANY(...)`. Exact scan is fine at this size and keeps results identical to numpy; an HNSW index is approximate and would slightly change results, so if added it must be optional and documented as a tradeoff. Libraries to check and add to `requirements.txt`: `psycopg` (v3) and the `pgvector` Python package. Add `docker-compose.yml` using the `pgvector/pgvector` image and `CREATE EXTENSION vector`.

**Getting vectors in without spending tokens.** `cmd_index` embeds through `CachedEmbedder` (`data/embedding_cache.sqlite`, keyed by model and SHA-256 of `chunk.embedding_text()`), so loading the same chunks into Postgres costs nothing if their text is unchanged. It only costs if chunk text changed (see open decisions 2 and 3). The dry run without `--yes` reports how many texts would need embedding: use it, and ask before `--yes`.

**Tests.** Keep the default suite DB-free so CI stays green without a database. Write a store contract test parametrized over `NumpyStore` and `PgVectorStore`, mark the Postgres one `integration` and skip it unless `DATABASE_URL` is set, and add an equivalence test (same query, same top-k and order as `NumpyStore`, with and without a `docs` filter). Then add a second CI job with a `pgvector/pgvector` service container that sets `DATABASE_URL`. Verify the workflow steps locally as far as possible; it cannot be run here.

**Rules that still apply.** No token spending, re-embedding or commits without asking (see working agreements). Report what was and was not verified, and add anything that does not work to [Known limitations](#known-limitations) with why, fixes and tradeoffs.

## Layout

| Package | Role |
|---|---|
| `docrag/corpus.py` | Read PDFs from `pdfs/`, take section headings from the PDF outline, cut pages at headings, split into chunks that keep doc, page and section |
| `docrag/devices.py` | Device names from file names; the registry that maps a device to its manual |
| `docrag/embeddings/` | `Embedder` interface; OpenAI and Gemini implementations; on-disk cache so text is embedded once |
| `docrag/vectorstore/` | `VectorStore` interface; in-memory cosine store (pgvector planned) |
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
python -m pytest       # no API calls, no embeddings
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

`--llm` is one of `openai`, `gemini`, `gemma`. With `gemini`, models in `GEMINI_MODELS` are tried in order; a 429 (or 404/503) moves to the next model without waiting, and the answering model is printed.

## Retrieval

Three interchangeable retrievers, chosen with `--retriever` on `ask` and `eval`:

| | |
|---|---|
| `dense` (default) | embed the question, nearest chunks by cosine similarity |
| `keyword` | Okapi BM25 over the chunk text, section and document name. English filler words are ignored, numbers and codes such as `HTM10` or `0706hex` are kept. Term statistics come from the chosen device's manual only |
| `hybrid` | both, fused with reciprocal rank fusion (only ranks matter, so no score scaling) |

`dense` stays the default because on the golden set it ordered the top 4 best (see [Results so far](#results-so-far)).

**Reranking.** `--rerank` (with any retriever) fetches `rerank_pool` candidates (default 8), has an LLM rate each 0 to 3 for how well it helps answer the question, and keeps the best `top_k`. `--rerank-llm` picks the model (default `openai`). It costs one extra LLM call per question, reported separately in the trace and in eval. If the reranker fails, the original order is kept and a warning is logged. It is off by default and not yet shown to help (see limitation 6).

`python -m docrag search --device HPP271 "question"` prints the three rankings side by side (scores, matched keywords, and where each fused hit stood in the other two lists). It calls no LLM, only embeds the query.

## How an answer is made trustworthy

1. **Retrieval gate.** If the best chunk scores below `--min-score` (default off), the LLM is not called and the answer is "insufficient".
2. **Schema.** The LLM must return JSON `{sufficient, statements[{text, doc, page, quote}]}`: the answer as short statements, each with its own source quote. Invalid output is repaired by re-asking with the error (up to 3 attempts). An answer with no statements is invalid.
3. **Model refusal.** If the model says the context does not answer the question, that is passed through.
4. **Citation check, per statement.** Each statement's quote must appear verbatim in a retrieved chunk of the same document and page. A statement whose quote cannot be found is dropped from the answer and reported. If no statement survives, the answer becomes "insufficient". This proves quotes are real, not that they support the statement; the eval judge measures that.

Every result carries a status (`answered`, `insufficient`, `error`, `needs_device`), the reason, the verified statements each with its citation and section, any dropped statements, the model that answered, and a trace (time per stage, tokens, cost estimate).

## Evaluation

The owner writes the golden set by hand: JSONL, one question per line. See `eval/golden.jsonl` (30 answerable and 10 unanswerable questions over HMW90, HPP271 and WAA151). Page numbers are **PDF page indexes** (what a PDF viewer's page box shows), not printed page numbers; for HMW90 the index is the printed number plus 2.

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

Measured: hit@k (a relevant page is in the top k), precision@k (share of the top k on a relevant page), page recall@k (share of the golden pages found), MRR; share answered, false refusals, judged correctness and faithfulness; whether a citation lands on a relevant page; share of unanswerable questions correctly refused (and the hallucination rate); dropped statements, errors, models used; latency and tokens/cost split into answering, reranking and judging. Each run is saved to `data/eval/<timestamp>_<label>/` (or `$EVAL_DIR`; past runs are in `eval/runs/`) as `results.jsonl` (everything per question: hits with their dense/keyword/rerank ranks, status and reason, verified and dropped statements, judge verdict and reason, trace) and `summary.json` (config, git commit and metrics), so runs can be compared and re-scored offline.

## Results so far

Golden set of 40 questions, `text-embedding-3-small`, answers by `gpt-4o-mini`, judge Gemini. **Small sample: one question moves a rate by about 3 points**, so read these as direction, not benchmark. The judge is an LLM and is not perfectly stable (see limitation 9).

| Run (device filter on unless noted) | hit@1 | hit@4 | MRR | answered | judged correct | faithful | unanswerable refused |
|---|---|---|---|---|---|---|---|
| dense, **no** device filter | 63% | 80% | 0.70 | 70% | 14/30 | 95% | 10/10 |
| dense, device filter | 80% | 93% | 0.86 | 77% | 17/30 | 100% | 10/10 |

**Current index** is variant A of [limitation 14](#14-the-dense-regression-after-the-re-embed-tiny-chunks-outrank-content) (562 chunks: headers stripped, chunks under 120 characters merged). Retrieval only, 30 answerable questions, device filter, top 4 (runs `data/eval/20260925-0715…`, `…0827…`, `…0828…`, not yet copied to `eval/runs/`):

| retriever | hit@1 | hit@3 | hit@4 | precision@4 | page recall@4 | MRR |
|---|---|---|---|---|---|---|
| dense | 70% | 93% | 93% | 42% | 89% | 0.81 |
| hybrid (RRF) | 67% | 83% | 87% | 38% | 81% | 0.75 |
| hybrid + rerank | 83% | 93% | 93% | 42% | 87% | 0.88 |

Reranking cost $0.0112 for the 40 questions (61,266 in / 3,280 out tokens). Reranking still gives the best hit@1 and MRR (+13 points and +0.07 over dense), but it no longer beats dense at hit@4 (93% both) and page recall@4 is now lower (87% vs 89%). It also loses two answerable questions that dense finds; see [limitation 6](#6-hybrid-and-the-reranker-have-not-beaten-dense-as-a-final-top-4). The table below is the same comparison one step earlier, on the 637-chunk index before the merge (rerank hit@4 97%, MRR 0.89): the difference is two questions and one run each, so read the two as the same result within noise.

**After the re-embed, before the merge** (637 chunks, running headers stripped, DKP gone; retrieval only, 30 answerable questions, device filter, top 4; runs `data/eval/20260925-0658…` to `…0659…`, not yet copied to `eval/runs/`):

| retriever | hit@1 | hit@3 | hit@4 | precision@4 | page recall@4 | MRR |
|---|---|---|---|---|---|---|
| dense | 67% | 90% | 93% | 42% | 89% | 0.78 |
| hybrid (RRF) | 63% | 83% | 87% | 38% | 81% | 0.74 |
| hybrid + rerank | 83% | 97% | 97% | 42% | 91% | 0.89 |

The reranking cost $0.0109 for the 40 questions (58,931 in / 3,496 out tokens). Dense fell from the table below (hit@1 80% to 67%, MRR 0.86 to 0.78) even though nothing about dense retrieval changed except the index; 5 of 30 questions moved (`hmw90-internal-1` from rank 7 to not retrieved, and four HMW90/HPP271 questions from rank 1 down to 2 to 4). The likely cause is tiny chunks, see [limitation 14](#14-the-dense-regression-after-the-re-embed-tiny-chunks-outrank-content); not yet confirmed by a counterfactual. Retrieval-only says nothing about answer quality: that needs a full answer run.

Before the re-embed (old index, kept for comparison). Retrieval-only ablation, device filter on, top 4 (and top 8 for the last column):

| retriever | hit@1 | hit@4 | precision@4 | page recall@4 | MRR | hit@8 |
|---|---|---|---|---|---|---|
| dense | 80% | 93% | 43% | 91% | 0.86 | 97% |
| keyword (BM25) | 63% | 83% | 37% | 79% | 0.72 | 83% |
| hybrid (RRF) | 70% | 87% | 39% | 81% | 0.78 | 100% |

Full dense run cost about $0.0095 for 40 answers, p50 latency 1.6 s, p95 3.3 s (judge tokens are free-tier Gemini and not priced). Precision@4 is low for every retriever partly by design: questions have one or two golden pages but four chunks are retrieved, often from the same page.

## Known limitations

Each item: evidence, why, what might fix it, tradeoffs. "Measured" means seen in a saved run; "Hypothesis" means not tested.

### 1. Answers are often incomplete on multi-cause questions
- **Evidence.** In the last full run 6 of the 23 answered questions were judged not correct; 5 were omissions (for example `hmw90-fail1` gave 1 of 3 documented causes, `hpp271-dont-work` 3 of about 8, `waa151-heater-ice` 1 of 2). Faithfulness was 100%: the gap is what is left out, not what is invented.
- **Why.** *Hypothesis:* `top_k=4` plus the prompt rule "answer only what was asked" makes a small model stop early; the first prompt version (before that rule) listed all three causes for `fail1`. Not A/B tested.
- **Fixes and tradeoffs.** Prompt A/B (cheap, but the same rule also removed padding that used to appear); larger `top_k` or neighbouring-chunk expansion (more context, more tokens and latency, more chance of off-topic statements); a stronger answer model (higher cost).

### 2. False refusals on vague questions
- **Evidence.** 7 of 30 answerable questions were refused in the last full run: 2 retrieval misses, 2 dropped by the citation check, 3 where the right chunk was rank 1 and the model still said "insufficient" (`sensor info is wrong`, `error does not disappear`, `What should I do if the display shows Internal 3?`).
- **Why.** The retrieved text does not clearly match a vague question, and tables are flattened (limitation 3), so the model cannot tell which row applies. Refusing is the chosen behavior: all 10 unanswerable questions were refused and none hallucinated.
- **Fixes and tradeoffs.** A clarifying question ("which error message or symptom?") in the same spirit as the device prompt: good UX, but new behavior to build and evaluate. Loosening the model's refusal (prompt) would answer more vague questions but directly trades away the no-hallucination property. The project decision is to prefer refusing.

### 3. Tables are extracted as flattened text
- **Evidence.** `pypdf` returns a troubleshooting row as running text with the column labels interleaved (`Possible cause: Solution: A value is written…`). Cause/remedy pairing is left to the LLM, and quotes that include the labels fail the verbatim check (`hpp271-manual-purge-modbus` in the reranker smoke run was dropped for exactly this).
- **Why.** Plain text extraction has no notion of table cells.
- **Fixes and tradeoffs.** Ignore the labels (`Possible cause:`, `Solution:`, `Problem:`, `Remedy:`) when comparing quotes: free and precise, slightly looser check. Layout-aware extraction (`pdfplumber`/PyMuPDF tables to markdown): better structure, more dependencies, slower ingest, needs a re-embed, table detection is never perfect. A vision model for tables/figures is the brief's multimodal extension: highest ceiling, highest cost.

### 4. The citation check is strict in two ways
- **Evidence.** (a) A real, contiguous quote longer than the 150-character chunk overlap that straddles a chunk boundary is found in neither chunk and is dropped (`hpp271-insight-reset-loop`: 234-character quote, right chunk at rank 1, answer lost). (b) The label problem in limitation 3. (c) The opposite failure, a check that is too weak: a statement can put a fact in its `text` that its verified `quote` does not contain. In the dense-misses run (`eval/runs/20260925-063604_…`) `hpp271-filtering-factor` answered "register 0308hex" while the cited p.89 quote says nothing about a register number (the register table lists `0308hex` as *Purge interval*; the filtering factor is `030Ahex`). The quote was real and on the page, so it passed; only the judge flagged it. The manual itself is inconsistent here: p.19 says "register 0308hex", the p.89 table says `030Ahex`. A rule "every code in the statement must appear in its quote" would have caught this but would also drop the correct baseline answer, whose quote starts after the register-number column of the same row.
- **Why.** Quotes are checked against each retrieved chunk separately.
- **Fixes and tradeoffs.** Check against the whole cited page's text (zero cost, no re-embed; about 1 of 30 questions affected). Note the check only proves the quote is on the page, not that it supports the statement; the eval judge measures that, and a looser check weakens the guarantee. A statement needing two pages must be two statements.

### 5. Dense retrieval misses codes and vocabulary gaps
- **Evidence.** The bare code `Internal 1` ranks the right table at 7 of 691; "wiped with acetone" ranks the right chunk at 11 (the manual never says acetone, it says "unsuitable chemicals").
- **Why.** Embeddings capture topic, not exact tokens; the chunk with the answer also opens with generic "Troubleshooting" text that dilutes it.
- **Fixes and tradeoffs.** BM25 fixed `Internal 1` (rank 1, and rank 2 in hybrid) but cannot fix the acetone case (no shared word). A reranker or query rewriting could bridge vocabulary, at the cost of an extra LLM call per question and its own errors (limitation 6).

### 6. Hybrid and the reranker have not beaten dense as a final top 4
- **Evidence.** Hybrid is worse than dense at hit@1 (70% vs 80%) but better at recall@8 (100% vs 97%). On chatty questions BM25 matches filler words and, because RRF rewards chunks present in both lists, pushes good dense hits down; stopword removal helped (hybrid MRR 0.72 to 0.78) but did not close the gap. The reranker smoke run (2 answerable questions) helped one question (both golden chunks moved from ranks 3 and 4 to 1 and 2) and hurt another (it gave the top score to a condensation chunk for "i cant switch sensor on"). On the same 3 smoke questions it made p50 latency 2.87 s instead of 2.60 s and p95 6.81 s instead of 2.67 s, and the reranking call cost about as much as answering ($0.0009 vs $0.0008), so it roughly doubles the OpenAI spend per question.
- **Why.** Equal-weight RRF lets keyword noise compete with a strong semantic signal; a small reranking model misjudges vague queries.
- **10-question sample** (`eval/golden_sample10.jsonl`, hybrid + rerank vs dense with no rerank, same index and prompt, `eval/runs/20260925-063046_…`): ordering improved (hit@1 88% to 100%, MRR 0.94 to 1.00) but answers did not: judged correct 5/8 to 4/8, one answer became a refusal, page recall@4 fell 96% to 90%, OpenAI cost went from $0.0026 to $0.0057 (2.2x), latency p50 1.74 s to 2.84 s and p95 5.9 s to 7.3 s. The correctness flips are mostly noise: for `hpp271-no-change-5min` the two answers were nearly identical and the same judge model flipped its verdict; for `hmw90-relay` the golden chunk was in both contexts, the model just wrote 1 statement instead of 4, and a different judge model (3.5 vs 3.1 flash-lite, chosen by the fallback chain) scored it. The refusal (`hpp271-wont-power-on`) is legitimate given its context: the reranker scored an unrelated "reset loop" chunk 3 and the chunk with the power-on causes did not reach the top 4.
- **What this sample cannot show.** Dense already had the golden page in the top 4 for all 8 answerable questions, so there was no recall to gain. The cases where a reranker could help (dense misses such as `hmw90-internal-1`, `hpp271-chemicals-slow`) were not in the random sample. The measured costs (money, latency) are real; the benefit is untested where it matters.
- **Dense-misses run** (`eval/golden_dense_misses.jsonl`, hybrid + rerank, `eval/runs/20260925-063604_…`, about $0.0017 of OpenAI tokens). Note the set was built from an older retrieval run: after the golden-page fix `hpp271-filtering-factor` is *not* a dense miss (dense had it at rank 1 and answered it correctly), so only two questions are true misses. Result: the reranker put the golden page at rank 1 for both. `hmw90-internal-1` (bare code) was then **answered correctly** (dense: refused, right chunk not retrieved). `hpp271-chemicals-slow` was retrieved at rank 1 but the model still refused, which is honest: the manual never says "acetone", so the chunk does not clearly answer the question. `hpp271-filtering-factor`, which dense got right, became wrong: the reranker put p.19 in the context, which contradicts the register table (see limitation 4c). So reranking fixed retrieval where dense failed, but changing the context set can also import conflicting text. With 2 to 3 questions this is illustrative, not proof.
- **Full-set retrieval comparison on the current index** (562 chunks, retrieval only, 30 answerable questions; table in [Results so far](#results-so-far)): hybrid + rerank has the best hit@1 (83% vs 70% dense) and MRR (0.88 vs 0.81), ties dense at hit@3 and hit@4 (93%), and is slightly worse at page recall@4 (87% vs 89%). No reranker call failed or fell back (every question has rerank scores). **What went wrong, question by question.** Two answerable questions that dense finds are lost, for two different reasons: (1) `hmw90-relay` (dense rank 2) is not in hybrid's top 8 at all, so the reranker never sees it: the reranker can only reorder the 8 candidates hybrid hands it (`rerank_pool`), so hybrid's recall (97% at 8, 87% at 4) is its ceiling and hybrid's keyword noise costs it here. (2) `waa151-dont-work` (dense rank 1) is inside the pool, at hybrid rank 8, and the reranker scored four other chunks above it; in the pre-merge run the same question was reranked to rank 2, so this is close to run-to-run and chunk-boundary variation, which was not measured. The reranker did fix the questions where hybrid or dense had the golden chunk lower down (`hmw90-internal-1` to rank 1, `hpp271-chemicals-slow` to 1, `hpp271-manual-purge-modbus` from 3 to 1, `hpp271-reading-incorrect` from 3 to 1), and `hpp271-filtering-factor` stays at rank 2 (dense had it at 1, and this is the case where rerank imported a conflicting chunk, see limitation 4c). **Fixes and tradeoffs for the two losses.** Feed the reranker the union of the dense and hybrid top candidates so it cannot lose what dense found (more tokens per call, so a higher reranking bill that was not estimated, and tuned by looking at these 30 questions, so it could overfit); a larger `rerank_pool` (more tokens, same idea); or use dense as the candidate generator when `rerank` is on (simple; loses BM25's help with exact codes). None of these was tried, because each needs another run and the sample is too small to tell a real gain from noise. **Earlier run, pre-merge index (637 chunks):** hybrid + rerank beat dense on every retrieval metric: hit@1 83% vs 67%, hit@4 97% vs 93%, page recall@4 91% vs 89%, MRR 0.89 vs 0.78. It cost $0.0109 per 40 questions in reranking calls alone, on top of the answering cost, and added a sequential LLM call per question (latency was not measured in retrieval-only runs; the smoke and sample runs above showed p50 +0.1 to +1.1 s and a worse p95). Better ranking did not translate into better answers in the 10-question sample, and a full answer run with the reranker has not been done, so it stays opt-in.
- **Fixes and tradeoffs.** Test the reranker on the known dense misses (cheap, targeted); weighted RRF (risk of overfitting to 30 questions); a stronger reranker model (cost); a full-set comparison before deciding (about $0.01, needs explicit go-ahead). Honest fallback: keep the reranker as "evaluated, not adopted". It is opt-in either way: nothing is built or called without `--rerank`.

### 7. Queries without a device get contaminated by other manuals
- **Evidence.** Without the device filter, 3 of the 5 questions with no golden page retrieved had the right chunk at rank 1 or 2 inside its own manual (hit@4 80% without, 93% with the filter). Manuals share wording ("restart the transmitter").
- **Fix implemented.** A device is required and retrieval is limited to its manual.
- **Tradeoffs.** The user must know the device name, and cross-manual questions are not possible. Inferring the device from the question (classifier) or asking is the alternative; asking was chosen.

### 8. The index was stale (fixed 2026-09-25)
Re-embedded after stripping running headers (637 chunks, DKP removed). The text below describes the old state.
- **Evidence.** `data/index` still contains the DKP manual (which must not be embedded) and pre-fix section labels (for example a table row on p.77 labelled `10.2` instead of `10.1`). Heading-splitting also created many small chunks (691 to 859, DKP included); they were first left unmerged, but the re-embed showed that was a mistake (limitation 14).
- **Fix.** `python -m docrag index --embedder openai --yes` rebuilds from `pdfs/`. Estimated at most about $0.0025 (unchanged text comes from the cache). Decide chunking questions first (strip running headers such as `User's Guide ____ 38 ____`, merge tiny chunks) so it is embedded once. The owner asked not to re-embed until BM25/hybrid/reranker work was done.

### 9. The evaluation itself is small and noisy
- **Evidence.** 40 questions; the judge flipped one verdict between runs (`hpp271-wont-power-on`) and gave one clearly wrong verdict (`hpp271-error-level-change`, marked unfaithful although the answering excerpt was in its context). Golden page numbers were wrong for HMW90 until fixed (printed vs PDF index), which had understated retrieval scores.
- **Fixes and tradeoffs.** More questions (the owner writes them by hand, which is slow); a judge from another family or two judges (cost); spot-check the `reason` field in `results.jsonl`. Use the numbers for direction only.

### 10. Provider details that are not fully verified
- Gemini fallback chain: only `gemini-3.1-flash-lite` and `gemini-3.5-flash-lite` have been seen answering in real runs; other names in `GEMINI_MODELS` are skipped on 404 if they do not exist, and 429 fallback was only exercised with fakes.
- `gemma-3-12b-it` through the Gemini API has never been called; it cannot enforce JSON, so it relies on validation and retry.
- `docrag/pricing.py` prices come from memory and must be checked; Gemini/Gemma show cost as unknown (free tier).

### 11. Scale
- The vector store and BM25 are in memory and exact. Fine for hundreds to a few thousand chunks (search takes about 1 ms); not a design for large collections. pgvector would add document add/delete without a full rebuild, indexed metadata filtering, and shared persistence, but would not change retrieval quality.

### 12. The `min_score` gate is uncalibrated
- The retrieval gate ("best cosine below X means insufficient, without calling the LLM") exists and is tested but is **off by default** (`min_score=0`). It was never calibrated. Evidence is only from a few observed queries, not a systematic measurement: a vague question whose top hit was the correct chunk scored 0.32, while the irrelevant top hit for the bare code `Internal 1` also scored 0.32 (clear questions reach 0.5 to 0.7). *Hypothesis:* scores of right and wrong top hits overlap, so one threshold would either refuse real vague questions or let junk through. The top scores of the unanswerable questions were never recorded, so this is untested.
- **Fixes and tradeoffs.** Calibrate per embedder on a larger golden set (costs a run and needs more unanswerable questions); the LLM's own "insufficient" already handles most cases, at the price of one LLM call.

### 13. Repository housekeeping
- The top-level `rag_system7.py`, `example output.txt`, `lessons learned.txt` and `requirements-legacy.txt` belong to the original single-file prototype and are kept as history. The new code is the `docrag/` package.
- Console output is now forced to UTF-8, because the manuals contain characters such as `‣` and `…` that crash Python's default cp1252 stdout when output is redirected on Windows.

### 14. The dense regression after the re-embed: tiny chunks outrank content
- **Evidence (measured, from saved runs and the index; no API calls).** After the re-embed dense hit@1 fell from 80% to 67% and MRR from 0.86 to 0.78 (hit@4 stayed 93%). 5 of 30 answerable questions moved. In the new index 82 of 637 chunks (13%) have under 100 characters of text: 31 are running-header remnants (`Chapter 4 ____ Operation`), 11 are heading-only chunks (`Display`, `Measurement Settings`) and 40 are other short pieces (front matter, captions). Those chunks take 11 of the 120 top-4 slots in the new run against 1 of 120 in the old run, and only 1 of the 11 sat on a golden page. Per question: `hmw90-internal-3-sentence` ("display shows Internal 3") lost rank 1 to the one-word chunk `Display` (score 0.479 against 0.440 for the golden chunk); `hmw90-relay` had `Chapter 4 ____ Operation` chunks at ranks 1 and 3 and the golden page at rank 4; `hmw90-fail1` fell from rank 1 to 3 (its old golden chunk no longer starts where it did and a `Show Command List` chunk outscored it). `hpp271-wont-power-on` is not a tiny-chunk case: the golden chunks kept their scores and a content chunk about sensor vitality rose slightly above them. `hmw90-internal-1` (a bare code) was not analysed.
- **Why.** *Mechanism, likely but not isolated:* a chunk's embedding text is `document | section label` plus the text, so for a very short chunk the section label dominates its vector, and a short query that shares words with the label (`display`, `relay`, `operation`) matches it better than a long content chunk. Heading-aware splitting made the heading-only chunks, and the header stripping (`strip_running_lines`) missed the per-chapter running heads because each appears on only about 4 to 15% of pages, under the 25% rule. The old index had neither the heading split nor as many short chunks. The 5 moves are also part noise: 30 questions, and changing every chunk boundary reshuffles near-tied scores (0.41 to 0.44 apart in three of the cases).
- **Fixes and tradeoffs.** (a) Merge chunks shorter than a minimum (for example 100 to 150 characters) into the following chunk on the same page: standard, cheap, needs a re-embed of the changed chunks only (well under $0.001); a merged chunk can start with a heading or a header remnant, which is milder than a chunk that is only that. (b) Lower the repetition threshold for header stripping: it would catch the chapter heads, but on these manuals the 5 to 10% band also holds real table headers (`Register`, `number` at the top of continued tables), so it trades recall for precision; not recommended alone. (c) Leave the section label out of the embedded text for short chunks: fixes the mechanism directly but loses the section signal that helps other questions; would need measuring.
- **Counterfactual (measured, dense retrieval only, same 30 questions, top 4).** Fix (a) is implemented (`min_chunk_chars=120`, merge into the next chunk on the page) and two variants were built into scratch indexes, embedding only the changed chunks (about $0.001 in total). **A**: merge and header stripping, 562 chunks. **B**: merge without header stripping, 633 chunks (this answers "would leaving the underscore headers in be better?").

| index | chunks | hit@1 | hit@3 | hit@4 | page recall@4 | MRR |
|---|---|---|---|---|---|---|
| old (before heading split and stripping) | n/a | 80% | 93% | n/a* | n/a | 0.86 |
| after re-embed, no merge | 637 | 67% | 90% | 93% | 89% | 0.78 |
| **A**: merge, headers stripped | 562 | 70% | 93% | 93% | 89% | 0.81 |
| **B**: merge, headers kept | 633 | 70% | 90% | 93% | 88% | 0.80 |

  \*The old run retrieved 8 and reported hit@1, hit@3 and hit@8 (97%); hit@1, hit@3 and MRR are comparable, hit@4 was not reported.
- **What this shows.** The merge fixed one of the five moved questions (`hmw90-internal-3-sentence`, rank 2 back to 1) and half fixed another (`hmw90-relay`, rank 4 to 2, in A only). Header stripping versus keeping the headers made no measurable difference (0.81 vs 0.80 MRR, and one question apart); A is kept because it has fewer junk chunks, not because it is proven better. The recovery is partial: hit@1 is still 70% against 80%. The remaining three moved questions are not tiny-chunk cases. `hmw90-internal-1` is the bare query "Internal 1": every top score is 0.31 to 0.32, so the old rank 7 was luck and the ranking there is noise. `hmw90-fail1` is a chunk-dilution effect: the old golden chunk was a focused 494-character chunk holding the "Transmitter does not recognize a valid serial command" row, while the new boundaries put that row into a 713-character chunk that also covers hysteresis and the service port, and its score fell from 0.673 to 0.578. `hpp271-wont-power-on` moved from rank 1 to 2 with unchanged scores because a sensor-vitality chunk edged above it. A merged chunk can still open with a header remnant (`Chapter 4 ____`, seen at rank 1 for `fail1`), so the noise is reduced, not removed.
- **Conclusion.** The regression was about one part chunking side effects (fixable, partly fixed), one part changed chunk boundaries that reshuffle near-tied scores (not fixable without re-tuning chunk size, which would overfit a 30-question set). The remaining 10-point gap in hit@1 is three questions. Chunk size and overlap were not explored; an equally valid finding is that at this sample size the old and new indexes are not distinguishable beyond these five cases.

## Not built

Web UI and document upload/management; multimodal retrieval (drawings, figures, scans, tables as images); factory-data integration (alarms, sensors); a local LLM (Gemma is reached through the Gemini API only, to show what a local-class model can do); pgvector (Docker is available); a clarifying-question flow. The brief treats the text MVP first and these as extensions.

## Tests

```bash
python -m pytest          # 149 tests, all with fakes: no API calls, no PDFs or index needed
```

GitHub Actions (`.github/workflows/tests.yml`) runs them on every push to `main` and on pull requests, in a clean Python 3.12 environment with the API keys set to blank, so a test that tried to call OpenAI or Gemini would fail instead of spending tokens.
