# Setup, usage and tests

How to install, run, evaluate and test docrag. What the project is and how well it works is in the [README](../README.md).

## Requirements

- Python 3.12 or newer.
- An OpenAI API key (answers and embeddings). A Google API key is optional (Gemini answers and the default judge for OpenAI answers).
- The manuals are in `pdfs/`: `HMW90 user guide.pdf`, `HPP271 user guide.pdf` and `WAA151 user guide.pdf`, all publicly available. The golden set (`eval/golden.jsonl`) refers to these three, so they must stay there to reproduce the results.
- Docker only if you want the Postgres store.

## Install

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in OPENAI_API_KEY (and GOOGLE_API_KEY if you use Gemini)
python -m pytest            # unit tests: no API calls, no PDFs, no index, no database
```

## Build the index

```bash
# Dry run: chunks the PDFs and reports how many embeddings are needed and the upper-bound cost. Embeds nothing.
python -m docrag index --embedder openai

# Create it. Text that was embedded before comes from data/embedding_cache.sqlite, so re-running is cheap.
python -m docrag index --embedder openai --yes
```

The index goes to `data/index/openai_text-embedding-3-small_c800o150/`. For the three manuals this is about 560 chunks and costs a few tenths of a cent.

## Ask

```bash
python -m docrag ask --device HPP271 "How do I ...?"
python -m docrag ask                       # interactive: asks which device first
python -m docrag ask --device HPP271 --verbose "..."     # also prints the retrieved chunk text
python -m docrag search --device HPP271 "Internal 1"     # dense, keyword and fused results side by side, no LLM
```

A device is required. Without one, nothing is processed and the CLI asks "Which device is this about?". A device name is the start of the PDF file name (`HPP271 user guide.pdf` is `HPP271`); a `pdfs/devices.json` can name devices explicitly.

Useful flags: `--llm openai|gemini|gemma`, `--retriever dense|keyword|hybrid`, `--rerank` (an LLM re-orders the candidates; costs an extra call per question), `--top-k N`.

**Devices.** A manual covering several devices yields one name each. To name them yourself, put a `devices.json` in `pdfs/`: `{"file.pdf": ["NAME", "OTHER"]}`. Without a known device the question is not embedded, retrieved or sent to an LLM.

**Gemini fallback.** With `--llm gemini`, models in `GEMINI_MODELS` are tried in order; a 429, 404, 500, 502, 503 or 504 (or a request timeout, `GEMINI_TIMEOUT_S`) moves to the next model, and the answering model is printed.

## Retrieval options

Three interchangeable retrievers, chosen with `--retriever` on `ask` and `eval`:

| | |
|---|---|
| `dense` (default) | embed the question, nearest chunks by cosine similarity |
| `keyword` | Okapi BM25 over the chunk text, section and document name. English filler words are ignored, numbers and codes such as `HTM10` or `0706hex` are kept. Term statistics come from the chosen device's manual only |
| `hybrid` | both, fused with reciprocal rank fusion (only ranks matter, so no score scaling) |

**Reranking.** `--rerank` (with any retriever) fetches `rerank_pool` candidates (default 8), has an LLM rate each 0 to 3 for how well it helps answer the question, and keeps the best `top_k`. `--rerank-llm` picks the model (default `openai`). It costs one extra LLM call per question, reported separately in the trace and in eval. If the reranker fails, the original order is kept and a warning is logged.

`python -m docrag search --device HPP271 "question"` prints the three rankings side by side (scores, matched keywords, and where each fused hit stood in the other two lists). It calls no LLM, only embeds the query.

## Evaluate

```bash
python -m docrag eval eval/golden_smoke.jsonl --llm openai            # 3 questions
python -m docrag eval eval/golden_sample10.jsonl --llm openai         # 10 questions
python -m docrag eval eval/golden.jsonl --retrieval-only              # 40 questions, retrieval metrics only
python -m docrag eval eval/golden.jsonl --llm openai                  # 40 questions, answers and judge
```

Runs are saved under `data/eval/<timestamp>_<label>/` (`results.jsonl` and `summary.json`). Every answering run spends API tokens; a full 40-question run costs roughly one cent for the answers. Retrieval-only runs cost only the query embeddings (a fraction of a cent), plus the reranker's calls with `--rerank`.

More options:

```bash
python -m docrag eval eval/golden.jsonl --retrieval-only --retriever hybrid   # hit@k, precision, recall, MRR; no answer LLM
python -m docrag eval eval/golden.jsonl --llm gemma --judge openai             # the judge defaults to a different family than --llm
python -m docrag eval eval/golden.jsonl --llm openai --judge none
python -m docrag eval eval/golden_sample10.jsonl --llm openai --retriever hybrid --rerank
python -m docrag eval eval/golden_dense_misses.jsonl --llm openai --retriever hybrid --rerank   # the questions dense misses
```

The smoke set (3 questions) and the sample (a seeded random sample stratified by device: 8 answerable, 2 unanswerable) are copied unedited from the golden set, and tests keep them in sync, pages included.

**Golden set format.** JSONL, one question per line:

```json
{"id": "q1", "question": "...", "device": "HMW90", "answerable": true, "reference_answer": "...", "relevant": [{"doc": "HMW90 user guide.pdf", "pages": [38, 39]}]}
{"id": "q2", "question": "a question the manuals do not answer", "device": "HMW90", "answerable": false}
```

Page numbers are **PDF page indexes** (what a PDF viewer's page box shows), not printed page numbers; for HMW90 the index is the printed number plus 2. Every item needs a `device` (it is what the user would type), and the eval refuses to start if one is missing or unknown. `--no-device-filter` searches all manuals instead, to measure what the filter adds.

**What is measured.** hit@k, precision@k, page recall@k and MRR; share answered, false refusals, judged correctness and faithfulness; whether a citation lands on a relevant page; share of unanswerable questions correctly refused (and the hallucination rate); dropped statements, errors, models used; latency and tokens/cost split into answering, reranking and judging. What each one means is in [evaluation-explained.md](evaluation-explained.md).

**What is saved.** `results.jsonl` holds everything per question: hits with their dense, keyword and rerank ranks, status and reason, verified and dropped statements, judge verdict and reason, and the trace. `summary.json` holds the config, git commit and metrics. Runs can be compared and re-scored offline. Set `$EVAL_DIR` to save elsewhere.

## Use Postgres instead of files (optional)

```bash
docker compose up -d                                   # Postgres 16 + pgvector on localhost:5433
export DATABASE_URL=postgresql://docrag:docrag@localhost:5433/docrag    # or set it in .env
python -m docrag index --embedder openai --store pgvector --yes        # loads the vectors (cached ones cost nothing)
python -m docrag ask --store pgvector --device HPP271 "..."            # also works for search and eval
```

`--store` defaults to `numpy`. Without the flag, nothing imports the Postgres packages or connects to a database, so you can drop the flag at any time to go back. Tests for the Postgres store are skipped unless `DATABASE_URL` is set:

```bash
DATABASE_URL=postgresql://docrag:docrag@localhost:5433/docrag python -m pytest -m integration
```

Stop the database with `docker compose down` (add `-v` to delete the stored data).

Checked against a real Postgres: 720 searches on the real index and full evaluation runs returned the same chunks in the same order as the in-memory store. Search takes about 3 to 7 ms against 0.2 to 0.3 ms in memory. Measurements, design and gaps: [pgvector.md](pgvector.md).

## Tests

```bash
python -m pytest          # 172 tests, all with fakes: no API calls, no PDFs, index or database needed (13 more are skipped without a database)

# The Postgres store tests (skipped unless DATABASE_URL is set):
docker compose up -d
DATABASE_URL=postgresql://docrag:docrag@localhost:5433/docrag python -m pytest -m integration
```

GitHub Actions (`.github/workflows/tests.yml`) runs them on every push to `main` and on pull requests, in a clean Python 3.12 environment with the API keys set to blank, so a test that tried to call OpenAI or Gemini would fail instead of spending tokens.

The original single-file prototype is kept as `rag_system7.py`.

## Configuration

Everything has a default in `docrag/config.py` and can be set in `.env` or the environment (see `.env.example`): model names, `GEMINI_MODELS` (the fallback chain), chunk size, `MIN_CHUNK_CHARS`, `TOP_K`, `RETRIEVER`, `STORE`, `DATABASE_URL`, `GEMINI_TIMEOUT_S` (per-request timeout for Gemini and Gemma calls, default 120 s), `GEMMA_MODEL`, and paths.

## Troubleshooting

- **"No index at ..."**: run `index --yes` first (for `--store pgvector`, run it with that flag).
- **"DATABASE_URL is not set"** or **"Cannot connect to Postgres"**: start the container (`docker compose up -d`, wait until it is healthy) and set `DATABASE_URL`, or drop `--store pgvector`.
- **Garbled characters when redirecting output on Windows**: the CLI already writes UTF-8; if another tool re-encodes it, set `PYTHONIOENCODING=utf-8`.
- **Different judge results between runs**: with Gemini the judge is a fallback chain, so the exact model can differ from run to run. Set `GEMINI_MODELS` to a single model to pin it (it applies to any Gemini use, including answering with `--llm gemini`).
