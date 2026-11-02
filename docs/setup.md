# Setup guide

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

## Evaluate

```bash
python -m docrag eval eval/golden_smoke.jsonl --llm openai            # 3 questions
python -m docrag eval eval/golden_sample10.jsonl --llm openai         # 10 questions
python -m docrag eval eval/golden.jsonl --retrieval-only              # 40 questions, retrieval metrics only
python -m docrag eval eval/golden.jsonl --llm openai                  # 40 questions, answers and judge
```

Runs are saved under `data/eval/<timestamp>_<label>/` (`results.jsonl` and `summary.json`). Every answering run spends API tokens; a full 40-question run costs roughly one cent for the answers. Retrieval-only runs cost only the query embeddings (a fraction of a cent), plus the reranker's calls with `--rerank`.

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

## Configuration

Everything has a default in `docrag/config.py` and can be set in `.env` or the environment (see `.env.example`): model names, `GEMINI_MODELS` (the fallback chain), chunk size, `MIN_CHUNK_CHARS`, `TOP_K`, `RETRIEVER`, `STORE`, `DATABASE_URL`, `GEMINI_TIMEOUT_S` (per-request timeout for Gemini and Gemma calls, default 120 s), `GEMMA_MODEL`, and paths.

## Troubleshooting

- **"No index at ..."**: run `index --yes` first (for `--store pgvector`, run it with that flag).
- **"DATABASE_URL is not set"** or **"Cannot connect to Postgres"**: start the container (`docker compose up -d`, wait until it is healthy) and set `DATABASE_URL`, or drop `--store pgvector`.
- **Garbled characters when redirecting output on Windows**: the CLI already writes UTF-8; if another tool re-encodes it, set `PYTHONIOENCODING=utf-8`.
- **Different judge results between runs**: with Gemini the judge is a fallback chain, so the exact model can differ from run to run. Set `GEMINI_MODELS` to a single model to pin it (it applies to any Gemini use, including answering with `--llm gemini`).
