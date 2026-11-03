# docrag

A question-answering assistant for technical equipment manuals, built as modular RAG and demonstrated on three public sensor manuals (PDFs in `pdfs/`). Every statement in an answer is cited with document, page, section and a verified quote, and the system says so when the manuals do not contain the answer.

## Results

From a full run of 40 test questions (30 answerable, 10 unanswerable), with `gpt-4o-mini` writing the answers and a Gemini model judging them:

| | |
|---|---|
| Right page among the 4 retrieved passages | 93% (28 of 30) |
| Answered | 80% (24 of 30) |
| Judged correct | 47% (14 of 30); 14 to 20 of 30 usable on hand review |
| Unanswerable questions answered with an invented answer | 0 of 10 |
| Cost and speed | about $0.0002 per question, 1.74 s typical wait |

**What the numbers show.** Search is not the bottleneck: only 2 of the 16 failures came from search. Most came from the answer step, where the model left out details it had been given or refused vague questions, and 2 were correct answers thrown away by the quote check. A later run with `gpt-4.1-mini` raised correct answers to 17 of 30 (hand-corrected) and halved false refusals, but answered 1 unanswerable question ([run-gpt-4.1-mini.md](docs/run-gpt-4.1-mini.md)).

The sample is small (one question moves a rate by about 3 points), so these numbers show direction, not a benchmark. Full breakdown, failure analysis and what each metric means: [evaluation explained](docs/evaluation-explained.md). Every measurement made during development: [results](docs/results.md).

## Highlights

- **Cited, verified answers.** Each statement carries its own source quote and is kept only if that quote is found word for word in the retrieved text.
- **Refuses rather than guesses.** A wrong instruction for industrial equipment is worse than none. All 10 unanswerable questions were refused in every full run with `gpt-4o-mini`; the price is that 20 to 23% of answerable questions were refused too.
- **Measured, not assumed.** Dense, BM25 keyword, hybrid and LLM-reranked retrieval were compared on the same test set. Hybrid, often called best practice, did worse here, so dense stays the default.
- **Swappable parts.** Embedders, vector stores, retrievers and LLMs sit behind small interfaces. An optional Postgres/pgvector store returns the same results as the in-memory one.
- **Tested and cost-aware.** 172 tests run with fakes in CI, with API keys blanked so no test can spend tokens. Embeddings are cached, indexing is a dry run unless confirmed, and every answer records time, tokens and cost.

## How an answer is made trustworthy

1. **Retrieval gate.** If the best chunk scores below `--min-score` (default off), the LLM is not called and the answer is "insufficient".
2. **Schema.** The LLM must return JSON `{sufficient, statements[{text, doc, page, quote}]}`: the answer as short statements, each with its own source quote. Invalid output is repaired by re-asking with the error (up to 3 attempts). An answer with no statements is invalid.
3. **Model refusal.** If the model says the context does not answer the question, that is passed through.
4. **Citation check, per statement.** Each statement's quote must appear verbatim in a retrieved chunk of the same document and page. A statement whose quote cannot be found is dropped from the answer and reported. If no statement survives, the answer becomes "insufficient". This proves quotes are real, not that they support the statement; the eval judge measures that.

Every result carries a status (`answered`, `insufficient`, `error`, `needs_device`), the reason, the verified statements each with its citation and section, any dropped statements, the model that answered, and a trace (time per stage, tokens, cost estimate).

## Layout

| Package | Role |
|---|---|
| `docrag/corpus.py` | Read PDFs from `pdfs/`, take section headings from the PDF outline, cut pages at headings, split into chunks that keep doc, page and section |
| `docrag/devices.py` | Device names from file names; the registry that maps a device to its manual |
| `docrag/embeddings/` | `Embedder` interface; OpenAI and Gemini implementations; on-disk cache so text is embedded once |
| `docrag/vectorstore/` | `VectorStore` interface; in-memory cosine store (default) and a Postgres/pgvector store |
| `docrag/retrieval/` | `Retriever` interface: dense, BM25 keyword, hybrid (RRF fusion), and an optional LLM reranker |
| `docrag/llm/` | `LLM` interface; OpenAI, Gemini (model fallback chain), Gemma via Gemini API |
| `docrag/rag/` | The pipeline, schema-validated answers, citation verification, trace (time, tokens, cost) |
| `docrag/eval/` | Golden-set loading, metrics, LLM judge, run reports |
| `docrag/factory.py` | The only place that maps provider names to implementations |

Diagrams of indexing, answering and evaluation: [architecture](docs/architecture.md).

## Known limitations

Every problem found is written up with evidence from a saved run, its cause (and whether that is measured or only a hypothesis), possible fixes and their tradeoffs, in [docs/limitations.md](docs/limitations.md). The main ones:

- **Answers are often incomplete on multi-cause questions,** and nothing warns the user. Of the 10 answered but not correct, 2 are acceptable on review, 4 partial but usable, 2 not usable and 2 about the wrong fault.
- **False refusals on vague questions.** The price of never guessing; a clarifying-question flow is not built.
- **Tables are extracted as flattened text,** so pairing a cause with its remedy is left to the LLM.
- **The citation check proves a quote is real, not that it supports the statement,** and it is strict about quotes that straddle a chunk boundary.
- **The evaluation is small, noisy and not independent:** 40 questions, an LLM judge, and test questions mostly written by Claude, the assistant that helped build the project.
- **A search-score threshold cannot detect unanswerable questions.** They match the manuals as strongly as answerable ones, so refusal is left to the model.

**Not built:** web UI and document upload, multimodal retrieval (drawings, figures, scans), a truly local LLM (Gemma is reached through an API only), and a clarifying-question flow.

## Running it

Python 3.12+.

```bash
pip install -r requirements.txt
cp .env.example .env                                   # then fill in the keys
python -m docrag index --embedder openai --yes         # build the index (cached)
python -m docrag ask --device HPP271 "How do I ...?"   # a device is required
```

Usage, retrieval options, evaluation, the Postgres store and tests: [setup guide](docs/setup.md).

## How the project went

The [decision log](docs/decision-log.md) records what was decided, why, what happened, and which decisions turned out to be wrong.
