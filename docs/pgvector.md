# Postgres / pgvector store

`--store pgvector` swaps the file-based vector store for Postgres with the pgvector extension. **The default is unchanged**: without the flag (or `STORE=numpy`) nothing imports `psycopg` or connects to a database, so the numpy path is exactly what it was and the flag can simply be dropped to revert. Non-goal, and verified: it does not change retrieval results.

```bash
docker compose up -d                                   # Postgres 16 + pgvector on localhost:5433
export DATABASE_URL=postgresql://docrag:docrag@localhost:5433/docrag   # or put it in .env
python -m docrag index --embedder openai --store pgvector        # dry run, embeds nothing
python -m docrag index --embedder openai --store pgvector --yes  # loads the vectors
python -m docrag ask --store pgvector --device HPP271 "How do I ...?"   # also: search, eval
```

`index --store pgvector --yes` re-embeds nothing that is in the embedding cache, so loading the current index costs $0 (the dry run says `Chunks needing new embeddings: 0`).

**How it works.** `docrag/vectorstore/pgvector_store.py` implements the existing `VectorStore` interface. One `chunks` table holds every index, told apart by `index_name` (the same name the numpy store uses for its directory: embedder plus chunk size and overlap), so several embedders or chunk settings can coexist; the vector column has no fixed dimension for that reason. Vectors are stored L2-normalized like `NumpyStore` does, the score is `1 - cosine distance` (cosine similarity, higher is better), the device filter is `WHERE doc = ANY(...)`, and ties come back in insertion order. The scan is exact, with no HNSW or IVFFlat index, because an approximate index could change the top-k.

**Verified** (against a real Postgres 16 with pgvector, in Docker on this machine):
- A store contract test runs against both stores (ranking, cosine scores independent of vector length, device filter, fewer than k, empty store, insertion order, unicode and missing sections, several `add` calls). Postgres-only tests cover ties, separate indexes with the same chunk ids and different dimensions, and rebuilds.
- **Equivalence on the real index:** the 562 chunks and vectors of `data/index` were loaded into Postgres and 720 searches compared with `NumpyStore` (60 queries made from stored vectors plus noise, with no filter and with each of the three manuals as filter, at k = 1, 4 and 8). Every result list had the same chunks in the same order, and scores agreed within 1e-5. This does not prove the two stay identical for every possible query: exact score ties are broken by insertion order in Postgres, whereas numpy's order on exact ties is not specified.
- The chunks, documents and order read back from Postgres equal the numpy index (`load_store` with and without the flag).
- Search latency on this index (100 queries, p50 / p95): numpy 0.2 to 0.3 ms; Postgres 7.0 / 12.3 ms without a device filter and 2.9 / 4.4 ms with one. An early version took about 50 ms because the query vector was passed twice, in the select list and in `ORDER BY`; computing the distance once in a subquery fixed it (found by timing query variants, not guessed).

**Not verified.** The GitHub Actions job `pgvector` (a `pgvector/pgvector:pg16` service container running `pytest -m integration`) has never run, because it cannot run before the first push. Its equivalence test skips there, since `data/` is not in the repository, so CI covers the contract tests only. An end-to-end `eval` through `--store pgvector` was not run: it would embed the 40 queries (a small API spend) and the retrieval-level equivalence above already covers what the store returns. Nothing was tested with a Postgres that already holds other data, or under concurrent writers.

**Update, end-to-end check (2026-09-25, later the same day).** The "not verified" item above was then run. Through `--store pgvector`: (a) a full 40-question retrieval-only run returned the **same chunks in the same order as numpy for all 40 questions** (hit@1 70%, hit@3 93%, hit@4 93%, MRR 0.806, identical to the numpy run); (b) the 10-question answer run returned identical chunks, **identical answer text for all 10, and identical judge verdicts** (2 of 8 correct, $0.0022, same as the numpy run of that sample). Scores between two runs differed by up to 5e-4, but two numpy runs on the same index differ by the same 5.3e-4 on the same question (`hpp271-zero-ma-startup`), so that is the embedding API returning slightly different query vectors from call to call, not the store. This still says nothing about concurrent writers, a database that holds other data, or larger collections, and the CI job has still not run.

**Tradeoffs and what is not built.** It adds a Docker dependency, a connection setting and a CI service; at this size it is about 10 to 30 times slower per search than numpy (still small next to the query embedding and the LLM call). The interface's `save(directory)` is a no-op and `load(directory)` raises, because a database is opened from a connection and not from a path; the smaller change was chosen over refactoring the interface into a factory. The reason to have it is architectural: rows can be added and removed without rebuilding a file, the device filter is an indexed column, and several users can share one persistent store. **The interface has no delete method yet**, so removing one document (what a document-upload UI would need) is possible in SQL but not exposed in code. BM25 is still built in memory from `store.chunks()`, so it is not in Postgres.


See also: [setup guide](setup.md), [architecture](architecture.md), [limitations](limitations.md).
