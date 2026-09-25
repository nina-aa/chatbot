import argparse
import re
import sys
from pathlib import Path

from docrag.config import Settings
from docrag.corpus import chunk_pages, load_corpus
from docrag.devices import DeviceRegistry, load_overrides
from docrag.embeddings.cache import CachedEmbedder
from docrag.eval.golden import check_devices, check_docs_exist, load_golden
from docrag.eval.report import format_summary, git_commit, new_run_dir, write_run
from docrag.eval.runner import NoLLM, run_eval, summarize
from docrag.factory import (
    EMBEDDER_PROVIDERS, LLM_PROVIDERS, default_judge, make_embedder, make_llm, same_family,
)
from docrag.llm.base import LLM
from docrag.rag.pipeline import RagPipeline
from docrag.retrieval.base import Retriever
from docrag.retrieval.bm25 import Bm25Retriever, matched_terms
from docrag.retrieval.dense import DenseRetriever
from docrag.retrieval.hybrid import HybridRetriever, fuse
from docrag.retrieval.rerank import LLMReranker, RerankingRetriever
from docrag.vectorstore.base import VectorStore
from docrag.vectorstore.numpy_store import NumpyStore

RETRIEVERS = ("dense", "keyword", "hybrid")
STORES = ("numpy", "pgvector")

# USD per 1M input tokens. Gemini embeddings are free-tier limited, not priced here.
EMBEDDING_PRICE_PER_M = {"openai:text-embedding-3-small": 0.02, "openai:text-embedding-3-large": 0.13}


def index_name(embedder_name: str, size: int, overlap: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", embedder_name)
    return f"{slug}_c{size}o{overlap}"


def index_dir(s: Settings, embedder_name: str, size: int, overlap: int) -> Path:
    return s.index_dir / index_name(embedder_name, size, overlap)


def store_kind(args, s: Settings) -> str:
    kind = getattr(args, "store", None) or s.store
    if kind not in STORES:
        raise SystemExit(f"Unknown store '{kind}'. Choose one of: {', '.join(STORES)}.")
    return kind


def open_pgvector(s: Settings, name: str) -> VectorStore:
    """Connects to Postgres. Imported here so the default numpy path needs neither the
    packages nor a database."""
    if not s.database_url:
        raise SystemExit(
            "DATABASE_URL is not set. Start Postgres with `docker compose up -d` and set it "
            "(see .env.example), or leave --store at its default (numpy)."
        )
    import psycopg

    from docrag.vectorstore.pgvector_store import PgVectorStore

    try:
        return PgVectorStore(s.database_url, name)
    except psycopg.OperationalError as e:
        raise SystemExit(f"Cannot connect to Postgres: {e}") from e


def cmd_index(args, s: Settings) -> None:
    size, overlap = args.chunk_size or s.chunk_size, args.chunk_overlap or s.chunk_overlap
    pages = load_corpus(s.pdf_dir)
    chunks = chunk_pages(pages, size, overlap, s.min_chunk_chars)
    embedder = CachedEmbedder(make_embedder(args.embedder, s), s.embedding_cache_path)
    target = index_dir(s, embedder.name, size, overlap)
    kind = store_kind(args, s)
    name = index_name(embedder.name, size, overlap)

    texts = [c.embedding_text() for c in chunks]
    new = embedder.count_uncached(texts)
    tokens = sum(len(t) for t in set(texts)) // 4  # rough: 4 chars per token
    price = EMBEDDING_PRICE_PER_M.get(embedder.name)
    print(f"Documents: {len({p.doc for p in pages})}, pages: {len(pages)}, chunks: {len(chunks)}")
    print(f"Chunks needing new embeddings: {new} (the rest come from the cache)")
    if price is not None and new:
        print(f"Estimated cost: ~${tokens * price / 1e6:.4f} (upper bound, ~{tokens:,} tokens)")
    print(f"Index location: {target}" if kind == "numpy" else f"Index location: Postgres, index '{name}'")

    if not args.yes:
        print("\nDry run only. Nothing embedded. Re-run with --yes to create embeddings.")
        return
    vectors = embedder.embed_documents(texts)
    if kind == "pgvector":
        store = open_pgvector(s, name)
        store.reset()  # a rebuild replaces the index, like saving over a directory
    else:
        store = NumpyStore()
    store.add(chunks, vectors)
    store.save(target)  # a no-op for Postgres
    print(f"Saved {len(store)} chunks.")


def load_store(args, s: Settings):
    """The existing index and the embedder that made it."""
    size, overlap = args.chunk_size or s.chunk_size, args.chunk_overlap or s.chunk_overlap
    embedder = make_embedder(args.embedder, s)
    target = index_dir(s, embedder.name, size, overlap)
    hint = f"Run `python -m docrag index --embedder {args.embedder} --yes"
    if store_kind(args, s) == "pgvector":
        store = open_pgvector(s, index_name(embedder.name, size, overlap))
        if not len(store):
            raise SystemExit(f"No index in Postgres. {hint} --store pgvector`.")
        return store, embedder
    if not target.exists():
        raise SystemExit(f"No index at {target}. {hint}`.")
    return NumpyStore.load(target), embedder


def build_retriever(kind: str, store, embedder, s: Settings, rerank_llm: str | None = None) -> Retriever:
    dense = DenseRetriever(store, embedder)
    if kind == "dense":
        base: Retriever = dense
    else:
        keyword = Bm25Retriever(store.chunks())
        base = keyword if kind == "keyword" else HybridRetriever(dense, keyword, pool=s.hybrid_pool)
    if rerank_llm:
        return RerankingRetriever(base, LLMReranker(make_llm(rerank_llm, s)), pool=s.rerank_pool)
    return base


def load_pipeline(args, s: Settings, llm: LLM) -> tuple[RagPipeline, str]:
    """Builds the pipeline over an existing index. Returns it with the embedder's name."""
    store, embedder = load_store(args, s)
    pipeline = RagPipeline(
        build_retriever(
            args.retriever or s.retriever, store, embedder, s,
            rerank_llm=(args.rerank_llm if args.rerank else None),
        ),
        llm,
        top_k=args.top_k or s.top_k,
        min_score=s.min_score if args.min_score is None else args.min_score,
        max_attempts=s.max_attempts,
        device_names=load_overrides(s.pdf_dir),
        require_device=not getattr(args, "no_device_filter", False),
    )
    return pipeline, embedder.name


def format_trace(t) -> str:
    cost = "n/a (no price for model)" if t.cost_usd is None else f"~${t.cost_usd:.5f}"
    if not t.llm_calls:
        cost = "$0 (LLM not called)"
    return (
        f"Time {t.total_s:.2f}s (embed query {t.embed_s:.2f}s, search {t.search_s * 1000:.0f}ms, "
        f"LLM {t.llm_s:.2f}s over {len(t.llm_calls)} call(s)" + (f", of which rerank {t.rerank_s:.2f}s" if t.rerank_s else "") + ") | "
        f"Tokens {t.input_tokens:,} in / {t.output_tokens:,} out | Cost {cost}"
    )


def describe(h) -> str:
    """How a hit was found: cosine for dense, rank for keyword, both when fused."""
    parts = []
    if h.dense_score is not None:
        parts.append(f"cos {h.dense_score:.2f}")
    if h.keyword_rank is not None:
        parts.append(f"bm25 #{h.keyword_rank}")
    if h.rerank_score is not None:
        parts.append(f"rerank {h.rerank_score}/3")
    return ", ".join(parts) or f"{h.score:.2f}"


def cmd_ask(args, s: Settings) -> None:
    pipeline, _ = load_pipeline(args, s, make_llm(args.llm, s))

    def answer(question: str, device: str | None) -> None:
        result = pipeline.ask(question, device)
        print()
        if result.status == "needs_device":
            if device:
                print(f"Unknown device '{device}'.")
            print("Which device is this about?")
            return
        print(f"[device: {device}]")
        if result.status == "answered":
            for st in result.statements:
                c = st.citation
                where = f"{c.doc}, p.{c.page}" + (f", {c.section}" if c.section else "")
                print(f"- {st.text}\n    source: {where}\n    \"{c.quote}\"")
        elif result.status == "insufficient":
            print(f"The documentation does not contain enough information to answer this. ({result.reason})")
        else:
            print(f"Could not get a valid answer: {result.reason}")
        if result.rejected_statements:
            print(f"\n[{len(result.rejected_statements)} statement(s) dropped: quote not found in retrieved text]")
            for st in result.rejected_statements:
                print(f"  - {st.text}")
        tried = f", {result.attempts} attempt(s)" if result.attempts else ""
        print(f"\n(model: {result.model or 'not called'}{tried})")
        print(format_trace(result.trace))
        print("Retrieved: " + "; ".join(f"{h.chunk.doc} p.{h.chunk.page} ({describe(h)})" for h in result.hits))
        if args.verbose:
            for i, h in enumerate(result.hits, 1):
                c = h.chunk
                print(f"\n--- {i}. {c.doc} p.{c.page} score={h.score:.3f} | {c.section}\n{c.text}")

    device = args.device
    if args.question:
        answer(args.question, device)
        return
    while not pipeline.devices.resolve(device):  # nothing is processed until a device is named
        device = input("Which device is this about? (empty to quit): ").strip()
        if not device:
            return
    while (question := input(f"\n[{device}] Question (empty to quit): ").strip()):
        answer(question, device)


def _snippet(text: str, width: int = 150) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def cmd_search(args, s: Settings) -> None:
    """Dense, keyword and fused results side by side. No LLM, only the query embedding."""
    store, embedder = load_store(args, s)
    registry = DeviceRegistry(store.docs(), load_overrides(s.pdf_dir))
    docs = registry.resolve(args.device)
    if docs is None:
        print(f"Unknown device '{args.device}'." if args.device else "")
        print("Which device is this about?")
        return
    k = args.top_k or s.top_k
    pool = max(s.hybrid_pool, k)
    dense = DenseRetriever(store, embedder).retrieve(args.question, pool, docs)
    keyword = Bm25Retriever(store.chunks()).retrieve(args.question, pool, docs)
    fused = fuse(dense.hits, keyword.hits, k)

    def show(title: str, hits, detail) -> None:
        print(f"\n=== {title} (top {k}) ===")
        for i, h in enumerate(hits[:k], 1):
            c = h.chunk
            print(f"{i}. {c.doc} p.{c.page}  {detail(h)}")
            print(f"   [{c.section or 'no section'}]")
            print(f"   {_snippet(c.text)}")
        if not hits:
            print("(no results)")

    print(f"[device: {args.device}]  query: {args.question}")
    show("DENSE (semantic)", dense.hits, lambda h: f"cosine {h.dense_score:.3f}")
    show(
        "KEYWORD (BM25)", keyword.hits,
        lambda h: f"bm25 {h.keyword_score:.2f}  matched: {', '.join(matched_terms(args.question, h.chunk)) or '-'}",
    )
    rank = lambda r: f"#{r}" if r else "-"
    show("HYBRID (fused)", fused, lambda h: f"rrf {h.score:.4f}  dense {rank(h.dense_rank)}  keyword {rank(h.keyword_rank)}")
    print(f"\n(embed query {dense.embed_s:.2f}s, dense {dense.search_s * 1000:.0f}ms, keyword {keyword.search_s * 1000:.0f}ms)")


def cmd_eval(args, s: Settings) -> None:
    items = load_golden(args.golden)
    check_docs_exist(items, {p.name for p in s.pdf_dir.glob("*.pdf")})
    if args.limit:
        items = items[: args.limit]

    judge_name = judge = None
    if args.retrieval_only:
        llm = NoLLM()
    else:
        llm = make_llm(args.llm, s)
        judge_name = default_judge(args.llm) if args.judge == "auto" else args.judge
        if judge_name != "none":
            judge = make_llm(judge_name, s)
            if same_family(args.llm, judge_name):
                print(f"Warning: judge '{judge_name}' is the same model family as '{args.llm}'. "
                      "Judges tend to favour their own family's answers.")
        else:
            judge_name = None

    pipeline, embedder_name = load_pipeline(args, s, llm)
    if not args.no_device_filter:
        check_devices(items, pipeline.devices)
    k = args.top_k or s.top_k
    ks = sorted({1, 3, k} & set(range(1, k + 1)))
    config = {
        "golden": str(args.golden), "n_questions": len(items), "embedder": embedder_name,
        "chunk_size": args.chunk_size or s.chunk_size, "chunk_overlap": args.chunk_overlap or s.chunk_overlap,
        "store": store_kind(args, s), "retriever": args.retriever or s.retriever,
        "rerank": (args.rerank_llm if args.rerank else None),
        "device_filter": not args.no_device_filter, "top_k": k, "min_score": s.min_score if args.min_score is None else args.min_score,
        "llm": None if args.retrieval_only else args.llm, "judge": judge_name,
        "gemini_models": s.gemini_model_chain if "gemini" in (args.llm, judge_name) else None,
        "git_commit": git_commit(),
    }

    def progress(i, n, r):
        status = r.result.status if r.result else f"rank {r.retrieval_rank}"
        print(f"[{i}/{n}] {r.item.id}: {status}")

    results = run_eval(
        pipeline, judge, items, args.retrieval_only, progress, use_device=not args.no_device_filter
    )
    summary = summarize(results, ks, judge_used=judge is not None)

    kind = args.retriever or s.retriever
    label = "retrieval" if args.retrieval_only else f"{args.llm}_judge-{judge_name or 'none'}"
    label = f"{label}_{kind}" if kind != "dense" else label
    label = f"{label}_rerank" if args.rerank else label
    run_dir = new_run_dir(s.eval_dir, label)
    write_run(run_dir, results, summary, config)
    print("\n" + format_summary(summary, config))
    print(f"\nSaved to {run_dir}")


def use_utf8_output() -> None:
    """Manual text contains characters such as '‣' and '…'. Redirected output on Windows
    defaults to cp1252 and would crash on them, so write UTF-8 and never fail on a character."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    use_utf8_output()
    parser = argparse.ArgumentParser(prog="docrag")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--embedder", choices=EMBEDDER_PROVIDERS, default="openai")
        p.add_argument("--chunk-size", type=int)
        p.add_argument("--chunk-overlap", type=int)
        p.add_argument(
            "--store", choices=STORES,
            help="where vectors live: numpy (files, default) or pgvector (Postgres, needs DATABASE_URL)",
        )

    def retrieval(p):
        p.add_argument("--retriever", choices=RETRIEVERS, help="dense (default), keyword (BM25) or hybrid")
        p.add_argument(
            "--rerank", action="store_true",
            help="have an LLM re-order the top candidates before answering (costs one extra LLM call per question)",
        )
        p.add_argument("--rerank-llm", choices=LLM_PROVIDERS, default="openai", help="model that reranks")
        p.add_argument("--top-k", type=int)
        p.add_argument("--min-score", type=float, help="below this best-hit score, answer 'insufficient'")

    p_index = sub.add_parser("index", help="chunk the PDFs and build an index (dry run unless --yes)")
    common(p_index)
    p_index.add_argument("--yes", action="store_true", help="actually create embeddings")
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", help="ask questions against an existing index")
    common(p_ask)
    retrieval(p_ask)
    p_ask.add_argument("--llm", choices=LLM_PROVIDERS, default="openai")
    p_ask.add_argument("--device", help="the device (manual) the question is about; required")
    p_ask.add_argument("--verbose", action="store_true", help="also print the retrieved chunk text")
    p_ask.add_argument("question", nargs="?", help="omit for interactive mode")
    p_ask.set_defaults(func=cmd_ask)

    p_search = sub.add_parser("search", help="show dense, keyword and fused results side by side (no LLM)")
    common(p_search)
    p_search.add_argument("--top-k", type=int)
    p_search.add_argument("--device", help="the device (manual) to search; required")
    p_search.add_argument("question")
    p_search.set_defaults(func=cmd_search)

    p_eval = sub.add_parser("eval", help="score the system on a golden set")
    common(p_eval)
    retrieval(p_eval)
    p_eval.add_argument("golden", type=Path, help="golden set (JSONL), see eval/golden.jsonl")
    p_eval.add_argument("--llm", choices=LLM_PROVIDERS, default="openai")
    p_eval.add_argument(
        "--judge", choices=(*LLM_PROVIDERS, "auto", "none"), default="auto",
        help="auto = a different model family than --llm",
    )
    p_eval.add_argument("--retrieval-only", action="store_true", help="no LLM calls, retrieval metrics only")
    p_eval.add_argument("--limit", type=int, help="only the first N questions")
    p_eval.add_argument(
        "--no-device-filter", action="store_true",
        help="search all manuals, ignoring the golden set's device field (to measure what the filter adds)",
    )
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args, Settings())
