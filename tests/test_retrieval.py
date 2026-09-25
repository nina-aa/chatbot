import math

from docrag.models import Chunk, Hit
from docrag.rag.pipeline import RagPipeline
from docrag.retrieval.base import Retrieval, Retriever
from docrag.retrieval.bm25 import Bm25Index, Bm25Retriever, matched_terms, tokenize
from docrag.retrieval.dense import DenseRetriever
from docrag.retrieval.hybrid import HybridRetriever, fuse
from docrag.vectorstore.numpy_store import NumpyStore
from tests.conftest import FakeEmbedder
from tests.test_pipeline import ScriptedLLM, answer


def chunk(id_, text, doc="d.pdf", page=1, section=None):
    return Chunk(id=id_, doc=doc, page=page, text=text, section=section)


# ---- tokenizer -----------------------------------------------------------------------


def test_tokenize_keeps_codes_and_numbers_whole():
    assert tokenize("HTM10 03: Restart!") == ["htm10", "03", "restart"]
    assert tokenize("registers 0706hex and 0806hex") == ["registers", "0706hex", "and", "0806hex"]
    assert tokenize("") == []


# ---- BM25 ----------------------------------------------------------------------------


def test_rare_term_beats_common_term():
    chunks = [chunk("a", "the probe the probe the probe"), chunk("b", "the probe uses htm10")] + [
        chunk(f"c{i}", "the probe is a probe") for i in range(5)
    ]
    ranked = Bm25Index(chunks).search("htm10 probe", 3)
    assert chunks[ranked[0][0]].id == "b"  # htm10 appears nowhere else


def test_bare_code_finds_the_exact_row():
    chunks = [
        chunk("prose", "The transmitter reports internal problems that you can clear by restarting."),
        chunk("table", "Internal 1 1 Internal problem with the transmitter. Restart the transmitter."),
        chunk("other", "Configuration registers and their addresses."),
    ]
    top = Bm25Index(chunks).search("Internal 1", 3)
    assert chunks[top[0][0]].id == "table"


def test_term_frequency_saturates_and_length_is_normalised():
    short = chunk("short", "error error")
    padded = chunk("padded", "error " + "filler " * 60)
    spam = chunk("spam", "error " * 40)
    index = Bm25Index([short, padded, spam])
    scores = {index.chunk(i).id: score for i, score in index.search("error", 3)}
    assert scores["short"] > scores["padded"]  # same word, far less padding
    assert scores["spam"] < 2 * scores["short"]  # 20x the repeats is nowhere near 20x the score


def test_score_matches_the_bm25_formula():
    chunks = [chunk("a", "alpha beta"), chunk("b", "beta gamma delta")]
    index = Bm25Index(chunks, k1=1.5, b=0.75)
    (i, score), = index.search("alpha", 5)
    # by hand: the header word "d" is a stopword, so lengths are 2 and 3 and avg = 2.5;
    # alpha is in 1 of 2 chunks
    idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    expected = idf * 1 * 2.5 / (1 + 1.5 * (1 - 0.75 + 0.75 * 2 / 2.5))
    assert chunks[i].id == "a" and math.isclose(score, expected)


def test_no_matching_term_gives_no_hits_and_ties_keep_document_order():
    chunks = [chunk("a", "restart it"), chunk("b", "restart it")]
    index = Bm25Index(chunks)
    assert index.search("nothing here", 5) == [] and index.search("", 5) == []
    assert [i for i, _ in index.search("restart", 5)] == [0, 1]


def test_retriever_limits_to_documents_and_weighs_terms_within_them():
    chunks = [chunk("a1", "error alpha", doc="A.pdf"), chunk("a2", "error beta", doc="A.pdf"),
              chunk("b1", "error gamma", doc="B.pdf"), chunk("b2", "other words", doc="B.pdf")]
    r = Bm25Retriever(chunks)
    only_b = r.retrieve("error", 5, docs={"B.pdf"})
    assert [h.chunk.id for h in only_b.hits] == ["b1"]
    everywhere = r.retrieve("error", 5)
    assert {h.chunk.id for h in everywhere.hits} == {"a1", "a2", "b1"}
    # "error" is rarer inside B (1 of 2 chunks) than overall (3 of 4), so it scores higher there
    assert only_b.hits[0].score > next(h.score for h in everywhere.hits if h.chunk.id == "b1")
    assert r.docs() == {"A.pdf", "B.pdf"} and only_b.hits[0].keyword_rank == 1


def test_matched_terms_lists_query_words_found_in_the_chunk():
    c = chunk("a", "Restart the transmitter", section="Errors")
    assert matched_terms("please restart the HTM10 transmitter restart", c) == ["restart", "transmitter"]  # "the" is filler; "please", "htm10" are not in the chunk


# ---- fusion --------------------------------------------------------------------------


def dense_hit(id_, rank, score=0.5):
    return Hit(chunk=chunk(id_, id_), score=score, dense_rank=rank, dense_score=score)


def kw_hit(id_, rank, score=5.0):
    return Hit(chunk=chunk(id_, id_), score=score, keyword_rank=rank, keyword_score=score)


def test_chunk_in_both_lists_beats_chunks_in_one():
    fused = fuse([dense_hit("x", 1), dense_hit("both", 2)], [kw_hit("y", 1), kw_hit("both", 2)], k=3)
    assert [h.chunk.id for h in fused][0] == "both"


def test_fused_hits_remember_where_they_stood_in_each_list():
    fused = {h.chunk.id: h for h in fuse([dense_hit("a", 1, 0.9), dense_hit("b", 2)], [kw_hit("b", 1, 7.0)], k=5)}
    assert (fused["a"].dense_rank, fused["a"].keyword_rank, fused["a"].keyword_score) == (1, None, None)
    assert (fused["b"].dense_rank, fused["b"].keyword_rank, fused["b"].keyword_score) == (2, 1, 7.0)
    assert math.isclose(fused["b"].score, 1 / 62 + 1 / 61)  # RRF with k=60


def test_fusion_ignores_score_scale_and_respects_k():
    fused = fuse([dense_hit("a", 1, 0.9)], [kw_hit("b", 1, 9999.0)], k=1)
    assert [h.chunk.id for h in fused] == ["a"]  # equal ranks tie; the huge BM25 score buys nothing
    assert fuse([], [], k=3) == []


def test_agreement_between_lists_can_outweigh_a_single_top_rank():
    # RRF rewards a chunk both rankings like: rank 30 in both (2/90) beats rank 1 in only one (1/61).
    dense = [dense_hit("top", 1)] + [dense_hit(f"d{i}", i) for i in range(2, 30)] + [dense_hit("low", 30)]
    keyword = [kw_hit(f"k{i}", i) for i in range(1, 30)] + [kw_hit("low", 30)]
    ids = [h.chunk.id for h in fuse(dense, keyword, k=3)]
    assert ids[0] == "low"
    assert set(ids[1:]) == {"top", "k1"}  # then the two rank-1 chunks, tied on 1/61


# ---- hybrid retriever ----------------------------------------------------------------


class StubDense(Retriever):
    def __init__(self, hits):
        self._hits = hits

    def retrieve(self, question, k, docs=None):
        return Retrieval(self._hits[:k], embed_s=0.25, search_s=0.5)

    def docs(self):
        return {"d.pdf"}


def test_hybrid_rescues_an_exact_match_the_dense_search_ranked_low():
    topics = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
    chunks = [chunk("row", "Internal 1 1 Internal problem with the transmitter")] + [
        chunk(f"n{i}", f"unrelated text about {t}") for i, t in enumerate(topics)
    ]
    dense_order = [dense_hit(f"n{i}", i + 1) for i in range(10)] + [dense_hit("row", 11)]
    for h in dense_order:  # give the stub real chunks
        h.chunk = next(c for c in chunks if c.id == h.chunk.id)
    hybrid = HybridRetriever(StubDense(dense_order), Bm25Retriever(chunks), pool=20)
    assert "row" not in [h.chunk.id for h in StubDense(dense_order).retrieve("Internal 1", 4).hits]
    result = hybrid.retrieve("Internal 1", 4)
    assert result.hits[0].chunk.id == "row"  # both rankings contribute; it now leads
    assert (result.hits[0].dense_rank, result.hits[0].keyword_rank) == (11, 1)
    assert result.embed_s == 0.25 and result.search_s >= 0.5  # timings are carried through


# ---- pipeline with other retrievers --------------------------------------------------

TEXTS = [chunk("0", "The screw speed is set on the operator panel under the Extruder menu.", page=7),
         chunk("1", "Pin 1 carries the positive supply and pin 2 the ground.", page=9)]
GOOD = ("d.pdf", 7, "screw speed is set on the operator panel")


def test_pipeline_answers_from_a_keyword_only_retriever():
    llm = ScriptedLLM(answer(citations=(GOOD,)))
    pipe = RagPipeline(Bm25Retriever(TEXTS), llm, top_k=2, require_device=False)
    result = pipe.ask("screw speed operator panel")
    assert result.status == "answered" and result.hits[0].chunk.id == "0"
    assert result.hits[0].keyword_rank == 1 and result.trace.embed_s == 0


def test_min_score_gate_uses_cosine_so_it_skips_keyword_only_retrieval():
    llm = ScriptedLLM(answer(citations=(GOOD,)))
    result = RagPipeline(Bm25Retriever(TEXTS), llm, top_k=2, min_score=0.99, require_device=False).ask("screw speed")
    assert result.status == "answered"  # BM25 scores are not cosines, so the gate does not apply


def test_min_score_gate_still_applies_when_a_dense_score_is_present():
    emb = FakeEmbedder()
    store = NumpyStore()
    store.add(TEXTS, emb.embed_documents([c.text for c in TEXTS]))
    hybrid = HybridRetriever(DenseRetriever(store, emb), Bm25Retriever(TEXTS))
    llm = ScriptedLLM()  # would fail if called
    result = RagPipeline(hybrid, llm, top_k=2, min_score=0.99, require_device=False).ask("zzz qqq")
    assert result.status == "insufficient" and result.reason == "low_retrieval_score"


# ---- stopwords -----------------------------------------------------------------------


def test_filler_words_neither_match_nor_count():
    from docrag.retrieval.bm25 import content_terms

    assert content_terms("My probe dont work at all, it can't start") == ["probe", "dont", "work", "start"]
    assert content_terms("HTM10 03 error 0706hex") == ["htm10", "03", "error", "0706hex"]  # codes survive
    chunks = [chunk("a", "the of and to in it is"), chunk("b", "restart the transmitter")]
    index = Bm25Index(chunks)
    assert index.search("the of and to", 5) == []  # a question of pure filler matches nothing


def test_chatty_question_is_ranked_by_its_content_words():
    chunks = [chunk("noise", "my sensor is at all times not in the way i think it can be"),
              chunk("answer", "Probe start-up: check the power input and wiring."),
              chunk("other", "Configuration registers and addresses.")]
    top = Bm25Index(chunks).search("my probe dont work at all, i cant start it", 3)
    assert chunks[top[0][0]].id == "answer"
