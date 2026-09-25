import time

from docrag.devices import DeviceRegistry
from docrag.llm.base import LLM, AllModelsFailed, LLMCall
from docrag.models import Hit
from docrag.rag.schema import LLMAnswer, RagResult
from docrag.rag.structured import StructuredOutputError, generate_validated
from docrag.rag.trace import Trace
from docrag.rag.verify import verify_statements
from docrag.retrieval.base import Retrieval, Retriever

SYSTEM_PROMPT = """You answer questions about industrial equipment documentation.

Rules:
- Use ONLY the context provided. Never use outside knowledge.
- If the context does not contain the answer, set "sufficient" to false. Do not guess, and do not answer partially from general knowledge.
- Answer only what was asked. Include a statement only if it directly answers the question. Do not add background, definitions or facts from neighbouring topics in the context, even if they are true and related.
- Write the answer as short statements. Every statement carries its own citation.
- A statement's "text" may only say what its "quote" says. Shorten or rephrase it if you like, but never add to it, and never put one fact in "text" and a different fact in "quote".
- A cause and its remedy belong in ONE statement (for example "Possible cause: X. Remedy: Y."), with a quote that covers both. Steps of a procedure are separate statements.
- "quote" is copied word for word from the context, one to three sentences. Do not quote a heading, a table title, a "Problem:" line or the user's question, unless the fact itself is stated there.
- "doc" and "page" are those of the context block the quote comes from.
- Reply with a single JSON object and nothing else.

Format:
{"sufficient": true, "statements": [{"text": "one fact or step", "doc": "file.pdf", "page": 12, "quote": "exact words from the context"}]}

Example of a cause and remedy kept together, when the context reads "... The filter is blocked. Replace the filter (see page 9). ...":
{"text": "Possible cause: the filter is blocked. Remedy: replace the filter.", "doc": "file.pdf", "page": 12, "quote": "The filter is blocked. Replace the filter (see page 9)."}

When the context does not answer the question:
{"sufficient": false, "statements": []}"""


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for h in hits:
        c = h.chunk
        header = f"[doc: {c.doc} | page: {c.page}" + (f" | section: {c.section}" if c.section else "") + "]"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n".join(blocks)


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        top_k: int,
        min_score: float = 0.0,
        max_attempts: int = 3,
        device_names: dict[str, list[str]] | None = None,
        require_device: bool = True,
    ):
        self._retriever, self._llm = retriever, llm
        self._top_k, self._min_score, self._max_attempts = top_k, min_score, max_attempts
        self.devices = DeviceRegistry(retriever.docs(), device_names)
        self._require_device = require_device

    def retrieve(self, question: str, k: int | None = None, device: str | None = None) -> list[Hit]:
        return self.retrieve_full(question, k, device).hits

    def retrieve_full(self, question: str, k: int | None = None, device: str | None = None) -> Retrieval:
        """Like retrieve(), but with timings and the LLM calls a reranker made."""
        return self._retriever.retrieve(question, k or self._top_k, self.devices.resolve(device))

    def ask(self, question: str, device: str | None = None) -> RagResult:
        """Answer a question about one device. Without a known device nothing is processed:
        no embedding, no LLM call, just the list of devices to choose from."""
        docs = self.devices.resolve(device)
        if self._require_device and docs is None:
            reason = "device_required" if not device else f"unknown device '{device}'"
            return RagResult(status="needs_device", reason=reason, devices=self.devices.names)

        started = time.perf_counter()
        retrieval = self._retriever.retrieve(question, self._top_k, docs)
        hits = retrieval.hits
        calls: list[LLMCall] = list(retrieval.llm_calls)  # reranker calls count toward the trace

        def done(**fields) -> RagResult:
            trace = Trace(
                embed_s=retrieval.embed_s, search_s=retrieval.search_s, llm_calls=calls,
                total_s=time.perf_counter() - started,
            )
            return RagResult(hits=hits, trace=trace, **fields)

        # Gate 1: nothing relevant retrieved. Saves an LLM call and cannot hallucinate.
        # The threshold is a cosine similarity, so it applies whenever the retriever gave one.
        cosines = [h.dense_score for h in hits if h.dense_score is not None]
        if not hits or (cosines and max(cosines) < self._min_score):
            return done(status="insufficient", reason="low_retrieval_score")

        prompt = f"Context:\n{format_context(hits)}\n\nQuestion: {question}"
        try:
            parsed, model, attempts = generate_validated(
                self._llm, LLMAnswer, prompt, SYSTEM_PROMPT, self._max_attempts, calls
            )
        except AllModelsFailed as e:
            return done(status="error", reason=f"all models failed: {e}")
        except StructuredOutputError as e:
            return done(
                status="error", reason=f"invalid model output: {e.last_error}", attempts=e.attempts
            )

        # Gate 2: the model itself says the context doesn't answer the question.
        if not parsed.sufficient:
            return done(
                status="insufficient", reason="model_reported_insufficient",
                model=model, attempts=attempts,
            )

        # Gate 3: each statement is only kept if its own quote is really in the context.
        # Unsupported statements are dropped, so the answer contains only backed claims.
        verified, rejected = verify_statements(parsed.statements, hits)
        if not verified:
            return done(
                status="insufficient", reason="no_verifiable_citations",
                rejected_statements=rejected, model=model, attempts=attempts,
            )
        citations = list({(s.citation.doc, s.citation.page, s.citation.quote): s.citation for s in verified}.values())
        return done(
            status="answered", answer=" ".join(s.text for s in verified),
            statements=verified, citations=citations,
            rejected_statements=rejected, model=model, attempts=attempts,
        )
