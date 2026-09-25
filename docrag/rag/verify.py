"""Check the LLM's citations against what was actually retrieved."""

import re

from docrag.models import Hit
from docrag.rag.schema import AnswerStatement, Citation, Statement

MIN_QUOTE_CHARS = 10  # shorter quotes match almost anywhere and prove nothing
_ELLIPSIS = re.compile(r"\.{3}|…")


def _norm(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_in_text(quote: str, text: str) -> bool:
    """True if the quote appears verbatim (ignoring case and whitespace) in text.

    "a ... b" is accepted when each segment appears; at least one segment must be long enough.
    """
    haystack = _norm(text)
    segments = [s for s in (_norm(p).strip(" .") for p in _ELLIPSIS.split(quote)) if s]
    if not any(len(s) >= MIN_QUOTE_CHARS for s in segments):
        return False
    return all(s in haystack for s in segments if len(s) >= MIN_QUOTE_CHARS)


def verify_statements(
    statements: list[Statement], hits: list[Hit]
) -> tuple[list[AnswerStatement], list[Statement]]:
    """Split statements into (verified, rejected).

    Verified means: a retrieved chunk from the same document and page contains the quote.
    This proves the quote is real, not that it supports the statement. The judge in the
    eval measures that.
    """
    verified, rejected = [], []
    for st in statements:
        source = next(
            (
                h.chunk
                for h in hits
                if h.chunk.doc == st.doc
                and h.chunk.page == st.page
                and quote_in_text(st.quote, h.chunk.text)
            ),
            None,
        )
        if source is None:
            rejected.append(st)
        else:
            citation = Citation(doc=source.doc, page=source.page, section=source.section, quote=st.quote)
            verified.append(AnswerStatement(text=st.text.strip(), citation=citation))
    return verified, rejected
