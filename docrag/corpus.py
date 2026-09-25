"""Reading the local PDF folder and splitting pages into chunks."""

import re
from pathlib import Path

from pypdf import PdfReader

from docrag.models import Chunk, Page

# Coarse-to-fine split points: paragraph, line, sentence, word.
_SEPARATORS = [r"\n\s*\n", r"\n", r"(?<=[.!?])\s+", r"\s+"]


def outline_entries(reader: PdfReader) -> list[tuple[int, int, str]]:
    """The PDF outline as (page, depth, title), in document order. Broken bookmarks are skipped."""
    entries: list[tuple[int, int, str]] = []

    def walk(items: list, depth: int) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item, depth + 1)
                continue
            try:
                page = reader.get_destination_page_number(item)
            except Exception:  # broken bookmark
                continue
            if page is not None and page >= 0 and item.title:
                entries.append((page + 1, depth, re.sub(r"\s+", " ", item.title).strip()))

    walk(reader.outline, 0)
    entries.sort(key=lambda e: e[0])  # stable: keeps document order within a page
    return entries


def _heading_offset(title: str, text: str) -> int | None:
    """Where the heading stands on a line of its own in the page text, if it does."""
    words = r"\s+".join(re.escape(w) for w in title.split())
    match = re.search(rf"(?im)^[ \t]*{words}[ \t]*$", text)
    return match.start() if match else None


def page_spans(
    entries: list[tuple[int, int, str]], texts: list[str]
) -> dict[int, list[tuple[int, str]]]:
    """For each page (1-based): [(offset, "1. Intro > 1.2 Scope"), ...] sorted by offset.

    A section runs from its offset to the next one, so a page holding the end of one
    section and the start of the next gets two spans. A heading that cannot be located
    in the page text is applied at the top of its page. Pages before the first
    heading have no span.
    """
    spans: dict[int, list[tuple[int, str]]] = {}
    path: list[str] = []
    i = 0
    for page_no, text in enumerate(texts, 1):
        starting = []
        while i < len(entries) and entries[i][0] <= page_no:
            starting.append(entries[i])
            i += 1
        placed = sorted(
            ((_heading_offset(title, text) or 0, depth, title) for _, depth, title in starting),
            key=lambda e: e[0],
        )
        result: list[tuple[int, str]] = [(0, " > ".join(path))] if path else []
        for offset, depth, title in placed:
            path = path[:depth] + [title]
            if result and result[-1][0] == offset:
                result[-1] = (offset, " > ".join(path))
            else:
                result.append((offset, " > ".join(path)))
        if result:
            spans[page_no] = result
    return spans


def _edge_key(line: str) -> str:
    """A line with digits and rule characters normalised, so page numbers do not hide repeats."""
    return re.sub(r"[\d_\s]+", " ", line).strip()


def strip_running_lines(
    texts: list[str], edge: int = 2, min_share: float = 0.25, min_pages: int = 4
) -> list[str]:
    """Remove running headers, footers and page numbers from the top and bottom of each page.

    A line counts if, digits ignored, it sits in the first or last `edge` lines of at
    least `min_share` of the pages (and `min_pages` of them). Nothing is specific to one
    manual, and lines in the middle of a page are never touched.
    """
    per_page = [[line for line in text.splitlines() if line.strip()] for text in texts]
    seen: dict[str, int] = {}
    for lines in per_page:
        for key in {_edge_key(line) for line in lines[:edge] + lines[-edge:]}:
            seen[key] = seen.get(key, 0) + 1
    needed = max(min_pages, min_share * len(texts))
    repeated = {key for key, count in seen.items() if count >= needed}
    if not repeated:
        return texts

    def clean(text: str) -> str:
        lines = text.splitlines()
        keep = [i for i, line in enumerate(lines) if line.strip()]
        drop = {i for i in keep[:edge] + keep[-edge:] if _edge_key(lines[i]) in repeated}
        return "\n".join(line for i, line in enumerate(lines) if i not in drop).strip()

    return [clean(text) for text in texts]


def load_pdf(path: Path) -> list[Page]:
    reader = PdfReader(path)
    texts = [(page.extract_text() or "").strip() for page in reader.pages]
    texts = strip_running_lines(texts)
    spans = page_spans(outline_entries(reader), texts)
    return [
        Page(doc=path.name, page=number, text=text, spans=spans.get(number, []))
        for number, text in enumerate(texts, 1)
        if text
    ]


def load_corpus(pdf_dir: Path) -> list[Page]:
    paths = sorted(pdf_dir.glob("*.pdf"))
    if not paths:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")
    return [page for path in paths for page in load_pdf(path)]


def _atomize(text: str, limit: int, separators: list[str] = _SEPARATORS) -> list[str]:
    """Break text into pieces no longer than limit, splitting at the coarsest boundary."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    if not separators:
        return [text[i : i + limit] for i in range(0, len(text), limit)]
    parts = re.split(separators[0], text)
    if len(parts) == 1:
        return _atomize(text, limit, separators[1:])
    return [piece for part in parts for piece in _atomize(part, limit, separators[1:])]


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Greedy merge of atoms into chunks of at most `size` chars, with word-aligned overlap."""
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    chunks: list[str] = []
    current = ""
    for atom in _atomize(text, size - overlap):
        if current and len(current) + 1 + len(atom) > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = tail[tail.find(" ") + 1 :] if " " in tail else tail
            if len(current) + 1 + len(atom) > size:  # overlap doesn't fit next to this atom
                current = ""
        current = f"{current} {atom}".strip()
    if current:
        chunks.append(current)
    return chunks


def _merge_short(pieces: list[tuple[str, str | None]], min_chars: int) -> list[tuple[str, str | None]]:
    """Fold each piece shorter than min_chars into the piece after it (the last one into the one before).

    A heading that stands alone on a short chunk would otherwise be embedded as little more
    than its section label and match short queries better than the content it introduces.
    The merged piece keeps the later piece's (more specific) section.
    """
    merged: list[tuple[str, str | None]] = []
    carry = ""
    for text, section in pieces:
        text = f"{carry}\n{text}" if carry else text
        carry = ""
        if len(text) < min_chars:
            carry = text
            continue
        merged.append((text, section))
    if carry:
        if merged:
            merged[-1] = (f"{merged[-1][0]}\n{carry}", merged[-1][1])
        else:
            merged.append((carry, pieces[-1][1]))
    return merged


def chunk_pages(pages: list[Page], size: int, overlap: int, min_chars: int = 0) -> list[Chunk]:
    """Chunk each page separately so every chunk keeps an exact page reference.

    A page is first cut at its section headings, so no chunk straddles two sections
    and each carries the section it belongs to. Chunks shorter than `min_chars` are
    merged into their neighbour on the same page (0 keeps every piece).
    """
    chunks = []
    for page in pages:
        bounds: list[tuple[int, str | None]] = list(page.spans)
        if not bounds or bounds[0][0] > 0:
            bounds.insert(0, (0, None))  # text before the first heading has no section
        ends = [start for start, _ in bounds[1:]] + [len(page.text)]
        pieces = [
            (text, section)
            for (start, section), end in zip(bounds, ends)
            for text in split_text(page.text[start:end], size, overlap)
        ]
        if min_chars:
            pieces = _merge_short(pieces, min_chars)
        for n, (text, section) in enumerate(pieces):
            chunks.append(
                Chunk(
                    id=f"{page.doc}:p{page.page}:c{n}",
                    doc=page.doc,
                    page=page.page,
                    text=text,
                    section=section,
                )
            )
    return chunks
