import pytest

from docrag.corpus import chunk_pages, split_text, strip_running_lines
from docrag.models import Page


def test_short_text_is_one_chunk():
    assert split_text("hello world", 100, 10) == ["hello world"]


def test_chunks_respect_size_and_cover_all_words():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = split_text(text, 200, 40)
    assert all(len(c) <= 200 for c in chunks)
    assert len(chunks) > 1
    assert {f"word{i}" for i in range(300)} <= set(" ".join(chunks).split())


def test_overlap_repeats_text_between_neighbours():
    text = " ".join(f"w{i}" for i in range(200))
    a, b = split_text(text, 100, 30)[:2]
    assert set(a.split()[-3:]) & set(b.split()[:6])


def test_prefers_paragraph_boundaries():
    text = ("a " * 20).strip() + "\n\n" + ("b " * 20).strip()
    chunks = split_text(text, 60, 0)
    assert chunks[0].startswith("a") and "b" not in chunks[0]


def test_unbroken_string_is_hard_cut():
    assert all(len(c) <= 50 for c in split_text("x" * 500, 50, 10))


def test_bad_overlap_rejected():
    with pytest.raises(ValueError):
        split_text("abc", 10, 10)


def test_chunks_keep_page_reference():
    pages = [Page(doc="a.pdf", page=3, text="word " * 100)]
    chunks = chunk_pages(pages, 100, 20)
    assert {c.page for c in chunks} == {3} and {c.doc for c in chunks} == {"a.pdf"}
    assert len({c.id for c in chunks}) == len(chunks)


_WORDS = "zero one two three four five six seven eight nine ten".split()


def _pages(n: int) -> list[str]:
    return [
        f"User's Guide ______ {i}\n{_WORDS[i]} heading\nBody of page {_WORDS[i]}.\nclosing {_WORDS[i]}\n{i}"
        for i in range(1, n + 1)
    ]


def test_running_header_and_page_number_are_stripped():
    cleaned = strip_running_lines(_pages(10))
    assert cleaned[2] == "three heading\nBody of page three.\nclosing three"


def test_repeated_line_in_the_middle_of_a_page_is_kept():
    texts = [f"Title {i}\nline a\nNote: check power\nline b\nline c\nend {i}" for i in range(10)]
    assert all("Note: check power" in t for t in strip_running_lines(texts))


def test_rare_edge_line_is_kept():
    texts = _pages(10)
    texts[0] = "Chapter 1 Intro\n" + texts[0]
    assert strip_running_lines(texts)[0].startswith("Chapter 1 Intro")


def test_short_documents_are_left_alone():
    texts = ["Header\nbody one\nx", "Header\nbody two\ny"]
    assert strip_running_lines(texts) == texts


def _sections_page(text: str, spans: list[tuple[int, str]]) -> Page:
    return Page(doc="a.pdf", page=1, text=text, spans=spans)


def test_heading_only_piece_is_merged_into_the_next_one():
    text = "Display\nIndicators on the Display\n" + "word " * 60
    spans = [(0, "Ops > Display"), (len("Display\n"), "Ops > Display > Indicators")]
    plain = chunk_pages([_sections_page(text, spans)], 400, 20)
    merged = chunk_pages([_sections_page(text, spans)], 400, 20, min_chars=50)
    assert [c.text for c in plain][0] == "Display"
    assert len(merged) == len(plain) - 1
    assert merged[0].text.startswith("Display\nIndicators on the Display")
    assert merged[0].section == "Ops > Display > Indicators"  # the later, more specific section


def test_short_last_piece_is_merged_into_the_previous_one():
    text = "word " * 60 + "\nEnd"
    spans = [(0, "A"), (len("word " * 60) + 1, "B")]
    chunks = chunk_pages([_sections_page(text, spans)], 400, 20, min_chars=50)
    assert chunks[-1].text.endswith("End") and chunks[-1].section == "A"


def test_merge_ids_are_sequential_and_min_chars_zero_changes_nothing():
    text = "Display\n" + "word " * 60
    page = _sections_page(text, [(0, "A"), (8, "B")])
    assert [c.id for c in chunk_pages([page], 400, 20, min_chars=50)] == ["a.pdf:p1:c0"]
    assert len(chunk_pages([page], 400, 20)) == 2
