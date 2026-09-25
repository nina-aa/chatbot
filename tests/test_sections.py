import io

from pypdf import PdfReader, PdfWriter

from docrag.corpus import _heading_offset, chunk_pages, outline_entries, page_spans
from docrag.models import Chunk, Page


def _reader() -> PdfReader:
    w = PdfWriter()
    for _ in range(6):
        w.add_blank_page(200, 200)
    intro = w.add_outline_item("1. Intro", 0)  # unicode space as in the real manuals
    w.add_outline_item("1.1 Scope", 1, parent=intro)
    w.add_outline_item("1.2 Safety", 2, parent=intro)
    w.add_outline_item("2 Operation", 4)
    buf = io.BytesIO()
    w.write(buf)
    buf.seek(0)
    return PdfReader(buf)


def test_outline_entries_have_page_depth_and_clean_title():
    assert outline_entries(_reader()) == [
        (1, 0, "1. Intro"),
        (2, 1, "1.1 Scope"),
        (3, 1, "1.2 Safety"),
        (5, 0, "2 Operation"),
    ]


def test_pdf_without_outline_gives_no_entries():
    w = PdfWriter()
    w.add_blank_page(100, 100)
    buf = io.BytesIO()
    w.write(buf)
    buf.seek(0)
    assert outline_entries(PdfReader(buf)) == []


ENTRIES = [(1, 0, "1 Intro"), (1, 1, "1.1 Scope"), (2, 0, "2 Table"), (3, 0, "3 Next")]


def test_pages_carry_the_path_until_the_next_heading():
    texts = ["1 Intro\nx\n1.1 Scope\nbody", "table row continues", "3 Next\nmore"]
    spans = page_spans([(1, 0, "1 Intro"), (1, 1, "1.1 Scope"), (3, 0, "3 Next")], texts)
    assert [s for _, s in spans[1]] == ["1 Intro", "1 Intro > 1.1 Scope"]
    assert spans[2] == [(0, "1 Intro > 1.1 Scope")]  # nothing new starts: path carries over
    assert spans[3] == [(0, "3 Next")]  # new top-level heading resets the path


def test_page_ending_one_section_and_starting_another_gets_two_spans():
    # The real case: a table row's tail, then a new heading further down the same page.
    text = "Problem: x\nCheck the scaling.\n10.2 Analog output error state\nThe probe sets the output."
    spans = page_spans([(1, 0, "10.1 Problems"), (2, 1, "10.2 Analog output error state")], ["a", text])
    assert spans[2][0] == (0, "10.1 Problems")
    offset, section = spans[2][1]
    assert section == "10.1 Problems > 10.2 Analog output error state"
    assert text[offset:].lstrip().startswith("10.2 Analog output error state")


def test_heading_not_found_in_text_applies_at_top_of_page():
    spans = page_spans([(1, 0, "Overview")], ["text without the heading line"])
    assert spans[1] == [(0, "Overview")]


def test_heading_must_stand_on_its_own_line():
    assert _heading_offset("Analog output error state", "see Analog output error state for details") is None
    assert _heading_offset("Analog output error state", "x\nAnalog  output error state\ny") == 2
    assert _heading_offset("10.2 Analog output", "a\n10.2 Analog output\nb") == 2


def test_pages_before_the_first_heading_have_no_span():
    assert page_spans([(2, 0, "A")], ["cover", "A\nbody"]) == {2: [(0, "A")]}


def test_chunks_never_straddle_a_heading_and_carry_their_own_section():
    text = "Tail of the table row. " * 6 + "\n10.2 Analog output error state\n" + "Purge lasts four minutes. " * 6
    offset = text.index("10.2")
    page = Page(doc="a.pdf", page=5, text=text, spans=[(0, "10.1 Problems"), (offset, "10.2 Analog")])
    chunks = chunk_pages([page], 120, 20)
    before = [c for c in chunks if c.section == "10.1 Problems"]
    after = [c for c in chunks if c.section == "10.2 Analog"]
    assert before and after and len(before) + len(after) == len(chunks)
    assert all("Purge" not in c.text for c in before)
    assert all("Tail of the table" not in c.text for c in after)
    assert [c.id for c in chunks] == [f"a.pdf:p5:c{i}" for i in range(len(chunks))]


def test_chunks_without_any_outline_have_no_section():
    chunks = chunk_pages([Page(doc="a.pdf", page=1, text="word " * 50)], 100, 10)
    assert chunks and all(c.section is None for c in chunks)


def test_text_before_the_first_heading_on_a_page_has_no_section():
    text = "intro text before " * 3 + "\nA Heading\nafter " * 3
    page = Page(doc="a.pdf", page=1, text=text, spans=[(text.index("A Heading"), "A Heading")])
    chunks = chunk_pages([page], 80, 0)
    assert chunks[0].section is None and chunks[-1].section == "A Heading"


def test_embedding_text_adds_context_but_chunk_text_stays_verbatim():
    c = Chunk(id="0", doc="HMW90 user guide.pdf", page=1, text="body", section="1 Intro")
    assert c.embedding_text() == "HMW90 user guide | 1 Intro\nbody"
    assert c.text == "body"
    assert Chunk(id="1", doc="a.pdf", page=1, text="body").embedding_text() == "a\nbody"
