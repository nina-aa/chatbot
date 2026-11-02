import pytest

from docrag.models import Chunk, Hit
from docrag.rag.schema import Statement
from docrag.rag.structured import extract_json
from docrag.rag.verify import quote_in_text, verify_statements


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}


def test_extract_json_rejects_text_without_json():
    with pytest.raises(ValueError):
        extract_json("no braces here")


TEXT = "The  compressor must be\nheated to the set temperature before starting the screw."


def test_quote_matching_ignores_case_and_whitespace():
    assert quote_in_text("compressor must be heated to the set temperature", TEXT)
    assert quote_in_text("THE COMPRESSOR MUST BE HEATED", TEXT)


def test_quote_matching_rejects_invented_or_paraphrased_text():
    assert not quote_in_text("the compressor needs warming up first", TEXT)


def test_too_short_quote_proves_nothing():
    assert not quote_in_text("the", TEXT)


def test_ellipsis_quote_needs_every_segment_present():
    assert quote_in_text("compressor must be ... before starting the screw", TEXT)
    assert not quote_in_text("compressor must be ... and then cooled down slowly", TEXT)


def test_verify_splits_verified_and_rejected():
    hits = [Hit(chunk=Chunk(id="0", doc="a.pdf", page=2, text=TEXT, section="S"), score=0.9)]
    q = "heated to the set temperature"
    good = Statement(text="Heat it first.", doc="a.pdf", page=2, quote=q)
    bad_page = Statement(text="x", doc="a.pdf", page=3, quote=q)
    bad_doc = Statement(text="y", doc="b.pdf", page=2, quote=q)
    verified, rejected = verify_statements([good, bad_page, bad_doc], hits)
    assert [(v.text, v.citation.section) for v in verified] == [("Heat it first.", "S")]
    assert rejected == [bad_page, bad_doc]
