"""The golden set: questions with known answers and known source pages, written in advance."""

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, model_validator

from docrag.devices import DeviceRegistry


class RelevantPages(BaseModel):
    doc: str  # exact PDF file name
    pages: list[int] = Field(min_length=1)


class GoldenItem(BaseModel):
    id: str
    question: str
    device: str | None = None  # the device the user names; retrieval is limited to its manual
    answerable: bool = True
    reference_answer: str | None = None
    relevant: list[RelevantPages] = []  # where the answer is found
    tags: list[str] = []

    @model_validator(mode="after")
    def _consistent(self):
        if self.answerable and not (self.reference_answer and self.relevant):
            raise ValueError("answerable questions need reference_answer and relevant")
        if not self.answerable and self.relevant:
            raise ValueError("unanswerable questions must not list relevant pages")
        return self


def load_golden(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            items.append(GoldenItem.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"{path}:{number}: {e}") from e
    ids = [i.id for i in items]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate ids in {path}: {sorted(duplicates)}")
    return items


def check_devices(items: list[GoldenItem], registry: DeviceRegistry) -> None:
    """Every question must name a known device, since the system refuses to answer without one."""
    missing = [i.id for i in items if not i.device]
    if missing:
        raise ValueError(f"golden items without a device: {missing}")
    unknown = sorted({i.device for i in items if registry.resolve(i.device) is None})
    if unknown:
        raise ValueError(f"golden set names unknown devices {unknown}. Known: {registry.names}")


def check_docs_exist(items: list[GoldenItem], known_docs: set[str]) -> None:
    """A typo in a doc name would silently score 0 on retrieval, so fail loudly instead."""
    unknown = {r.doc for i in items for r in i.relevant} - known_docs
    if unknown:
        raise ValueError(f"golden set refers to documents not in the corpus: {sorted(unknown)}")
