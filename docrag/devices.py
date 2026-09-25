"""Which device(s) a manual covers. The user names a device; retrieval is limited to its manual."""

import json
import re
from pathlib import Path

OVERRIDES_FILE = "devices.json"  # optional, in the PDF folder: {"file.pdf": ["NAME", ...]}


def derive_names(doc: str) -> list[str]:
    """Device names from a file name: 'HMW90 user guide.pdf' -> ['HMW90'].

    A manual for several devices ('DKP202-DKP203-...-User-Guide-M2103.pdf') yields one name each.
    """
    stem = re.sub(r"\s*\(\d+\)$", "", Path(doc).stem)  # "... (1)" left by repeated downloads
    head = re.split(r"[\s_-]*user[\s_-]*guide", stem, flags=re.I)[0]
    return [n for n in re.split(r"[-\s,]+", head) if n] or [stem]


def load_overrides(pdf_dir: Path) -> dict[str, list[str]]:
    path = pdf_dir / OVERRIDES_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class DeviceRegistry:
    """Device name -> the documents that cover it. Names match case-insensitively."""

    def __init__(self, docs: set[str], overrides: dict[str, list[str]] | None = None):
        overrides = overrides or {}
        self._docs: dict[str, set[str]] = {}
        for doc in sorted(docs):
            for name in overrides.get(doc) or derive_names(doc):
                self._docs.setdefault(name, set()).add(doc)

    @property
    def names(self) -> list[str]:
        return sorted(self._docs)

    def resolve(self, name: str | None) -> set[str] | None:
        """The documents for a device name, or None if the name is not a known device."""
        if not name:
            return None
        wanted = name.strip().casefold()
        for known, docs in self._docs.items():
            if known.casefold() == wanted:
                return docs
        return None
