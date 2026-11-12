"""Text normalization used by the language router."""

from __future__ import annotations

import re
import unicodedata
from typing import Final

_QUOTES: Final[dict[int, str]] = {
    **{ord(mark): "'" for mark in "\u2018\u2019\u02bc`\u00b4"},
    **{ord(mark): '"' for mark in "\u201c\u201d"},
}
_NON_WORD: Final[re.Pattern[str]] = re.compile("[^a-z0-9]+")


def dedup_key(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).translate(_QUOTES).casefold().replace("'", "")
    return _NON_WORD.sub(" ", folded).strip()
