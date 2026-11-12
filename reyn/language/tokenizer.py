"""WordPiece tokenization for the packaged text encoder."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import Final

PAD: Final[str] = "[PAD]"
UNK: Final[str] = "[UNK]"
CLS: Final[str] = "[CLS]"
SEP: Final[str] = "[SEP]"
MAX_WORD: Final[int] = 100


def _is_whitespace(char: str) -> bool:
    return char in " \t\n\r" or unicodedata.category(char) == "Zs"


def _is_control(char: str) -> bool:
    if char in "\t\n\r":
        return False
    return unicodedata.category(char).startswith("C")


def _is_punctuation(char: str) -> bool:
    code = ord(char)
    if 33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or (123 <= code <= 126):
        return True
    return unicodedata.category(char).startswith("P")


def _is_chinese(code: int) -> bool:
    return (
        19968 <= code <= 40959
        or 13312 <= code <= 19903
        or 131072 <= code <= 173791
        or (173824 <= code <= 177983)
        or (177984 <= code <= 178207)
        or (178208 <= code <= 183983)
        or (63744 <= code <= 64255)
        or (194560 <= code <= 195103)
    )


def basic_tokens(text: str) -> list[str]:
    cleaned = []
    for char in text:
        code = ord(char)
        if code == 0 or code == 65533 or _is_control(char):
            continue
        if _is_chinese(code):
            cleaned.append(f" {char} ")
        else:
            cleaned.append(" " if _is_whitespace(char) else char)
    tokens: list[str] = []
    for word in "".join(cleaned).split():
        lowered = unicodedata.normalize("NFD", word.lower())
        stripped = "".join(char for char in lowered if unicodedata.category(char) != "Mn")
        current: list[str] = []
        for char in stripped:
            if _is_punctuation(char):
                if current:
                    tokens.append("".join(current))
                    current = []
                tokens.append(char)
            else:
                current.append(char)
        if current:
            tokens.append("".join(current))
    return tokens


class WordPiece:
    def __init__(self, vocab: Sequence[str]) -> None:
        self.vocab = tuple(vocab)
        self.ids = {token: index for index, token in enumerate(self.vocab)}
        if len(self.ids) != len(self.vocab):
            raise ValueError("a vocabulary entry is listed twice")
        for special in (PAD, UNK, CLS, SEP):
            if special not in self.ids:
                raise ValueError(f"the vocabulary lacks {special}")
        self.pad, self.unk, self.cls, self.sep = (self.ids[name] for name in (PAD, UNK, CLS, SEP))

    def pieces(self, word: str) -> list[str]:
        if len(word) > MAX_WORD:
            return [UNK]
        out: list[str] = []
        start = 0
        while start < len(word):
            end = len(word)
            found = None
            while start < end:
                piece = word[start:end] if start == 0 else "##" + word[start:end]
                if piece in self.ids:
                    found = piece
                    break
                end -= 1
            if found is None:
                return [UNK]
            out.append(found)
            start = end
        return out

    def encode(self, text: str, *, max_length: int = 64) -> list[int]:
        ids = [self.ids[piece] for word in basic_tokens(text) for piece in self.pieces(word)]
        return [self.cls, *ids[: max(0, max_length - 2)], self.sep]

    def batch(self, texts: Sequence[str], *, max_length: int = 64) -> tuple[list[list[int]], list[list[int]]]:
        rows = [self.encode(text, max_length=max_length) for text in texts]
        width = max((len(row) for row in rows), default=0)
        ids = [row + [self.pad] * (width - len(row)) for row in rows]
        mask = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        return (ids, mask)
