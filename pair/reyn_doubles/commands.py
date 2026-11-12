"""Exact tested phrases and persistent per-Fox addressing."""

from __future__ import annotations

from collections.abc import Iterable


class CommandState:
    """Held commands for Fox 1 and Fox 2, ordered by physical port within a team."""

    def __init__(self, phrases: Iterable[str]) -> None:
        self.phrases = frozenset(phrases)
        self._held: list[str | None] = [None, None]

    @property
    def current(self) -> tuple[str | None, str | None]:
        return self._held[0], self._held[1]

    @staticmethod
    def recipients(recipient: str) -> tuple[int, ...]:
        try:
            return {"both": (0, 1), "fox1": (0,), "fox2": (1,)}[recipient]
        except (KeyError, TypeError):
            raise ValueError("recipient must be 'both', 'fox1' or 'fox2'") from None

    def say(self, phrase: str, *, recipient: str = "both") -> tuple[str | None, str | None]:
        rows = self.recipients(recipient)
        if phrase not in self.phrases:
            raise ValueError("Unsupported phrase; choose an exact entry from commands.json")
        for row in rows:
            self._held[row] = phrase
        return self.current

    def clear(self, *, recipient: str = "both") -> tuple[str | None, str | None]:
        for row in self.recipients(recipient):
            self._held[row] = None
        return self.current
