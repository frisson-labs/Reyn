"""The evaluated pure-neural request path; explicit clear and persistent declines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reyn.language.context import CHARACTERS


@dataclass(frozen=True)
class Reply:
    outcome: str
    row: str | None
    name: str
    changed: bool
    score: float
    reason: str
    asked: str = ""
    stand_in: bool = False
    said: str = ""
    extra_buttons: bool = False
    character: str = ""


class Steering:
    def __init__(self, policy: Any, router: Any, *, character: str):
        if character not in CHARACTERS:
            raise ValueError("unsupported policy character context")
        self.policy, self.router, self.character = policy, router, character
        self.current, self.holding = -1, None

    def say(self, text: str, *, stage: str | None = None) -> Reply:
        # Match OriginalR0Frontend: no character/stage parser, repair, or exact lookup.
        route = self.router.route(text)
        before = self.current
        if route.outcome == "clear":
            self.current, self.holding = -1, None
        elif route.outcome != "abstain":
            self.current = int(self.policy.command_ids([route.row])[0])
            self.holding = route.row
        return Reply(
            route.outcome,
            route.row,
            route.name,
            self.current != before,
            route.score,
            route.reason,
            asked=route.outcome,
            character=self.character,
        )
