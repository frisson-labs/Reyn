"""Experimental Fox doubles inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reyn_doubles.inference import Doubles

__all__ = ["Doubles"]


def __getattr__(name: str) -> Any:
    if name == "Doubles":
        from reyn_doubles.inference import Doubles

        return Doubles
    raise AttributeError(name)
