"""Assisted annotation: turn a quick human hint into a full outline.

Drawing a box around a cell is several times faster than drawing its outline. An
outliner takes the image and the boxes and returns one polygon per box. What comes
out is recorded as *assisted* and not reviewed: it can train a model, but it cannot
grade one until a person has checked it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from histocel.core.geometry import Point, Rect


class AssistError(RuntimeError):
    """Raised when an outliner is unknown, not installed, or fails."""


# (image, boxes) -> one outline per box, or None where no outline could be made.
Outliner = Callable[[Any, Sequence[Rect]], list[tuple[Point, ...] | None]]

_FACTORIES: dict[str, Callable[[str | None], Outliner]] = {}


def register_assist(name: str, factory: Callable[[str | None], Outliner]) -> None:
    """Make an outliner available under ``name``. The factory receives the model name."""
    _FACTORIES[name] = factory


def available_assists() -> list[str]:
    return sorted(_FACTORIES)


def get_outliner(name: str, model: str | None = None) -> Outliner:
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise AssistError(
            f"unknown assist {name!r}. Available: {', '.join(available_assists())}"
        ) from None
    return factory(model)


def _sam(model: str | None) -> Outliner:
    # Imported here so that histocel works without the optional extra installed.
    from histocel.assist.sam import sam_outliner

    return sam_outliner(model)


register_assist("sam", _sam)
