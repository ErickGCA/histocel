"""The canonical annotation: one document per slide, in slide pixel coordinates.

Every annotation tool is converted to this format on the way in, and every
training format is produced from it on the way out. Each cell records where its
outline came from, so a dataset can tell human work from machine work.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from histocel.core.geometry import MIN_POLYGON_POINTS, Point, Rect

SCHEMA_VERSION = 1

METHOD_MANUAL = "manual"  # outline drawn by a person
METHOD_ASSISTED = "assisted"  # outline produced by a tool from a human hint (e.g. a box)
METHOD_MODEL = "model"  # outline predicted by a trained model
METHODS = (METHOD_MANUAL, METHOD_ASSISTED, METHOD_MODEL)


class AnnotationError(ValueError):
    """Raised when a document does not follow the canonical format."""


@dataclass(frozen=True)
class Cell:
    """One annotated cell."""

    label: str
    polygon: tuple[Point, ...]
    method: str = METHOD_MANUAL
    reviewed: bool = False
    source: str | None = None  # where it came from, e.g. "coco:batch1.json"
    annotator: str | None = None
    box: Rect | None = None  # the original human hint, when the outline was assisted

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise AnnotationError(f"method must be one of {METHODS}, got {self.method!r}")
        if len(self.polygon) < MIN_POLYGON_POINTS:
            raise AnnotationError(
                f"a cell outline needs at least {MIN_POLYGON_POINTS} points, "
                f"got {len(self.polygon)}"
            )

    @property
    def is_gold(self) -> bool:
        """True when a person drew or checked the outline; only gold cells may grade a model."""
        return self.method == METHOD_MANUAL or self.reviewed


@dataclass(frozen=True)
class SlideAnnotation:
    """Everything known about one slide."""

    slide_id: str
    image: str  # file name of the image, relative to the project's image folder
    width: int
    height: int
    cells: tuple[Cell, ...] = ()
    group: str | None = None  # animal, patient or block the slide belongs to
    # Classes the annotator looked for. ``None`` means "all classes of the project".
    # Without this, a missing class could mean "absent" or "nobody looked".
    searched_labels: tuple[str, ...] | None = field(default=None)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise AnnotationError("slide dimensions must be positive")

    @property
    def is_gold(self) -> bool:
        """True when every cell on the slide is gold."""
        return all(cell.is_gold for cell in self.cells)

    def counts(self) -> dict[str, int]:
        """Number of cells per label."""
        totals: dict[str, int] = {}
        for cell in self.cells:
            totals[cell.label] = totals.get(cell.label, 0) + 1
        return totals


def _points(raw: Sequence[Sequence[float]]) -> tuple[Point, ...]:
    return tuple((float(x), float(y)) for x, y in raw)


def cell_to_dict(cell: Cell) -> dict[str, Any]:
    return {
        "label": cell.label,
        "polygon": [[round(x, 2), round(y, 2)] for x, y in cell.polygon],
        "method": cell.method,
        "reviewed": cell.reviewed,
        "source": cell.source,
        "annotator": cell.annotator,
        "box": [round(v, 2) for v in cell.box] if cell.box is not None else None,
    }


def cell_from_dict(data: Mapping[str, Any]) -> Cell:
    box = data.get("box")
    return Cell(
        label=data["label"],
        polygon=_points(data["polygon"]),
        method=data.get("method", METHOD_MANUAL),
        reviewed=bool(data.get("reviewed", False)),
        source=data.get("source"),
        annotator=data.get("annotator"),
        box=tuple(float(v) for v in box) if box is not None else None,  # type: ignore[arg-type]
    )


def to_dict(annotation: SlideAnnotation) -> dict[str, Any]:
    """Plain, JSON-ready representation of a slide annotation."""
    return {
        "schema_version": SCHEMA_VERSION,
        "slide_id": annotation.slide_id,
        "image": annotation.image,
        "width": annotation.width,
        "height": annotation.height,
        "group": annotation.group,
        "searched_labels": (
            list(annotation.searched_labels) if annotation.searched_labels is not None else None
        ),
        "cells": [cell_to_dict(cell) for cell in annotation.cells],
    }


def from_dict(data: Mapping[str, Any]) -> SlideAnnotation:
    """Rebuild a slide annotation, refusing documents written by a newer schema."""
    found = data.get("schema_version")
    if found != SCHEMA_VERSION:
        raise AnnotationError(
            f"unsupported schema_version {found!r}; this histocel reads version {SCHEMA_VERSION}"
        )
    try:
        searched = data.get("searched_labels")
        return SlideAnnotation(
            slide_id=str(data["slide_id"]),
            image=str(data["image"]),
            width=int(data["width"]),
            height=int(data["height"]),
            group=data.get("group"),
            searched_labels=tuple(searched) if searched is not None else None,
            cells=tuple(cell_from_dict(c) for c in data.get("cells", ())),
        )
    except KeyError as missing:
        raise AnnotationError(f"missing field {missing}") from None
