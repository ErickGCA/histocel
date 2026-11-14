"""Cut a slide into overlapping tiles and put per-tile results back together.

The guarantee this module exists for: if the overlap is at least as large as the
largest cell, every cell appears whole in at least one tile. Combined with
:func:`merge_duplicates`, a cell on the border between tiles is counted once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from shapely import STRtree

from histocel.core.geometry import Point, Polygon, Rect, clip_to_rect, to_shape, translate


@dataclass(frozen=True)
class Tile:
    """A rectangular window on a slide, in slide pixel coordinates."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def rect(self) -> Rect:
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def _starts(length: int, size: int, overlap: int) -> list[int]:
    """Window start positions along one axis, covering ``[0, length)`` completely."""
    if length <= size:
        return [0]
    step = size - overlap
    starts = list(range(0, length - size + 1, step))
    if starts[-1] != length - size:
        starts.append(length - size)  # last window is shifted back to end at the edge
    return starts


def tile_grid(width: int, height: int, size: int, overlap: int = 0) -> list[Tile]:
    """Windows of ``size`` x ``size`` pixels covering a ``width`` x ``height`` slide.

    Neighbouring windows share at least ``overlap`` pixels. A slide smaller than
    ``size`` along an axis yields a single, smaller window on that axis.
    """
    if size <= 0:
        raise ValueError("tile size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than the tile size")
    if width <= 0 or height <= 0:
        raise ValueError("slide dimensions must be positive")
    return [
        Tile(x, y, min(x + size, width), min(y + size, height))
        for y in _starts(height, size, overlap)
        for x in _starts(width, size, overlap)
    ]


@dataclass(frozen=True)
class TileCell:
    """A cell as seen from inside one tile."""

    index: int  # position of the cell in the slide-level list it came from
    polygon: list[Point]  # tile coordinates
    visible: float  # fraction of the cell's area that falls inside the tile


def cells_in_tile(
    polygons: Sequence[Polygon], tile: Tile, min_visible: float = 0.25
) -> list[TileCell]:
    """Cells visible in ``tile``, cut by its border and moved to tile coordinates.

    A cell is kept when at least ``min_visible`` of its area is inside the tile.
    Smaller slivers are dropped: they carry too little of the cell to learn from.
    """
    if not 0 < min_visible <= 1:
        raise ValueError("min_visible must be in (0, 1]")
    found = []
    for index, polygon in enumerate(polygons):
        clipped = clip_to_rect(polygon, tile.rect)
        if clipped is None:
            continue
        piece, visible = clipped
        if visible >= min_visible:
            found.append(TileCell(index, translate(piece, -tile.x0, -tile.y0), visible))
    return found


def to_slide_coords(polygon: Polygon, tile: Tile) -> list[Point]:
    """Move a polygon from tile coordinates back to slide coordinates."""
    return translate(polygon, tile.x0, tile.y0)


@dataclass(frozen=True)
class Detection:
    """One predicted cell, in slide coordinates."""

    label: str
    polygon: list[Point]
    score: float = 1.0
    whole: bool = True  # False when the outline was cut by the border of its tile


def merge_duplicates(detections: Sequence[Detection], threshold: float = 0.5) -> list[Detection]:
    """Collapse detections of the same cell coming from overlapping tiles.

    Greedy suppression: outlines that were not cut by a tile border come first, then
    the most confident (ties broken by larger area, so the whole copy beats the
    truncated one) and a candidate is
    dropped when it overlaps an already kept detection by more than ``threshold``
    of the smaller area. Merging ignores the label: two tiles disagreeing on the
    class of one cell still describe one cell.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    shapes = [to_shape(d.polygon) for d in detections]
    order = sorted(
        range(len(detections)),
        key=lambda i: (detections[i].whole, detections[i].score, shapes[i].area),
        reverse=True,
    )
    tree = STRtree(shapes)
    kept: set[int] = set()
    for i in order:
        shape = shapes[i]
        duplicate = False
        for j in tree.query(shape):
            j = int(j)
            if j in kept:
                smaller = min(shape.area, shapes[j].area)
                if shape.intersection(shapes[j]).area / smaller > threshold:
                    duplicate = True
                    break
        if not duplicate:
            kept.add(i)
    return [detections[i] for i in order if i in kept]
