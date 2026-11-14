"""Polygon and rectangle primitives, in pixel coordinates.

A polygon is a sequence of ``(x, y)`` points, implicitly closed. A rectangle is
``(x0, y0, x1, y1)`` with ``x0 < x1`` and ``y0 < y1``.
"""

from __future__ import annotations

from collections.abc import Sequence

from shapely import make_valid
from shapely.geometry import MultiPolygon, box
from shapely.geometry import Polygon as _ShapelyPolygon
from shapely.geometry.base import BaseGeometry

Point = tuple[float, float]
Polygon = Sequence[Point]
Rect = tuple[float, float, float, float]

MIN_POLYGON_POINTS = 3


class InvalidPolygonError(ValueError):
    """Raised when a polygon cannot describe an area."""


def to_shape(polygon: Polygon) -> BaseGeometry:
    """Build a valid shapely geometry from a polygon, repairing self-intersections."""
    if len(polygon) < MIN_POLYGON_POINTS:
        raise InvalidPolygonError(
            f"a polygon needs at least {MIN_POLYGON_POINTS} points, got {len(polygon)}"
        )
    shape = _ShapelyPolygon(polygon)
    if not shape.is_valid:
        shape = make_valid(shape)
    if shape.is_empty or shape.area == 0:
        raise InvalidPolygonError("polygon has zero area")
    return shape


def _largest_polygon(shape: BaseGeometry) -> _ShapelyPolygon | None:
    """Return the largest polygonal part of a geometry, or None if it has none."""
    if isinstance(shape, _ShapelyPolygon):
        return None if shape.is_empty else shape
    parts = [
        g for g in getattr(shape, "geoms", []) if isinstance(g, _ShapelyPolygon | MultiPolygon)
    ]
    flat: list[_ShapelyPolygon] = []
    for part in parts:
        flat.extend(part.geoms if isinstance(part, MultiPolygon) else [part])
    flat = [p for p in flat if not p.is_empty]
    return max(flat, key=lambda p: p.area) if flat else None


def _exterior_points(shape: _ShapelyPolygon) -> list[Point]:
    coords = list(shape.exterior.coords)[:-1]  # shapely repeats the first point at the end
    return [(float(x), float(y)) for x, y in coords]


def area(polygon: Polygon) -> float:
    """Area of a polygon in square pixels."""
    return float(to_shape(polygon).area)


def bounds(polygon: Polygon) -> Rect:
    """Axis-aligned bounding rectangle of a polygon."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def rect_to_polygon(rect: Rect) -> list[Point]:
    """Corners of a rectangle, clockwise from the top-left."""
    x0, y0, x1, y1 = rect
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def translate(polygon: Polygon, dx: float, dy: float) -> list[Point]:
    """Move every point of a polygon by ``(dx, dy)``."""
    return [(x + dx, y + dy) for x, y in polygon]


def clip_to_rect(polygon: Polygon, rect: Rect) -> tuple[list[Point], float] | None:
    """Cut a polygon by a rectangle.

    Returns the largest piece that falls inside ``rect`` and the fraction of the
    original area it keeps, or ``None`` when nothing is left. The piece is a true
    geometric intersection: points are never just clamped to the border.
    """
    shape = to_shape(polygon)
    piece = _largest_polygon(shape.intersection(box(*rect)))
    if piece is None or piece.area == 0:
        return None
    return _exterior_points(piece), float(piece.area / shape.area)


def overlap_over_smaller(a: Polygon, b: Polygon) -> float:
    """Intersection area divided by the area of the smaller polygon.

    Unlike IoU, this stays close to 1 when one polygon is a truncated copy of the
    other, which is exactly what two tiles produce for a cell on their border.
    """
    shape_a, shape_b = to_shape(a), to_shape(b)
    smaller = min(shape_a.area, shape_b.area)
    return float(shape_a.intersection(shape_b).area / smaller)


def iou(a: Polygon, b: Polygon) -> float:
    """Intersection over union of two polygons."""
    shape_a, shape_b = to_shape(a), to_shape(b)
    union = shape_a.union(shape_b).area
    return float(shape_a.intersection(shape_b).area / union) if union else 0.0
