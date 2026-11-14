"""QuPath GeoJSON: one file per image, exported from QuPath's annotation export.

Coordinates are pixels of the full-resolution image with the origin at the top-left
corner, which is what histocel uses, so no conversion is needed. The class comes from
each feature's ``properties.classification.name``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from histocel.core.geometry import MIN_POLYGON_POINTS
from histocel.project import SourceSpec
from histocel.sources import RawCell, RawSlide, SourceError, find_image, image_size, register


def _exterior(geometry: dict[str, Any]) -> list[list[float]] | None:
    """Outer ring of a Polygon, or of the largest part of a MultiPolygon."""
    kind, coordinates = geometry.get("type"), geometry.get("coordinates") or []
    if kind == "Polygon" and coordinates:
        return coordinates[0]
    if kind == "MultiPolygon" and coordinates:
        return max((part[0] for part in coordinates if part), key=len, default=None)
    return None


def _class_name(properties: dict[str, Any]) -> str | None:
    classification = properties.get("classification")
    if isinstance(classification, dict):
        return classification.get("name")
    return classification if isinstance(classification, str) else None


def _read_file(path: Path, images_dir: Path) -> RawSlide:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SourceError(f"{path.name} is not valid JSON: {error}") from None
    features = data.get("features") if isinstance(data, dict) else data
    if not isinstance(features, list):
        raise SourceError(f"{path.name} is not a GeoJSON FeatureCollection or feature list")

    slide_id = path.stem
    image = find_image(images_dir, slide_id)
    width, height = image_size(image)
    cells, unclassified, unsupported = [], 0, 0
    for feature in features:
        ring = _exterior(feature.get("geometry") or {})
        if ring is None or len(ring) - 1 < MIN_POLYGON_POINTS:
            unsupported += 1
            continue
        name = _class_name(feature.get("properties") or {})
        if not name:
            unclassified += 1
            continue
        # GeoJSON repeats the first point at the end of a ring; histocel does not.
        points = ring[:-1] if ring[0] == ring[-1] else ring
        cells.append(RawCell(str(name), tuple((float(x), float(y)) for x, y, *_ in points)))

    notes = []
    if unclassified:
        notes.append(f"{unclassified} object(s) without a classification were skipped")
    if unsupported:
        notes.append(f"{unsupported} object(s) that are not polygons were skipped")
    return RawSlide(slide_id, image.name, width, height, tuple(cells), tuple(notes))


@register("qupath")
def read(spec: SourceSpec, images_dir: Path) -> list[RawSlide]:
    files = sorted(spec.path.glob("*.geojson")) if spec.path.is_dir() else [spec.path]
    if not files:
        raise SourceError(f"no .geojson files in {spec.path}")
    return [_read_file(path, images_dir) for path in files]
