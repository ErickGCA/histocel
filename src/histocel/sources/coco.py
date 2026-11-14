"""COCO instance segmentation JSON, as exported by most annotation tools."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from histocel.core.annotation import SlideAnnotation
from histocel.core.geometry import MIN_POLYGON_POINTS
from histocel.project import SourceSpec
from histocel.sources import RawCell, RawSlide, SourceError, register, slide_id_from


def _polygon(flat: list[float]) -> tuple[tuple[float, float], ...]:
    return tuple((float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat) - 1, 2))


@register("coco")
def read(spec: SourceSpec, images_dir: Path) -> list[RawSlide]:
    try:
        data = json.loads(spec.path.read_text(encoding="utf-8"))
        categories = {c["id"]: str(c["name"]) for c in data["categories"]}
        images = {i["id"]: i for i in data["images"]}
        annotations = data["annotations"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise SourceError(f"{spec.path.name} is not a COCO annotation file: {error}") from None

    cells: dict[Any, list[RawCell]] = {image_id: [] for image_id in images}
    skipped: dict[Any, int] = {}
    for annotation in annotations:
        image_id = annotation.get("image_id")
        if image_id not in images:
            continue
        segmentation = annotation.get("segmentation")
        # Polygons come as a list of flat [x0, y0, x1, y1, ...] rings. Anything else
        # (RLE masks, missing outlines) cannot become a polygon here.
        rings = segmentation if isinstance(segmentation, list) else []
        rings = [r for r in rings if isinstance(r, list) and len(r) >= 2 * MIN_POLYGON_POINTS]
        if not rings or annotation.get("category_id") not in categories:
            skipped[image_id] = skipped.get(image_id, 0) + 1
            continue
        largest = max(rings, key=len)
        cells[image_id].append(RawCell(categories[annotation["category_id"]], _polygon(largest)))

    slides = []
    for image_id, info in images.items():
        if not cells[image_id] and image_id not in skipped:
            continue  # exports often list images nobody annotated
        notes = ()
        if skipped.get(image_id):
            notes = (f"{skipped[image_id]} annotation(s) without a polygon outline were skipped",)
        slides.append(
            RawSlide(
                slide_id=slide_id_from(str(info["file_name"])),
                image=Path(str(info["file_name"]).replace("\\", "/")).name,
                width=int(info["width"]),
                height=int(info["height"]),
                cells=tuple(cells[image_id]),
                notes=notes,
            )
        )
    return slides


def to_coco(annotations: Iterable[SlideAnnotation]) -> dict[str, Any]:
    """Build a COCO document from canonical annotations (used for export and for tests)."""
    annotations = list(annotations)
    names = sorted({cell.label for a in annotations for cell in a.cells})
    category_ids = {name: index + 1 for index, name in enumerate(names)}
    images, items = [], []
    for image_id, annotation in enumerate(annotations, start=1):
        images.append(
            {
                "id": image_id,
                "file_name": annotation.image,
                "width": annotation.width,
                "height": annotation.height,
            }
        )
        for cell in annotation.cells:
            xs = [p[0] for p in cell.polygon]
            ys = [p[1] for p in cell.polygon]
            items.append(
                {
                    "id": len(items) + 1,
                    "image_id": image_id,
                    "category_id": category_ids[cell.label],
                    "segmentation": [[round(v, 2) for point in cell.polygon for v in point]],
                    "bbox": [
                        round(min(xs), 2),
                        round(min(ys), 2),
                        round(max(xs) - min(xs), 2),
                        round(max(ys) - min(ys), 2),
                    ],
                    "iscrowd": 0,
                }
            )
    return {
        "images": images,
        "annotations": items,
        "categories": [{"id": i, "name": name} for name, i in category_ids.items()],
    }
