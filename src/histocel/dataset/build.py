"""Split the slides, cut them into tiles and write a YOLO segmentation dataset.

The dataset is never edited by hand and never kept as a source of truth: the same
annotations and configuration always produce the same files, and ``manifest.json``
records a hash that identifies exactly what a model was trained on.

Test slides are not part of the dataset at all. They are only ever read by
evaluation, as whole slides, from the canonical annotations.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from histocel import __version__
from histocel.check import ERROR, run_checks
from histocel.core.annotation import SlideAnnotation
from histocel.core.tiling import Tile, cells_in_tile, tile_grid
from histocel.project import Project

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
SIDES = ("train", "val")


class BuildError(ValueError):
    """Raised when the dataset cannot be built from the project as it is."""


@dataclass
class BuildReport:
    path: Path
    dataset_hash: str
    classes: list[str]
    slides: dict[str, int] = field(default_factory=dict)
    tiles: dict[str, int] = field(default_factory=dict)
    background_tiles: dict[str, int] = field(default_factory=dict)
    cells: dict[str, int] = field(default_factory=dict)


def tile_name(slide_id: str, tile: Tile) -> str:
    """File stem of a tile; it carries the tile's position on the slide."""
    return f"{slide_id}__y{tile.y0:05d}_x{tile.x0:05d}"


def parse_tile_name(stem: str) -> tuple[str, int, int]:
    """Inverse of :func:`tile_name`: ``(slide_id, x0, y0)``."""
    slide_id, _, position = stem.rpartition("__")
    y_part, x_part = position.split("_")
    return slide_id, int(x_part[1:]), int(y_part[1:])


def _keep_background(slide_id: str, tile: Tile, share: float, seed: int) -> bool:
    """Deterministic choice of which empty tiles become background examples."""
    if share <= 0:
        return False
    if share >= 1:
        return True
    digest = hashlib.sha256(f"{seed}:{slide_id}:{tile.x0}:{tile.y0}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < share


def _label_line(class_index: int, polygon: Sequence[tuple[float, float]], tile: Tile) -> str:
    values = []
    for x, y in polygon:
        values.append(f"{min(max(x / tile.width, 0.0), 1.0):.6f}")
        values.append(f"{min(max(y / tile.height, 0.0), 1.0):.6f}")
    return f"{class_index} {' '.join(values)}"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_dataset(
    project: Project, annotations: Sequence[SlideAnnotation], name: str = "default"
) -> BuildReport:
    """Write ``datasets/<name>/`` for the project, replacing any previous build."""
    result = run_checks(project, annotations)
    if not result.ok or result.split is None:
        errors = "\n".join(f"  - {f.message}" for f in result.findings if f.level == ERROR)
        raise BuildError(f"the project has problems that must be fixed first:\n{errors}")
    if not result.split.val:
        raise BuildError(
            "no slides left for validation. Add slides or groups, or raise split.val_fraction."
        )

    grouping = project.dataset.grouping
    classes = project.taxonomy.names(grouping)
    class_index = {name_: index for index, name_ in enumerate(classes)}
    by_id = {a.slide_id: a for a in annotations}

    out = project.root / "datasets" / name
    if out.exists():
        shutil.rmtree(out)
    report = BuildReport(path=out, dataset_hash="", classes=classes)
    files: dict[str, str] = {}

    for side in SIDES:
        images_dir = out / "images" / side
        labels_dir = out / "labels" / side
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        slide_ids = getattr(result.split, side)
        report.slides[side] = len(slide_ids)
        report.tiles[side] = report.background_tiles[side] = report.cells[side] = 0

        for slide_id in slide_ids:
            annotation = by_id[slide_id]
            polygons = [cell.polygon for cell in annotation.cells]
            with Image.open(project.images_dir / annotation.image) as opened:
                image = opened.convert("RGB")
            if image.size != (annotation.width, annotation.height):
                raise BuildError(
                    f"{slide_id}: image is {image.size[0]}x{image.size[1]} but the annotation "
                    f"says {annotation.width}x{annotation.height}"
                )
            for tile in tile_grid(
                annotation.width, annotation.height, project.tiling.size, project.tiling.overlap
            ):
                visible = cells_in_tile(polygons, tile, project.dataset.min_visible)
                if not visible and not _keep_background(
                    slide_id, tile, project.dataset.background, project.split.seed
                ):
                    continue
                lines = [
                    _label_line(
                        class_index[
                            project.taxonomy.group_of(annotation.cells[seen.index].label, grouping)
                        ],
                        seen.polygon,
                        tile,
                    )
                    for seen in visible
                ]
                stem = tile_name(slide_id, tile)
                crop = image.crop(tile.rect)
                crop.save(images_dir / f"{stem}.png")
                label_text = "\n".join(lines) + ("\n" if lines else "")
                (labels_dir / f"{stem}.txt").write_text(label_text, encoding="utf-8", newline="\n")

                # Hash the content, not the encoded file, so the hash does not depend on
                # the PNG encoder of whoever builds it.
                files[f"images/{side}/{stem}.png"] = _sha256(crop.tobytes())
                files[f"labels/{side}/{stem}.txt"] = _sha256(label_text.encode())
                report.tiles[side] += 1
                report.cells[side] += len(lines)
                report.background_tiles[side] += not lines

    # No `path` key: the training engine then resolves the folders relative to this file,
    # so the dataset can be moved or uploaded as it is.
    data_yaml = "train: images/train\nval: images/val\nnames:\n" + "".join(
        f"  {index}: {name_}\n" for index, name_ in enumerate(classes)
    )
    (out / "data.yaml").write_text(data_yaml, encoding="utf-8", newline="\n")

    listing = "".join(f"{path}:{digest}\n" for path, digest in sorted(files.items()))
    report.dataset_hash = _sha256((data_yaml + listing).encode())
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_hash": report.dataset_hash,
        "histocel_version": __version__,
        "classes": classes,
        "config": {
            "grouping": grouping,
            "tiling": {"size": project.tiling.size, "overlap": project.tiling.overlap},
            "min_visible": project.dataset.min_visible,
            "background": project.dataset.background,
            "val_fraction": project.split.val_fraction,
            "seed": project.split.seed,
        },
        "split": {side: list(getattr(result.split, side)) for side in ("train", "val", "test")},
        "counts": {
            "slides": report.slides,
            "tiles": report.tiles,
            "background_tiles": report.background_tiles,
            "cells": report.cells,
        },
        "files": dict(sorted(files.items())),
    }
    (out / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return report


def load_manifest(dataset_dir: Path) -> dict[str, Any]:
    """Read the manifest of a built dataset."""
    path = dataset_dir / MANIFEST_NAME
    if not path.is_file():
        raise BuildError(f"{dataset_dir} is not a built dataset (no {MANIFEST_NAME})")
    return json.loads(path.read_text(encoding="utf-8"))
