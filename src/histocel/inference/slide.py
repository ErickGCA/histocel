"""Whole-slide inference.

The slide is cut with the same tile size and overlap the model was trained with,
each tile is analysed on its own, every detection is moved back to slide
coordinates and detections of the same cell coming from neighbouring tiles are
merged, so a cell on a tile border is counted once.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from histocel.core.annotation import METHOD_MODEL, Cell, SlideAnnotation
from histocel.core.geometry import InvalidPolygonError, bounds, to_shape
from histocel.core.tiling import Detection, Tile, merge_duplicates, tile_grid, to_slide_coords
from histocel.engines import EngineError, get_engine
from histocel.engines.base import Predictor

MODEL_FILE = "model.json"
BORDER_TOLERANCE = 1.5  # pixels from a tile edge at which an outline counts as cut by it
PALETTE = [(230, 60, 60), (40, 150, 240), (60, 190, 90), (240, 170, 30), (170, 90, 220)]


class InferenceError(RuntimeError):
    """Raised when a model cannot be loaded or used."""


@dataclass(frozen=True)
class ModelCard:
    """What a model folder says about itself."""

    path: Path
    engine: str
    weights: Path
    classes: tuple[str, ...]
    tile_size: int
    tile_overlap: int
    image_size: int
    dataset_hash: str


def load_model_card(model_dir: Path) -> ModelCard:
    card_file = model_dir / MODEL_FILE
    if not card_file.is_file():
        raise InferenceError(
            f"{model_dir} is not a histocel model folder (no {MODEL_FILE}). "
            "Models are written by `histocel train` under runs/<run>/model."
        )
    try:
        data = json.loads(card_file.read_text(encoding="utf-8"))
        card = ModelCard(
            path=model_dir,
            engine=data["engine"],
            weights=model_dir / data["weights"],
            classes=tuple(data["classes"]),
            tile_size=int(data["tiling"]["size"]),
            tile_overlap=int(data["tiling"]["overlap"]),
            image_size=int(data["image_size"]),
            dataset_hash=data.get("trained_on", {}).get("dataset_hash", ""),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise InferenceError(f"{card_file} is damaged: {error}") from None
    if not card.weights.is_file():
        raise InferenceError(f"weights file {card.weights} is missing")
    return card


def open_predictor(
    card: ModelCard, confidence: float = 0.25, device: str | None = None
) -> Predictor:
    """Load the model's weights with the engine that trained them."""
    try:
        return get_engine(card.engine).load(card.weights, card.image_size, confidence, device)
    except EngineError as error:
        raise InferenceError(str(error)) from None


def _is_whole(polygon: Sequence[tuple[float, float]], tile: Tile, width: int, height: int) -> bool:
    """False when the outline touches a tile edge that is not also the slide's edge."""
    x0, y0, x1, y1 = bounds(polygon)
    return not (
        (tile.x0 > 0 and x0 <= BORDER_TOLERANCE)
        or (tile.y0 > 0 and y0 <= BORDER_TOLERANCE)
        or (tile.x1 < width and x1 >= tile.width - BORDER_TOLERANCE)
        or (tile.y1 < height and y1 >= tile.height - BORDER_TOLERANCE)
    )


def predict_slide(
    image: Image.Image, predictor: Predictor, card: ModelCard, merge_threshold: float = 0.5
) -> list[Detection]:
    """Cells found on a whole slide, in slide coordinates, each counted once."""
    image = image.convert("RGB")
    width, height = image.size
    found: list[Detection] = []
    for tile in tile_grid(width, height, card.tile_size, card.tile_overlap):
        for detection in predictor(image.crop(tile.rect)):
            if not 0 <= detection.class_index < len(card.classes):
                raise InferenceError(
                    f"the model returned class {detection.class_index}, but its card lists "
                    f"{len(card.classes)} classes"
                )
            try:
                to_shape(detection.polygon)
            except InvalidPolygonError:
                continue  # a degenerate outline carries no cell
            found.append(
                Detection(
                    label=card.classes[detection.class_index],
                    polygon=to_slide_coords(detection.polygon, tile),
                    score=detection.score,
                    whole=_is_whole(detection.polygon, tile, width, height),
                )
            )
    return merge_duplicates(found, merge_threshold)


def count(detections: Sequence[Detection], classes: Sequence[str]) -> dict[str, int]:
    """Cells per class, in the model's class order, including zeros."""
    totals = dict.fromkeys(classes, 0)
    for detection in detections:
        totals[detection.label] += 1
    return totals


def to_annotation(
    slide_id: str,
    image_name: str,
    size: tuple[int, int],
    detections: Sequence[Detection],
    source: str,
) -> SlideAnnotation:
    """Predictions as a canonical annotation, marked as made by a model and not reviewed."""
    return SlideAnnotation(
        slide_id=slide_id,
        image=image_name,
        width=size[0],
        height=size[1],
        cells=tuple(
            Cell(label=d.label, polygon=tuple(d.polygon), method=METHOD_MODEL, source=source)
            for d in detections
        ),
    )


def render(
    image: Image.Image, detections: Sequence[Detection], classes: Sequence[str]
) -> Image.Image:
    """A copy of the slide with every detected cell outlined in its class colour."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    colors = {name: PALETTE[index % len(PALETTE)] for index, name in enumerate(classes)}
    line = max(1, round(max(canvas.size) / 600))
    for detection in detections:
        points = [(x, y) for x, y in detection.polygon]
        draw.line([*points, points[0]], fill=colors[detection.label], width=line)
    return canvas
