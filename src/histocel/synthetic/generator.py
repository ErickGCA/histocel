"""Draw fake slides whose every cell is known exactly.

A synthetic slide is a pinkish, noisy background with coloured ellipses standing in
for cells. It looks nothing like real tissue and is not meant to: its job is to give
the pipeline an input whose correct output is known, so plumbing errors (a cell
counted twice at a tile border, a class id shifted by one) show up as exact failures.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from histocel.core.annotation import METHOD_MANUAL, Cell, SlideAnnotation
from histocel.core.geometry import Point

OUTLINE_POINTS = 24
BACKGROUND = (238, 204, 220)  # eosin-like pink
MAX_PLACEMENT_ATTEMPTS = 200


@dataclass(frozen=True)
class CellStyle:
    """How one class of fake cell is drawn."""

    color: tuple[int, int, int]
    radius: tuple[float, float]  # min and max semi-axis, in pixels
    weight: float = 1.0  # relative frequency


def default_styles() -> dict[str, CellStyle]:
    """Two easy-to-tell-apart classes: large pale cells and small dark ones."""
    return {
        "large": CellStyle(color=(150, 110, 190), radius=(14.0, 22.0), weight=3.0),
        "small": CellStyle(color=(70, 40, 110), radius=(6.0, 10.0), weight=1.0),
    }


def _ellipse(cx: float, cy: float, rx: float, ry: float, angle: float) -> list[Point]:
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    points = []
    for k in range(OUTLINE_POINTS):
        t = 2 * math.pi * k / OUTLINE_POINTS
        x, y = rx * math.cos(t), ry * math.sin(t)
        points.append((cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a))
    return points


def generate_slide(
    slide_id: str,
    width: int = 640,
    height: int = 480,
    n_cells: int = 40,
    styles: Mapping[str, CellStyle] | None = None,
    seed: int = 0,
    group: str | None = None,
) -> tuple[Image.Image, SlideAnnotation]:
    """Create one synthetic slide and its exact annotation.

    Cells never overlap and never touch the slide border, so the answer key is
    unambiguous. The same arguments always produce the same slide.
    """
    styles = dict(styles) if styles is not None else default_styles()
    rng = np.random.default_rng(seed)
    labels = list(styles)
    weights = np.array([styles[name].weight for name in labels], dtype=float)
    weights /= weights.sum()

    noise = rng.normal(0, 6, size=(height, width, 3))
    base = np.clip(np.array(BACKGROUND, dtype=float) + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(image)

    placed: list[tuple[float, float, float]] = []  # centre x, centre y, bounding radius
    cells: list[Cell] = []
    for _ in range(n_cells):
        label = labels[int(rng.choice(len(labels), p=weights))]
        style = styles[label]
        for _attempt in range(MAX_PLACEMENT_ATTEMPTS):
            rx = float(rng.uniform(*style.radius))
            ry = float(rng.uniform(*style.radius))
            reach = max(rx, ry)
            if width <= 2 * reach + 4 or height <= 2 * reach + 4:
                raise ValueError("slide is too small for the requested cell size")
            cx = float(rng.uniform(reach + 2, width - reach - 2))
            cy = float(rng.uniform(reach + 2, height - reach - 2))
            if all(math.hypot(cx - px, cy - py) > reach + pr + 2 for px, py, pr in placed):
                break
        else:
            raise ValueError(
                f"could not place {n_cells} non-overlapping cells on a {width}x{height} slide; "
                "ask for fewer cells or a larger slide"
            )
        placed.append((cx, cy, reach))
        outline = _ellipse(cx, cy, rx, ry, float(rng.uniform(0, math.pi)))
        draw.polygon(outline, fill=style.color)
        cells.append(
            Cell(
                label=label,
                polygon=tuple(outline),
                method=METHOD_MANUAL,
                reviewed=True,
                source="synthetic",
                annotator="histocel.synthetic",
            )
        )

    image = image.filter(ImageFilter.GaussianBlur(radius=0.6))
    annotation = SlideAnnotation(
        slide_id=slide_id,
        image=f"{slide_id}.png",
        width=width,
        height=height,
        cells=tuple(cells),
        group=group,
        searched_labels=tuple(labels),
    )
    return image, annotation
