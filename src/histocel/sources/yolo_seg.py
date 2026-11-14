"""YOLO segmentation labels: one ``.txt`` per image, normalised polygon coordinates.

Each line is ``class_index x1 y1 x2 y2 ...`` with coordinates between 0 and 1.
The format stores neither class names nor image sizes, so the source must list
``names`` in ``histocel.yaml`` and the images must be in the project's image folder.
"""

from __future__ import annotations

from pathlib import Path

from histocel.core.geometry import MIN_POLYGON_POINTS
from histocel.project import SourceSpec
from histocel.sources import RawCell, RawSlide, SourceError, find_image, image_size, register


@register("yolo-seg")
def read(spec: SourceSpec, images_dir: Path) -> list[RawSlide]:
    if spec.names is None:
        raise SourceError(
            f"{spec.path.name}: yolo-seg labels only store class numbers. "
            "Add `names: [first-class, second-class, ...]` to this source."
        )
    if not spec.path.is_dir():
        raise SourceError(f"{spec.path} must be a folder of .txt label files")

    slides = []
    for label_file in sorted(spec.path.glob("*.txt")):
        slide_id = label_file.stem
        image = find_image(images_dir, slide_id)
        width, height = image_size(image)
        cells = []
        for number, line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            where = f"{label_file.name}:{number}"
            try:
                index = int(parts[0])
                values = [float(v) for v in parts[1:]]
            except ValueError:
                raise SourceError(f"{where}: expected numbers, got {line[:40]!r}") from None
            if len(values) % 2 or len(values) < 2 * MIN_POLYGON_POINTS:
                raise SourceError(
                    f"{where}: a polygon needs at least {MIN_POLYGON_POINTS} x/y pairs. "
                    "Lines with 4 values are boxes, not outlines."
                )
            if not 0 <= index < len(spec.names):
                raise SourceError(
                    f"{where}: class {index} has no name; `names` lists {len(spec.names)}"
                )
            polygon = tuple(
                (values[i] * width, values[i + 1] * height) for i in range(0, len(values), 2)
            )
            cells.append(RawCell(spec.names[index], polygon))
        slides.append(RawSlide(slide_id, image.name, width, height, tuple(cells)))
    return slides
