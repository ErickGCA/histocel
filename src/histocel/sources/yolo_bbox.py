"""YOLO detection labels: one ``.txt`` per image, one box per line.

Each line is ``class_index x_center y_center width height``, normalised to 0..1.
Boxes are not outlines, so a source in this format must also name an ``assist``
that turns each box into one (see :mod:`histocel.assist`).
"""

from __future__ import annotations

from pathlib import Path

from histocel.core.geometry import rect_to_polygon
from histocel.project import SourceSpec
from histocel.sources import RawCell, RawSlide, SourceError, find_image, image_size, register

VALUES_PER_BOX = 4


@register("yolo-bbox")
def read(spec: SourceSpec, images_dir: Path) -> list[RawSlide]:
    if spec.names is None:
        raise SourceError(
            f"{spec.path.name}: yolo-bbox labels only store class numbers. "
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
            if len(values) != VALUES_PER_BOX:
                raise SourceError(
                    f"{where}: a box has {VALUES_PER_BOX} values after the class, found "
                    f"{len(values)}. If these are outlines, use `format: yolo-seg`."
                )
            if not 0 <= index < len(spec.names):
                raise SourceError(
                    f"{where}: class {index} has no name; `names` lists {len(spec.names)}"
                )
            cx, cy, w, h = values
            box = (
                max((cx - w / 2) * width, 0.0),
                max((cy - h / 2) * height, 0.0),
                min((cx + w / 2) * width, float(width)),
                min((cy + h / 2) * height, float(height)),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                raise SourceError(f"{where}: the box has no area")
            cells.append(RawCell(spec.names[index], tuple(rect_to_polygon(box)), box=box))
        slides.append(RawSlide(slide_id, image.name, width, height, tuple(cells)))
    return slides
