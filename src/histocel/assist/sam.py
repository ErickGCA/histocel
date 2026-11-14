"""Box-to-outline with the Segment Anything family, through Ultralytics.

Optional extra: ``pip install "histocel[sam]"``. The default model is MobileSAM,
small enough to run on a CPU; its weights are downloaded on first use.
"""

from __future__ import annotations

from collections.abc import Sequence

from histocel.assist import AssistError, Outliner
from histocel.core.geometry import MIN_POLYGON_POINTS, Point, Rect

DEFAULT_MODEL = "mobile_sam.pt"


def sam_outliner(model: str | None = None) -> Outliner:
    try:
        from ultralytics import SAM
    except ImportError:
        raise AssistError(
            'the sam assist is not installed. Install it with: pip install "histocel[sam]"'
        ) from None
    try:
        sam = SAM(model or DEFAULT_MODEL)
    except Exception as error:  # the library raises many unrelated types
        raise AssistError(f"could not load {model or DEFAULT_MODEL}: {error}") from error

    def outline(image, boxes: Sequence[Rect]) -> list[tuple[Point, ...] | None]:
        if not boxes:
            return []
        try:
            result = sam(image, bboxes=[list(box) for box in boxes], verbose=False)[0]
        except Exception as error:
            raise AssistError(f"sam failed: {error}") from error
        masks = result.masks.xy if result.masks is not None else []
        outlines: list[tuple[Point, ...] | None] = []
        for index in range(len(boxes)):
            points = masks[index] if index < len(masks) else []
            if len(points) >= MIN_POLYGON_POINTS:
                outlines.append(tuple((float(x), float(y)) for x, y in points))
            else:
                outlines.append(None)
        return outlines

    return outline
