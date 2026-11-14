"""Read and write canonical annotations on disk.

File access lives here, outside ``histocel.core``, so the core stays pure.
"""

from __future__ import annotations

import json
from pathlib import Path

from histocel.core.annotation import SlideAnnotation, from_dict, to_dict


def save_annotation(annotation: SlideAnnotation, folder: Path) -> Path:
    """Write ``<folder>/<slide_id>.json``. Output is byte-stable for equal input."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{annotation.slide_id}.json"
    text = json.dumps(to_dict(annotation), ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    return path


def load_annotation(path: Path) -> SlideAnnotation:
    """Read one canonical annotation file."""
    return from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_annotations(folder: Path) -> list[SlideAnnotation]:
    """Read every canonical annotation in a folder, ordered by slide id."""
    return [load_annotation(p) for p in sorted(folder.glob("*.json"))]
