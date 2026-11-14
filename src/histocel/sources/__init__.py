"""Input adapters: one per annotation format.

An adapter reads what an annotation tool exported and returns raw slides: pixel
polygons with the class names exactly as the tool wrote them. Resolving names,
adding provenance and writing the canonical format happen later, in one place.

To support a new tool, write a function ``read(spec, images_dir) -> list[RawSlide]``
and decorate it with ``@register("name")``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from histocel.core.geometry import Point
from histocel.project import SourceSpec

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".jfif", ".webp")


class SourceError(ValueError):
    """Raised when an annotation export cannot be read."""


@dataclass(frozen=True)
class RawCell:
    label: str  # as written by the annotation tool
    polygon: tuple[Point, ...]  # slide pixel coordinates


@dataclass(frozen=True)
class RawSlide:
    slide_id: str
    image: str  # file name inside the project's image folder
    width: int
    height: int
    cells: tuple[RawCell, ...]
    notes: tuple[str, ...] = ()  # things the user should know, e.g. skipped objects


Reader = Callable[[SourceSpec, Path], list[RawSlide]]
_READERS: dict[str, Reader] = {}


def register(name: str) -> Callable[[Reader], Reader]:
    def decorator(reader: Reader) -> Reader:
        _READERS[name] = reader
        return reader

    return decorator


def formats() -> list[str]:
    _load_builtin()
    return sorted(_READERS)


def read_source(spec: SourceSpec, images_dir: Path) -> list[RawSlide]:
    """Read one annotation batch with the adapter its format names."""
    _load_builtin()
    try:
        reader = _READERS[spec.format]
    except KeyError:
        raise SourceError(
            f"unknown format {spec.format!r}. Available: {', '.join(sorted(_READERS))}"
        ) from None
    if not spec.path.exists():
        raise SourceError(f"{spec.path} does not exist")
    return reader(spec, images_dir)


def _load_builtin() -> None:
    from histocel.sources import coco, qupath, yolo_seg  # noqa: F401  (registers on import)


def slide_id_from(file_name: str) -> str:
    """Slide id for an image file name: the name without folders and image suffixes."""
    name = Path(file_name.replace("\\", "/")).name
    while name.lower().endswith(IMAGE_SUFFIXES):
        name = name[: name.rfind(".")]
    return name


def find_image(images_dir: Path, slide_id: str) -> Path:
    """The image file of a slide, whatever its extension."""
    matches = sorted(
        p
        for p in images_dir.glob(f"{slide_id}.*")
        if p.suffix.lower() in IMAGE_SUFFIXES and slide_id_from(p.name) == slide_id
    )
    if not matches:
        raise SourceError(f"no image for slide {slide_id!r} in {images_dir}")
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise SourceError(f"several images match slide {slide_id!r}: {names}")
    return matches[0]


def image_size(path: Path) -> tuple[int, int]:
    """Width and height of an image, read from its header."""
    with Image.open(path) as image:
        return image.size
