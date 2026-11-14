"""Turn annotation exports into canonical annotations.

This is the one place where raw names become official classes and where every
cell receives its provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from histocel.core.annotation import Cell, SlideAnnotation
from histocel.core.geometry import InvalidPolygonError, to_shape
from histocel.core.taxonomy import UnknownLabelError
from histocel.project import Project
from histocel.sources import SourceError, read_source
from histocel.store import save_annotation


class IngestError(ValueError):
    """Raised when the annotations cannot be imported as they are."""


@dataclass
class ImportReport:
    slides: int = 0
    cells: int = 0
    notes: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)  # stale canonical files deleted


def build_annotations(project: Project) -> tuple[list[SlideAnnotation], list[str]]:
    """Read every source of the project and return canonical annotations plus notes."""
    if not project.sources:
        raise IngestError("no sources declared in histocel.yaml; nothing to import")

    slides: dict[str, SlideAnnotation] = {}
    notes: list[str] = []
    unknown: dict[str, set[str]] = {}

    for spec in project.sources:
        origin = f"{spec.format}:{spec.path.name}"
        try:
            raw_slides = read_source(spec, project.images_dir)
        except SourceError as error:
            raise IngestError(f"{origin}: {error}") from None

        if spec.searched is not None:
            try:
                searched = tuple(project.taxonomy.resolve(name) for name in spec.searched)
            except UnknownLabelError as error:
                raise IngestError(f"{origin}: `searched`: {error}") from None
        else:
            searched = None

        for raw in raw_slides:
            notes.extend(f"{raw.slide_id}: {note}" for note in raw.notes)
            cells, dropped = [], 0
            for raw_cell in raw.cells:
                try:
                    label = project.taxonomy.resolve(raw_cell.label)
                except UnknownLabelError:
                    unknown.setdefault(raw_cell.label, set()).add(origin)
                    continue
                try:
                    to_shape(raw_cell.polygon)
                except InvalidPolygonError:
                    dropped += 1
                    continue
                cells.append(
                    Cell(
                        label=label,
                        polygon=raw_cell.polygon,
                        method=spec.method,
                        reviewed=spec.reviewed,
                        source=origin,
                        annotator=spec.annotator,
                    )
                )
            if dropped:
                notes.append(f"{raw.slide_id}: {dropped} outline(s) with no area were dropped")

            group = project.slide_groups.get(raw.slide_id, spec.group)
            previous = slides.get(raw.slide_id)
            if previous is None:
                slides[raw.slide_id] = SlideAnnotation(
                    slide_id=raw.slide_id,
                    image=raw.image,
                    width=raw.width,
                    height=raw.height,
                    cells=tuple(cells),
                    group=group,
                    searched_labels=searched,
                )
                continue
            # The same slide annotated in two batches: the cells add up.
            if (previous.width, previous.height) != (raw.width, raw.height):
                raise IngestError(
                    f"slide {raw.slide_id!r} has size {previous.width}x{previous.height} in one "
                    f"source and {raw.width}x{raw.height} in {origin}"
                )
            both = {previous.searched_labels, searched}
            merged_searched = (
                None
                if None in both
                else tuple(dict.fromkeys((*previous.searched_labels, *searched)))  # type: ignore[misc]
            )
            slides[raw.slide_id] = SlideAnnotation(
                slide_id=raw.slide_id,
                image=previous.image,
                width=previous.width,
                height=previous.height,
                cells=(*previous.cells, *cells),
                group=previous.group or group,
                searched_labels=merged_searched,
            )

    if unknown:
        lines = [f"  {name!r} (in {', '.join(sorted(where))})" for name, where in unknown.items()]
        raise IngestError(
            "these class names are not declared in histocel.yaml:\n"
            + "\n".join(sorted(lines))
            + f"\nDeclared classes: {', '.join(project.taxonomy.names())}. "
            "Add each name as a new class or as an alias of an existing one."
        )
    return [slides[slide_id] for slide_id in sorted(slides)], notes


def import_project(project: Project) -> ImportReport:
    """Rebuild the project's ``canonical/`` folder from its sources."""
    annotations, notes = build_annotations(project)
    report = ImportReport(notes=notes)
    written: set[Path] = set()
    for annotation in annotations:
        written.add(save_annotation(annotation, project.canonical_dir))
        report.slides += 1
        report.cells += len(annotation.cells)
    # canonical/ is derived from the sources: files no source produces any more are stale.
    for stale in sorted(set(project.canonical_dir.glob("*.json")) - written):
        stale.unlink()
        report.removed.append(stale.name)
    return report
