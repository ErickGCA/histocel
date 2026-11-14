"""Look for problems in a project before any GPU time is spent on it.

Errors would make training wrong or impossible. Warnings are things that weaken
the result and that the user should know about.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from histocel.core.annotation import SlideAnnotation
from histocel.core.geometry import InvalidPolygonError, bounds, to_shape
from histocel.core.splitting import LeakageError, Split, split_slides
from histocel.project import Project

ERROR, WARNING, INFO = "error", "warning", "info"
FEW_EXAMPLES = 20  # below this, a class is hard to learn
FEW_TEST_EXAMPLES = 10  # below this, a class score on the test set is mostly noise
OUTSIDE_TOLERANCE = 1.0  # pixels an outline may stick out of the image


@dataclass(frozen=True)
class Finding:
    level: str
    message: str


@dataclass(frozen=True)
class CheckResult:
    findings: tuple[Finding, ...]
    split: Split | None
    counts: dict[str, dict[str, int]]  # side -> class -> cells

    @property
    def ok(self) -> bool:
        return not any(f.level == ERROR for f in self.findings)


def run_checks(project: Project, annotations: Sequence[SlideAnnotation]) -> CheckResult:
    findings: list[Finding] = []

    def add(level: str, message: str) -> None:
        findings.append(Finding(level, message))

    if not annotations:
        add(ERROR, "no canonical annotations found. Run `histocel import` first.")
        return CheckResult(tuple(findings), None, {})

    classes = project.taxonomy.names()
    by_id = {a.slide_id: a for a in annotations}
    extents: list[float] = []

    for annotation in annotations:
        if not (project.images_dir / annotation.image).is_file():
            add(ERROR, f"{annotation.slide_id}: image {annotation.image} is missing")
        invalid = outside = 0
        unknown: set[str] = set()
        for cell in annotation.cells:
            if cell.label not in classes:
                unknown.add(cell.label)
            try:
                to_shape(cell.polygon)
            except InvalidPolygonError:
                invalid += 1
                continue
            x0, y0, x1, y1 = bounds(cell.polygon)
            extents.append(max(x1 - x0, y1 - y0))
            if (
                x0 < -OUTSIDE_TOLERANCE
                or y0 < -OUTSIDE_TOLERANCE
                or x1 > annotation.width + OUTSIDE_TOLERANCE
                or y1 > annotation.height + OUTSIDE_TOLERANCE
            ):
                outside += 1
        if unknown:
            names = ", ".join(sorted(unknown))
            add(ERROR, f"{annotation.slide_id}: classes not declared in histocel.yaml: {names}")
        if invalid:
            add(ERROR, f"{annotation.slide_id}: {invalid} outline(s) with no area")
        if outside:
            add(WARNING, f"{annotation.slide_id}: {outside} outline(s) extend outside the image")
        if not annotation.cells:
            add(WARNING, f"{annotation.slide_id}: no cells annotated")

    # Groups
    grouped = [a for a in annotations if a.group is not None]
    if not grouped:
        add(
            INFO,
            "no slide has a group (animal, patient, block). Each slide is treated as "
            "independent; if several come from the same individual, declare it.",
        )
    elif len(grouped) < len(annotations):
        missing = ", ".join(a.slide_id for a in annotations if a.group is None)
        add(WARNING, f"some slides have a group and these do not: {missing}")

    # Split
    split: Split | None = None
    try:
        split = split_slides(
            {a.slide_id: a.group for a in annotations},
            test=project.split.test,
            val_fraction=project.split.val_fraction,
            seed=project.split.seed,
        )
    except (LeakageError, KeyError, ValueError) as error:
        add(ERROR, f"split: {str(error).strip(chr(34) + chr(39))}")

    counts: dict[str, dict[str, int]] = {}
    if split is not None:
        for side in ("train", "val", "test"):
            totals = dict.fromkeys(classes, 0)
            for slide_id in getattr(split, side):
                for label, n in by_id[slide_id].counts().items():
                    if label in totals:
                        totals[label] += n
            counts[side] = totals

        if not split.test:
            add(
                WARNING,
                "no test slides set aside (`split.test` is empty). Without them there is "
                "no honest final score: the validation set also picks the model.",
            )
        for slide_id in split.test:
            machine = sum(1 for cell in by_id[slide_id].cells if not cell.is_gold)
            if machine:
                add(
                    ERROR,
                    f"test slide {slide_id}: {machine} outline(s) made by a tool or model and "
                    "never reviewed. Only outlines drawn or checked by a person may grade a "
                    "model: review them, or take the slide out of `split.test`.",
                )
        if not split.train:
            add(ERROR, "no slides left for training")
        for label in classes:
            total = sum(counts[side][label] for side in counts)
            if counts["train"][label] == 0:
                add(ERROR, f"class {label!r} has no examples in the training slides")
            elif total < FEW_EXAMPLES:
                add(WARNING, f"class {label!r} has only {total} examples in the whole project")
            if split.test and counts["test"][label] < FEW_TEST_EXAMPLES:
                add(
                    WARNING,
                    f"class {label!r} has {counts['test'][label]} example(s) in the test "
                    "slides; its score will be very uncertain",
                )

    # Classes searched
    partial = [
        a.slide_id
        for a in annotations
        if a.searched_labels is not None and set(classes) - set(a.searched_labels)
    ]
    if partial:
        add(
            WARNING,
            f"{len(partial)} slide(s) were annotated for only some of the classes "
            f"({', '.join(partial[:5])}{'...' if len(partial) > 5 else ''}). On those slides "
            "a missing class means nobody looked, not that it is absent.",
        )

    # Tiling
    if extents:
        largest = max(extents)
        if project.tiling.overlap < largest:
            suggestion = int(math.ceil(largest / 8) * 8)
            add(
                WARNING,
                f"tiling.overlap is {project.tiling.overlap} px but the largest cell spans "
                f"{largest:.0f} px. Cells larger than the overlap can be cut in every tile; "
                f"use an overlap of at least {suggestion}.",
            )
        if project.tiling.size < 2 * largest:
            add(
                WARNING,
                f"tiling.size is {project.tiling.size} px, less than twice the largest cell "
                f"({largest:.0f} px). Use larger tiles.",
            )

    return CheckResult(tuple(findings), split, counts)
