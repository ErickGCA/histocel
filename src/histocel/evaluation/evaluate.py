"""Run a model over the project's test slides and compare with their annotations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PIL import Image

from histocel import __version__
from histocel.core.annotation import SlideAnnotation
from histocel.core.metrics import bootstrap, match_slide, summarize
from histocel.engines.base import Predictor
from histocel.inference.slide import ModelCard, predict_slide
from histocel.project import Project

FEW_SLIDES = 10


class EvaluationError(ValueError):
    """Raised when the project and the model cannot be evaluated together."""


def _grouping_of(project: Project, card: ModelCard) -> str | None:
    """The class grouping the model was trained with, found from its class list."""
    candidates = [None, *project.taxonomy.groupings]
    for grouping in candidates:
        if tuple(project.taxonomy.names(grouping)) == card.classes:
            return grouping
    raise EvaluationError(
        f"the model predicts {', '.join(card.classes)}, which matches neither the classes "
        "of this project nor any of its groupings"
    )


def evaluate_model(
    project: Project,
    annotations: Sequence[SlideAnnotation],
    card: ModelCard,
    predictor: Predictor,
    iou_threshold: float = 0.5,
    confidence: float = 0.25,
    samples: int = 1000,
) -> dict[str, Any]:
    """Score ``predictor`` on the test slides of ``project``."""
    if not project.split.test:
        raise EvaluationError(
            "no test slides set aside. List them under `split.test` in histocel.yaml; "
            "slides used for training or validation cannot grade the model."
        )
    by_id = {a.slide_id: a for a in annotations}
    missing = [slide_id for slide_id in project.split.test if slide_id not in by_id]
    if missing:
        raise EvaluationError(f"test slides without annotations: {', '.join(missing)}")
    grouping = _grouping_of(project, card)

    scores, per_slide = [], []
    for slide_id in project.split.test:
        annotation = by_id[slide_id]
        machine = sum(1 for cell in annotation.cells if not cell.is_gold)
        if machine:
            raise EvaluationError(
                f"test slide {slide_id} has {machine} outline(s) made by a tool or model and "
                "never reviewed; they cannot grade a model"
            )
        with Image.open(project.images_dir / annotation.image) as image:
            detections = predict_slide(image, predictor, card)
        truth = [
            (project.taxonomy.group_of(cell.label, grouping), cell.polygon)
            for cell in annotation.cells
        ]
        score = match_slide(
            slide_id, truth, [(d.label, d.polygon) for d in detections], iou_threshold
        )
        scores.append(score)
        per_slide.append(
            {
                "slide_id": slide_id,
                "found": len(score.matches),
                "missed": len(score.missed),
                "extra": len(score.extra),
                "counts": {
                    name: {
                        "annotated": score.true_count(name),
                        "predicted": score.predicted_count(name),
                    }
                    for name in card.classes
                },
            }
        )

    values = summarize(scores, card.classes)
    intervals = bootstrap(scores, card.classes, samples=samples, seed=project.split.seed)
    notes = []
    if len(scores) < 2:
        notes.append("Only one test slide: no confidence intervals can be computed.")
    elif len(scores) < FEW_SLIDES:
        notes.append(
            f"Only {len(scores)} test slides: the confidence intervals are wide and "
            "themselves uncertain."
        )
    return {
        "histocel_version": __version__,
        "model": {"dataset_hash": card.dataset_hash, "classes": list(card.classes)},
        "settings": {
            "iou_threshold": iou_threshold,
            "confidence": confidence,
            "bootstrap_samples": samples,
            "interval": 0.95,
        },
        "test_slides": list(project.split.test),
        "metrics": {
            name: {
                "value": value,
                "low": intervals[name][0] if name in intervals else None,
                "high": intervals[name][1] if name in intervals else None,
            }
            for name, value in values.items()
        },
        "slides": per_slide,
        "notes": notes,
    }
