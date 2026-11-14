"""Compare predicted cells with annotated cells, slide by slide.

Detection, classification, outline quality and counting are measured separately,
because a single combined score hides which of them failed:

- detection: was each cell found, regardless of its class?
- classification: among the cells that were found, was the class right?
- outline: how well do the matched outlines overlap?
- counting: per class and per slide, how far is the predicted count from the true one?

Statistics are pooled over slides before a score is computed, so a class that is
absent from one slide does not distort the result. Uncertainty comes from
resampling whole slides, the unit that is actually independent.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from shapely import STRtree

from histocel.core.geometry import Polygon, to_shape

Labelled = tuple[str, Polygon]


@dataclass(frozen=True)
class SlideScore:
    """Outcome of matching one slide's predictions against its annotations."""

    slide_id: str
    matches: tuple[tuple[str, str, float], ...]  # (true label, predicted label, IoU)
    missed: tuple[str, ...]  # true labels of cells nobody predicted
    extra: tuple[str, ...]  # predicted labels with no cell behind them

    def true_count(self, label: str) -> int:
        return sum(t == label for t, _, _ in self.matches) + self.missed.count(label)

    def predicted_count(self, label: str) -> int:
        return sum(p == label for _, p, _ in self.matches) + self.extra.count(label)


def match_slide(
    slide_id: str,
    truth: Sequence[Labelled],
    predicted: Sequence[Labelled],
    iou_threshold: float = 0.5,
) -> SlideScore:
    """Pair predictions with annotated cells, one to one, ignoring class.

    Candidate pairs are those whose outlines overlap with IoU of at least
    ``iou_threshold``; they are accepted from the best overlap down. The threshold
    is part of the result: scores obtained with different thresholds do not compare.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    true_shapes = [to_shape(polygon) for _, polygon in truth]
    predicted_shapes = [to_shape(polygon) for _, polygon in predicted]

    candidates: list[tuple[float, int, int]] = []
    if true_shapes and predicted_shapes:
        tree = STRtree(true_shapes)
        for j, shape in enumerate(predicted_shapes):
            for i in tree.query(shape):
                i = int(i)
                intersection = shape.intersection(true_shapes[i]).area
                if intersection == 0:
                    continue
                iou = intersection / (shape.area + true_shapes[i].area - intersection)
                if iou >= iou_threshold:
                    candidates.append((iou, i, j))

    used_true: set[int] = set()
    used_predicted: set[int] = set()
    matches = []
    for iou, i, j in sorted(candidates, key=lambda c: (-c[0], c[1], c[2])):
        if i in used_true or j in used_predicted:
            continue
        used_true.add(i)
        used_predicted.add(j)
        matches.append((truth[i][0], predicted[j][0], float(iou)))
    return SlideScore(
        slide_id=slide_id,
        matches=tuple(matches),
        missed=tuple(label for i, (label, _) in enumerate(truth) if i not in used_true),
        extra=tuple(label for j, (label, _) in enumerate(predicted) if j not in used_predicted),
    )


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _f1(tp: int, fp: int, fn: int) -> float | None:
    return _ratio(2 * tp, 2 * tp + fp + fn)


def summarize(scores: Sequence[SlideScore], classes: Sequence[str]) -> dict[str, float | None]:
    """Pooled metrics over a set of slides. A metric that cannot be computed is ``None``."""
    matches = [m for score in scores for m in score.matches]
    tp = len(matches)
    fn = sum(len(score.missed) for score in scores)
    fp = sum(len(score.extra) for score in scores)
    iou_sum = sum(iou for _, _, iou in matches)

    result: dict[str, float | None] = {
        "detection/precision": _ratio(tp, tp + fp),
        "detection/recall": _ratio(tp, tp + fn),
        "detection/f1": _f1(tp, fp, fn),
        "outline/mean_iou": _ratio(iou_sum, tp),
        "classification/accuracy": _ratio(sum(t == p for t, p, _ in matches), tp),
        # Secondary, for comparison with other work: binary panoptic quality.
        "panoptic_quality": _ratio(iou_sum, tp + 0.5 * fp + 0.5 * fn),
    }
    for label in classes:
        right = sum(t == label and p == label for t, p, _ in matches)
        true_total = sum(score.true_count(label) for score in scores)
        predicted_total = sum(score.predicted_count(label) for score in scores)
        result[f"class/{label}/precision"] = _ratio(right, predicted_total)
        result[f"class/{label}/recall"] = _ratio(right, true_total)
        result[f"class/{label}/f1"] = _f1(right, predicted_total - right, true_total - right)

        pairs = [(score.true_count(label), score.predicted_count(label)) for score in scores]
        n = len(pairs)
        result[f"count/{label}/mean_absolute_error"] = _ratio(sum(abs(p - t) for t, p in pairs), n)
        result[f"count/{label}/mean_error"] = _ratio(sum(p - t for t, p in pairs), n)
        r2 = None
        if n >= 2:
            mean_true = sum(t for t, _ in pairs) / n
            total = sum((t - mean_true) ** 2 for t, _ in pairs)
            if total:
                r2 = 1 - sum((t - p) ** 2 for t, p in pairs) / total
        result[f"count/{label}/r2"] = r2
    return result


def bootstrap(
    scores: Sequence[SlideScore],
    classes: Sequence[str],
    samples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Confidence interval of every metric, from resampling slides with replacement.

    With fewer than two slides there is nothing to resample and the result is empty.
    Intervals from a handful of slides are wide and themselves unstable; that is the
    honest picture of a small test set, not a flaw of the method.
    """
    if len(scores) < 2:
        return {}
    rng = random.Random(seed)
    drawn: dict[str, list[float]] = {}
    for _ in range(samples):
        resample = [scores[rng.randrange(len(scores))] for _ in scores]
        for name, value in summarize(resample, classes).items():
            if value is not None:
                drawn.setdefault(name, []).append(value)
    tail = (1 - level) / 2
    intervals = {}
    for name, values in drawn.items():
        values.sort()
        low = values[int(tail * (len(values) - 1))]
        high = values[int(round((1 - tail) * (len(values) - 1)))]
        intervals[name] = (low, high)
    return intervals
