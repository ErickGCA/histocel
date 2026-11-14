"""Decide which slides train the model, which tune it and which grade it.

The unit of splitting is the *group* (an animal, a patient, a tissue block), never
a tile and never a single slide when slides share a group: slides from one group
look alike, and spreading them across sides inflates the score.
"""

from __future__ import annotations

import random
from collections.abc import Collection, Mapping
from dataclasses import dataclass


class LeakageError(ValueError):
    """Raised when a split would put related slides on different sides."""


@dataclass(frozen=True)
class Split:
    """Slide ids on each side. The test side is only ever read by evaluation."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def side_of(self, slide_id: str) -> str:
        for side in ("train", "val", "test"):
            if slide_id in getattr(self, side):
                return side
        raise KeyError(slide_id)


def _groups(slides: Mapping[str, str | None]) -> dict[str, list[str]]:
    """Map group -> slide ids. A slide without a group is its own group."""
    groups: dict[str, list[str]] = {}
    for slide_id in sorted(slides):
        key = slides[slide_id]
        groups.setdefault(f"group:{key}" if key is not None else f"slide:{slide_id}", []).append(
            slide_id
        )
    return groups


def split_slides(
    slides: Mapping[str, str | None],
    test: Collection[str] = (),
    val_fraction: float = 0.2,
    seed: int = 0,
) -> Split:
    """Split slides into train, validation and a frozen test set.

    ``slides`` maps slide id to its group (or ``None``). ``test`` is the explicit,
    hand-picked list of test slides; it is never sampled. The validation side is
    drawn from the remaining groups, deterministically for a given ``seed``.
    """
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    unknown = sorted(set(test) - set(slides))
    if unknown:
        raise KeyError(f"test slides not found in the project: {', '.join(unknown)}")

    groups = _groups(slides)
    test_set = set(test)
    remaining: list[list[str]] = []
    for key, members in groups.items():
        inside = [s for s in members if s in test_set]
        if inside and len(inside) != len(members):
            outside = [s for s in members if s not in test_set]
            raise LeakageError(
                f"{key} is only partly in the test set: {', '.join(inside)} are in, "
                f"{', '.join(outside)} are not. Slides of one group must stay together."
            )
        if not inside:
            remaining.append(members)

    random.Random(seed).shuffle(remaining)
    total = sum(len(members) for members in remaining)
    target = val_fraction * total
    val: list[str] = []
    train: list[str] = []
    for position, members in enumerate(remaining):
        groups_left = len(remaining) - position
        needs_more = len(val) < target
        # Never let validation swallow the last group: training needs at least one.
        if needs_more and (train or groups_left > 1):
            val.extend(members)
        else:
            train.extend(members)
    return Split(tuple(sorted(train)), tuple(sorted(val)), tuple(sorted(test_set)))


def kfold(
    slides: Mapping[str, str | None], k: int, seed: int = 0
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Group-aware k-fold: ``k`` (train, val) pairs where each group validates once."""
    groups = list(_groups(slides).values())
    if not 2 <= k <= len(groups):
        raise ValueError(f"k must be between 2 and the number of groups ({len(groups)})")
    random.Random(seed).shuffle(groups)
    folds: list[list[str]] = [[] for _ in range(k)]
    # Largest groups first, each into the currently smallest fold: balanced sizes.
    for members in sorted(groups, key=len, reverse=True):
        min(folds, key=len).extend(members)
    pairs = []
    for i, fold in enumerate(folds):
        train = [s for j, other in enumerate(folds) if j != i for s in other]
        pairs.append((tuple(sorted(train)), tuple(sorted(fold))))
    return pairs
