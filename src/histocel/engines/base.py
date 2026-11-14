"""The contract every training engine fulfils, and the registry that finds them."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class EngineError(RuntimeError):
    """Raised when an engine is unknown, not installed, or fails to train."""


@dataclass(frozen=True)
class TrainSettings:
    """What the user asks of a training run. Engines map these to their own options."""

    model: str | None = None  # engine-specific starting point; None = the engine's default
    epochs: int = 100
    image_size: int = 640
    batch: int = 16
    patience: int = 30  # stop after this many epochs without improvement
    seed: int = 0
    device: str | None = None  # None = let the engine choose (GPU when available)


@dataclass
class TrainResult:
    weights: Path  # best weights, wherever the engine left them
    epochs_run: int
    # Scores on the validation tiles, as reported by the engine. They guide training;
    # they are not the final score, which comes from whole test slides.
    validation: dict[str, float] = field(default_factory=dict)
    engine_version: str = ""
    model: str = ""  # the starting point actually used


class Engine(Protocol):
    name: str

    def train(self, dataset_dir: Path, work_dir: Path, settings: TrainSettings) -> TrainResult:
        """Train on a dataset built by ``histocel build``; write engine files to ``work_dir``."""
        ...


_FACTORIES: dict[str, Callable[[], Engine]] = {}


def register_engine(name: str, factory: Callable[[], Engine]) -> None:
    """Make an engine available under ``name``. The factory runs only when it is used."""
    _FACTORIES[name] = factory


def available_engines() -> list[str]:
    return sorted(_FACTORIES)


def get_engine(name: str) -> Engine:
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise EngineError(
            f"unknown engine {name!r}. Available: {', '.join(available_engines())}"
        ) from None
    return factory()


def _ultralytics() -> Engine:
    # Imported here so that histocel works without the training extra installed.
    from histocel.engines.ultralytics import UltralyticsEngine

    return UltralyticsEngine()


register_engine("ultralytics", _ultralytics)
