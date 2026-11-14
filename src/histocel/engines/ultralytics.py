"""Ultralytics YOLO segmentation as a training engine.

This is the only module that imports ``ultralytics``. It is an optional extra:
``pip install "histocel[train]"``.
"""

from __future__ import annotations

from pathlib import Path

from histocel.engines.base import EngineError, TrainResult, TrainSettings

DEFAULT_MODEL = "yolo11n-seg.pt"

# Ultralytics metric names -> names that say what they are.
_METRICS = {
    "metrics/precision(M)": "mask_precision",
    "metrics/recall(M)": "mask_recall",
    "metrics/mAP50(M)": "mask_map50",
    "metrics/mAP50-95(M)": "mask_map50_95",
    "metrics/precision(B)": "box_precision",
    "metrics/recall(B)": "box_recall",
    "metrics/mAP50(B)": "box_map50",
    "metrics/mAP50-95(B)": "box_map50_95",
}


class UltralyticsEngine:
    name = "ultralytics"

    def __init__(self) -> None:
        try:
            import ultralytics
        except ImportError:
            raise EngineError(
                "the ultralytics engine is not installed. "
                'Install it with: pip install "histocel[train]"'
            ) from None
        self._ultralytics = ultralytics

    def train(self, dataset_dir: Path, work_dir: Path, settings: TrainSettings) -> TrainResult:
        model_name = settings.model or DEFAULT_MODEL
        model = self._ultralytics.YOLO(model_name)
        if getattr(model, "task", "segment") != "segment":
            raise EngineError(
                f"{model_name} is a {model.task!r} model; histocel needs a segmentation "
                f"model, such as {DEFAULT_MODEL}"
            )
        work_dir.mkdir(parents=True, exist_ok=True)
        options = {
            "data": str((dataset_dir / "data.yaml").resolve()),
            "epochs": settings.epochs,
            "imgsz": settings.image_size,
            "batch": settings.batch,
            "patience": settings.patience,
            "seed": settings.seed,
            "deterministic": True,
            "project": str(work_dir.resolve()),
            "name": "train",
            "exist_ok": True,
            "verbose": False,
        }
        if settings.device is not None:
            options["device"] = settings.device
        try:
            model.train(**options)
        except Exception as error:  # the library raises many unrelated types
            raise EngineError(f"ultralytics training failed: {error}") from error

        trainer = model.trainer
        weights = Path(trainer.best)
        if not weights.is_file():
            weights = Path(trainer.last)
        if not weights.is_file():
            raise EngineError("ultralytics finished without writing any weights")

        reported = dict(getattr(trainer, "metrics", None) or {})
        validation = {
            ours: float(reported[theirs]) for theirs, ours in _METRICS.items() if theirs in reported
        }
        return TrainResult(
            weights=weights,
            epochs_run=int(getattr(trainer, "epoch", settings.epochs - 1)) + 1,
            validation=validation,
            engine_version=str(self._ultralytics.__version__),
            model=model_name,
        )
