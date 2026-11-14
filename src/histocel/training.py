"""Run a training and leave behind a record of it and a self-describing model.

Every run gets its own folder under ``runs/``:

- ``run.json``: what was trained, on which data, with which settings and code version.
- ``model/``: the weights plus ``model.json`` with the classes and tiling the model
  expects. Inference reads everything it needs from this folder, so a model can never
  be used with the wrong class list or tile size.
- ``engine/``: whatever the training engine wrote (curves, logs).
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from histocel import __version__
from histocel.dataset.build import BuildError, load_manifest
from histocel.engines import EngineError, TrainSettings, get_engine
from histocel.project import Project

RUN_FILE = "run.json"
MODEL_FILE = "model.json"
RECORD_VERSION = 1


class TrainingError(RuntimeError):
    """Raised when a training run cannot start or does not finish."""


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    path: Path
    data: dict[str, Any]

    @property
    def model_dir(self) -> Path:
        return self.path / "model"


def _run_id(name: str | None, now: datetime) -> str:
    stamp = now.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{name}" if name else stamp


def train_run(
    project: Project,
    settings: TrainSettings,
    dataset: str = "default",
    engine: str = "ultralytics",
    name: str | None = None,
    now: datetime | None = None,
) -> RunRecord:
    """Train ``engine`` on a built dataset and record the run."""
    dataset_dir = project.root / "datasets" / dataset
    try:
        manifest = load_manifest(dataset_dir)
    except BuildError:
        raise TrainingError(
            f"dataset {dataset!r} has not been built. Run `histocel build` first."
        ) from None

    now = now or datetime.now(timezone.utc)
    run_dir = project.root / "runs" / _run_id(name, now)
    if run_dir.exists():
        raise TrainingError(f"{run_dir} already exists; choose another --name")

    try:
        trainer = get_engine(engine)
    except EngineError as error:
        raise TrainingError(str(error)) from None

    run_dir.mkdir(parents=True)
    started = time.monotonic()
    try:
        result = trainer.train(dataset_dir, run_dir / "engine", settings)
    except EngineError as error:
        raise TrainingError(f"{error} (partial files kept in {run_dir})") from None
    duration = time.monotonic() - started

    model_dir = run_dir / "model"
    model_dir.mkdir()
    weights = model_dir / f"weights{result.weights.suffix}"
    shutil.copyfile(result.weights, weights)

    model_card = {
        "record_version": RECORD_VERSION,
        "engine": engine,
        "engine_version": result.engine_version,
        "weights": weights.name,
        "classes": manifest["classes"],
        "tiling": manifest["config"]["tiling"],
        "image_size": settings.image_size,
        "trained_on": {"dataset": dataset, "dataset_hash": manifest["dataset_hash"]},
        "histocel_version": __version__,
        "created": now.isoformat(timespec="seconds"),
    }
    (model_dir / MODEL_FILE).write_text(
        json.dumps(model_card, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    record = {
        "record_version": RECORD_VERSION,
        "run_id": run_dir.name,
        "created": now.isoformat(timespec="seconds"),
        "histocel_version": __version__,
        "engine": {"name": engine, "version": result.engine_version, "model": result.model},
        "dataset": {
            "name": dataset,
            "hash": manifest["dataset_hash"],
            "classes": manifest["classes"],
            "split": manifest["split"],
            "counts": manifest["counts"],
        },
        "settings": asdict(settings),
        "epochs_run": result.epochs_run,
        "duration_seconds": round(duration, 1),
        # Scores on validation tiles, used to steer training. The honest final score
        # comes from `histocel evaluate` on whole test slides.
        "validation_tiles": result.validation,
    }
    (run_dir / RUN_FILE).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return RunRecord(run_dir.name, run_dir, record)


def list_runs(project: Project) -> list[RunRecord]:
    """Recorded runs of a project, oldest first. Folders without a record are ignored."""
    runs_dir = project.root / "runs"
    records = []
    for path in sorted(runs_dir.glob(f"*/{RUN_FILE}")) if runs_dir.is_dir() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        records.append(RunRecord(path.parent.name, path.parent, data))
    return records
