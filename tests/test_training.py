import json
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

from histocel.cli import app
from histocel.dataset import build_dataset
from histocel.engines import (
    EngineError,
    TrainResult,
    TrainSettings,
    available_engines,
    get_engine,
    register_engine,
)
from histocel.ingest import import_project
from histocel.project import load_project
from histocel.store import load_annotations
from histocel.training import TrainingError, list_runs, train_run

CONFIG = """
classes: [large, small]
sources: [{format: coco, path: raw/annotations/all.coco.json}]
split: {test: [s4], val_fraction: 0.3}
tiling: {size: 160, overlap: 48}
training: {epochs: 7, batch: 4}
"""
NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


class FakeEngine:
    """Stands in for a real training library: writes a weights file and reports scores."""

    name = "fake"
    calls: list = []

    def train(self, dataset_dir, work_dir, settings):
        FakeEngine.calls.append((dataset_dir, work_dir, settings))
        assert (dataset_dir / "data.yaml").is_file()
        work_dir.mkdir(parents=True, exist_ok=True)
        weights = work_dir / "best.pt"
        weights.write_bytes(b"weights")
        return TrainResult(
            weights=weights,
            epochs_run=settings.epochs,
            validation={"mask_map50": 0.5},
            engine_version="0.0",
            model=settings.model or "default-model",
        )


class BrokenEngine:
    name = "broken"

    def train(self, dataset_dir, work_dir, settings):
        raise EngineError("out of memory")


@pytest.fixture(autouse=True)
def _engines():
    FakeEngine.calls = []
    register_engine("fake", FakeEngine)
    register_engine("broken", BrokenEngine)


@pytest.fixture
def built(make_project):
    root, _ = make_project(CONFIG)
    project = load_project(root)
    import_project(project)
    report = build_dataset(project, load_annotations(project.canonical_dir))
    return project, report


def test_engines_are_found_by_name():
    assert {"fake", "ultralytics"} <= set(available_engines())
    assert get_engine("fake").name == "fake"
    with pytest.raises(EngineError, match="unknown engine"):
        get_engine("nope")


def test_a_run_records_what_was_trained_on_what(built):
    project, report = built
    run = train_run(project, TrainSettings(epochs=3), engine="fake", name="first", now=NOW)

    assert run.run_id == "20260102-030405-first"
    record = json.loads((run.path / "run.json").read_text(encoding="utf-8"))
    assert record["dataset"]["hash"] == report.dataset_hash
    assert record["dataset"]["split"]["test"] == ["s4"]
    assert record["engine"] == {"name": "fake", "version": "0.0", "model": "default-model"}
    assert record["settings"]["epochs"] == 3 and record["epochs_run"] == 3
    assert record["validation_tiles"] == {"mask_map50": 0.5}
    assert record["histocel_version"]

    dataset_dir, work_dir, _ = FakeEngine.calls[0]
    assert dataset_dir == project.root / "datasets" / "default"
    assert work_dir == run.path / "engine"


def test_the_model_folder_describes_itself(built):
    project, report = built
    run = train_run(project, TrainSettings(image_size=320), engine="fake", now=NOW)
    card = json.loads((run.model_dir / "model.json").read_text(encoding="utf-8"))
    assert (run.model_dir / card["weights"]).read_bytes() == b"weights"
    assert card["classes"] == ["large", "small"]
    assert card["tiling"] == {"size": 160, "overlap": 48}
    assert card["image_size"] == 320
    assert card["trained_on"]["dataset_hash"] == report.dataset_hash


def test_runs_are_listed_in_order_and_names_cannot_collide(built):
    project, _ = built
    assert list_runs(project) == []
    train_run(project, TrainSettings(), engine="fake", name="a", now=NOW)
    train_run(project, TrainSettings(), engine="fake", name="b", now=NOW)
    (project.root / "runs" / "not-a-run").mkdir()
    assert [r.run_id for r in list_runs(project)] == ["20260102-030405-a", "20260102-030405-b"]
    with pytest.raises(TrainingError, match="already exists"):
        train_run(project, TrainSettings(), engine="fake", name="a", now=NOW)


def test_training_needs_a_built_dataset_and_a_known_engine(built, make_project):
    project, _ = built
    with pytest.raises(TrainingError, match="histocel build"):
        train_run(project, TrainSettings(), dataset="missing", engine="fake")
    with pytest.raises(TrainingError, match="unknown engine"):
        train_run(project, TrainSettings(), engine="nope", now=NOW)
    assert list_runs(project) == []


def test_a_failed_training_leaves_no_record(built):
    project, _ = built
    with pytest.raises(TrainingError, match="out of memory"):
        train_run(project, TrainSettings(), engine="broken", now=NOW)
    assert list_runs(project) == []


def test_cli_train_uses_the_configuration_and_accepts_overrides(built):
    project, _ = built
    runner = CliRunner()
    base = ["-p", str(project.root), "--engine", "fake"]

    result = runner.invoke(app, ["train", *base, "--name", "from-config"])
    assert result.exit_code == 0, result.output
    assert FakeEngine.calls[-1][2] == TrainSettings(epochs=7, batch=4)

    result = runner.invoke(
        app, ["train", *base, "--name", "quick", "--epochs", "1", "--device", "cpu"]
    )
    assert result.exit_code == 0, result.output
    assert FakeEngine.calls[-1][2] == TrainSettings(epochs=1, batch=4, device="cpu")

    listed = runner.invoke(app, ["runs", "-p", str(project.root)])
    assert listed.exit_code == 0
    assert "from-config" in listed.output and "quick" in listed.output and "0.500" in listed.output

    assert runner.invoke(app, ["train", *base, "--dataset", "missing"]).exit_code == 1


def test_real_engine_reports_how_to_install_it_when_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_ultralytics(name, *args, **kwargs):
        if name == "ultralytics":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ultralytics)
    with pytest.raises(EngineError, match=r"pip install \"histocel\[train\]\""):
        get_engine("ultralytics")


@pytest.mark.slow
def test_ultralytics_trains_one_epoch_on_synthetic_tiles(built):
    """The real thing, on CPU. Skipped unless the training extra is installed."""
    pytest.importorskip("ultralytics")
    project, _ = built
    settings = TrainSettings(epochs=1, image_size=160, batch=4, device="cpu")
    run = train_run(project, settings, name="real", now=NOW)
    card = json.loads((run.model_dir / "model.json").read_text(encoding="utf-8"))
    assert (run.model_dir / card["weights"]).stat().st_size > 100_000
    assert run.data["epochs_run"] == 1
    assert "mask_map50" in run.data["validation_tiles"]
