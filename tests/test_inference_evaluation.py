import csv
import hashlib
import json

import pytest
from PIL import Image
from typer.testing import CliRunner

from histocel.cli import app
from histocel.core.tiling import cells_in_tile, tile_grid
from histocel.dataset import build_dataset
from histocel.engines import TileDetection, TrainResult, TrainSettings, register_engine
from histocel.evaluation import EvaluationError, evaluate_model
from histocel.inference import (
    InferenceError,
    load_model_card,
    open_predictor,
    predict_slide,
    render,
    to_annotation,
)
from histocel.ingest import import_project
from histocel.project import load_project
from histocel.store import load_annotations
from histocel.training import train_run

CONFIG = """
classes: [large, small]
sources: [{format: coco, path: raw/annotations/all.coco.json}]
split: {test: [s4, s5], val_fraction: 0.3}
tiling: {size: 160, overlap: 48}
"""


def _key(tile):
    return hashlib.sha256(tile.convert("RGB").tobytes()).hexdigest()


class OracleEngine:
    """An engine that already knows the answer for every tile it has been taught.

    It lets the whole pipeline be tested against exact expectations: what comes out
    of inference and evaluation can only differ from the truth through a bug in
    tiling, merging or scoring.
    """

    name = "oracle"
    answers: dict = {}

    def train(self, dataset_dir, work_dir, settings):
        work_dir.mkdir(parents=True, exist_ok=True)
        weights = work_dir / "best.pt"
        weights.write_bytes(b"oracle")
        return TrainResult(weights=weights, epochs_run=settings.epochs, engine_version="0")

    def load(self, weights, image_size, confidence, device):
        return lambda tile: OracleEngine.answers.get(_key(tile), [])


def teach(project, annotations, classes, keep=lambda cell: True, relabel=lambda label: label):
    """Fill the oracle with the true cells of every tile of every slide."""
    OracleEngine.answers = {}
    for annotation in annotations:
        with Image.open(project.images_dir / annotation.image) as image:
            image = image.convert("RGB")
            for tile in tile_grid(
                image.width, image.height, project.tiling.size, project.tiling.overlap
            ):
                kept = [c for c in annotation.cells if keep(c)]
                OracleEngine.answers[_key(image.crop(tile.rect))] = [
                    TileDetection(
                        classes.index(relabel(kept[seen.index].label)), tuple(seen.polygon), 0.9
                    )
                    for seen in cells_in_tile([c.polygon for c in kept], tile, min_visible=0.05)
                ]


@pytest.fixture(autouse=True)
def _engine():
    register_engine("oracle", OracleEngine)
    OracleEngine.answers = {}


@pytest.fixture
def trained(make_project):
    def make(config=CONFIG):
        root, _ = make_project(config, slides=5, cells=12)
        project = load_project(root)
        import_project(project)
        annotations = load_annotations(project.canonical_dir)
        build_dataset(project, annotations)
        run = train_run(project, TrainSettings(epochs=1), engine="oracle", name="t")
        card = load_model_card(run.model_dir)
        return project, annotations, card, run

    return make


# --- inference ------------------------------------------------------------------------


def test_whole_slide_inference_counts_every_cell_once(trained):
    project, annotations, card, _ = trained()
    teach(project, annotations, list(card.classes))
    predictor = open_predictor(card)
    for annotation in annotations:
        with Image.open(project.images_dir / annotation.image) as image:
            detections = predict_slide(image, predictor, card)
        counts = {}
        for detection in detections:
            counts[detection.label] = counts.get(detection.label, 0) + 1
        assert counts == annotation.counts(), annotation.slide_id


def test_predictions_become_unreviewed_model_annotations_and_a_marked_image(trained):
    project, annotations, card, _ = trained()
    teach(project, annotations, list(card.classes))
    annotation = annotations[0]
    with Image.open(project.images_dir / annotation.image) as image:
        detections = predict_slide(image, open_predictor(card), card)
        marked = render(image, detections, card.classes)
        assert marked.size == image.size
        assert marked.tobytes() != image.convert("RGB").tobytes()
        predicted = to_annotation("x", "x.png", image.size, detections, "model:abc")
    assert len(predicted.cells) == len(annotation.cells)
    assert not predicted.is_gold
    assert {cell.method for cell in predicted.cells} == {"model"}


def test_a_model_folder_must_be_complete(trained, tmp_path):
    _, _, card, run = trained()
    assert (card.tile_size, card.tile_overlap, card.classes) == (160, 48, ("large", "small"))
    with pytest.raises(InferenceError, match="not a histocel model folder"):
        load_model_card(tmp_path)
    (run.model_dir / "weights.pt").unlink()
    with pytest.raises(InferenceError, match="weights file"):
        load_model_card(run.model_dir)
    (run.model_dir / "model.json").write_text("{}", encoding="utf-8")
    with pytest.raises(InferenceError, match="damaged"):
        load_model_card(run.model_dir)


# --- evaluation -----------------------------------------------------------------------


def _value(report, name):
    return report["metrics"][name]["value"]


def test_a_perfect_model_scores_perfectly_on_the_test_slides(trained):
    project, annotations, card, _ = trained()
    teach(project, annotations, list(card.classes))
    report = evaluate_model(project, annotations, card, open_predictor(card), samples=100)

    assert report["test_slides"] == ["s4", "s5"]
    assert _value(report, "detection/f1") == pytest.approx(1.0)
    assert _value(report, "classification/accuracy") == pytest.approx(1.0)
    assert _value(report, "outline/mean_iou") > 0.98
    for name in ("large", "small"):
        assert _value(report, f"count/{name}/mean_absolute_error") == 0
    assert report["metrics"]["detection/f1"]["low"] == pytest.approx(1.0)
    assert all(s["missed"] == 0 and s["extra"] == 0 for s in report["slides"])
    assert any("Only 2 test slides" in note for note in report["notes"])


def test_a_model_blind_to_one_class_shows_it_in_the_right_places(trained):
    project, annotations, card, _ = trained()
    teach(project, annotations, list(card.classes), keep=lambda cell: cell.label != "small")
    report = evaluate_model(project, annotations, card, open_predictor(card), samples=50)

    small = sum(
        by_id.counts().get("small", 0) for by_id in annotations if by_id.slide_id in ("s4", "s5")
    )
    assert small > 0
    assert _value(report, "class/small/recall") == 0
    assert _value(report, "class/large/f1") == pytest.approx(1.0)
    assert _value(report, "detection/precision") == pytest.approx(1.0)
    assert _value(report, "detection/recall") < 1
    assert _value(report, "count/small/mean_error") == pytest.approx(-small / 2)
    assert _value(report, "count/large/mean_absolute_error") == 0


def test_a_model_that_confuses_classes_still_detects_everything(trained):
    project, annotations, card, _ = trained()
    teach(project, annotations, list(card.classes), relabel=lambda label: "large")
    report = evaluate_model(project, annotations, card, open_predictor(card), samples=50)
    assert _value(report, "detection/f1") == pytest.approx(1.0)
    assert _value(report, "classification/accuracy") < 1
    assert _value(report, "count/small/mean_error") < 0 < _value(report, "count/large/mean_error")


def test_a_model_trained_on_a_grouping_is_graded_on_that_grouping(trained):
    config = CONFIG + "groupings: {single: {cell: [large, small]}}\ndataset: {grouping: single}\n"
    project, annotations, card, _ = trained(config)
    assert card.classes == ("cell",)
    teach(project, annotations, ["cell"], relabel=lambda label: "cell")
    report = evaluate_model(project, annotations, card, open_predictor(card), samples=50)
    assert _value(report, "class/cell/f1") == pytest.approx(1.0)


def test_evaluation_refuses_what_would_make_the_score_meaningless(trained):
    project, annotations, card, _ = trained()
    predictor = open_predictor(card)
    config = project.root / "histocel.yaml"

    config.write_text(CONFIG.replace("[s4, s5]", "[]"), encoding="utf-8")
    with pytest.raises(EvaluationError, match="no test slides"):
        evaluate_model(load_project(project.root), annotations, card, predictor)

    config.write_text(CONFIG.replace("[large, small]", "[large, small, other]"), encoding="utf-8")
    with pytest.raises(EvaluationError, match="matches neither"):
        evaluate_model(load_project(project.root), annotations, card, predictor)

    config.write_text(
        CONFIG.replace("all.coco.json}", "all.coco.json, method: assisted}"), encoding="utf-8"
    )
    assisted = load_project(project.root)
    import_project(assisted)
    with pytest.raises(EvaluationError, match="never reviewed"):
        evaluate_model(assisted, load_annotations(assisted.canonical_dir), card, predictor)


# --- command line ---------------------------------------------------------------------


def test_cli_from_synthetic_slides_to_an_evaluation(tmp_path):
    runner = CliRunner()
    demo = tmp_path / "demo"
    for command in (["synth", str(demo)], ["import", "-p", str(demo)], ["build", "-p", str(demo)]):
        assert runner.invoke(app, command).exit_code == 0
    project = load_project(demo)
    annotations = load_annotations(project.canonical_dir)
    teach(project, annotations, project.taxonomy.names())

    assert runner.invoke(app, ["predict", str(project.images_dir), "-p", str(demo)]).exit_code == 1
    trained = runner.invoke(app, ["train", "-p", str(demo), "--engine", "oracle", "--name", "o"])
    assert trained.exit_code == 0, trained.output

    out = tmp_path / "out"
    predicted = runner.invoke(
        app, ["predict", str(project.images_dir), "-p", str(demo), "--out", str(out)]
    )
    assert predicted.exit_code == 0, predicted.output
    rows = list(csv.DictReader((out / "counts.csv").open(encoding="utf-8")))
    assert len(rows) == 6
    for row, annotation in zip(rows, annotations, strict=True):
        assert row["slide"] == annotation.slide_id
        assert int(row["total"]) == len(annotation.cells)
        assert int(row["small"]) == annotation.counts().get("small", 0)
    assert (out / "synthetic-001.png").is_file()
    assert len(load_annotations(out / "canonical")) == 6
    draft = json.loads((out / "predictions.coco.json").read_text(encoding="utf-8"))
    assert len(draft["images"]) == 6
    assert len(draft["annotations"]) == sum(len(a.cells) for a in annotations)

    evaluated = runner.invoke(app, ["evaluate", "-p", str(demo)])
    assert evaluated.exit_code == 0, evaluated.output
    assert "detection/f1" in evaluated.output and "1.000" in evaluated.output
    run_dir = next((demo / "runs").iterdir())
    report = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert report["metrics"]["count/large/mean_absolute_error"]["value"] == 0

    explicit = runner.invoke(
        app,
        [
            "predict",
            str(project.images_dir / "synthetic-001.png"),
            "-m",
            str(run_dir / "model"),
            "--out",
            str(tmp_path / "out2"),
        ],
    )
    assert explicit.exit_code == 0, explicit.output
    assert runner.invoke(app, ["evaluate", "-p", str(demo), "--run", "nope"]).exit_code == 1
    assert (
        runner.invoke(app, ["predict", str(tmp_path / "missing.png"), "-p", str(demo)]).exit_code
        == 1
    )


@pytest.mark.slow
def test_ultralytics_model_runs_inference_and_evaluation(make_project):
    """The real engine, end to end on CPU. Skipped unless the training extra is installed."""
    pytest.importorskip("ultralytics")
    root, _ = make_project(CONFIG, slides=5, cells=12)
    project = load_project(root)
    import_project(project)
    annotations = load_annotations(project.canonical_dir)
    build_dataset(project, annotations)
    settings = TrainSettings(epochs=1, image_size=160, batch=4, device="cpu")
    run = train_run(project, settings, name="real")
    card = load_model_card(run.model_dir)
    predictor = open_predictor(card, confidence=0.05, device="cpu")
    with Image.open(project.images_dir / annotations[0].image) as image:
        detections = predict_slide(image, predictor, card)
    assert all(d.label in card.classes and len(d.polygon) >= 3 for d in detections)
    report = evaluate_model(project, annotations, card, predictor, confidence=0.05, samples=20)
    assert set(report["metrics"]) >= {"detection/f1", "count/large/mean_absolute_error"}
