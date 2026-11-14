import pytest
from PIL import Image
from typer.testing import CliRunner

from histocel.cli import app
from histocel.core.tiling import Detection, merge_duplicates
from histocel.dataset import BuildError, build_dataset, parse_tile_name
from histocel.dataset.build import load_manifest
from histocel.ingest import import_project
from histocel.project import load_project
from histocel.store import load_annotations

CONFIG = """
classes: [large, small]
sources: [{format: coco, path: raw/annotations/all.coco.json}]
split: {test: [s4], val_fraction: 0.3}
tiling: {size: 160, overlap: 48}
"""


def _build(make_project, config=CONFIG, slides=4, cells=14):
    root, truth = make_project(config, slides=slides, cells=cells)
    project = load_project(root)
    import_project(project)
    annotations = load_annotations(project.canonical_dir)
    return project, annotations, truth, build_dataset(project, annotations)


def _labels(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        values = [float(v) for v in parts[1:]]
        rows.append((int(parts[0]), list(zip(values[0::2], values[1::2], strict=True))))
    return rows


def test_dataset_has_the_expected_layout_and_no_test_slides(make_project):
    project, _, _, report = _build(make_project)
    assert report.path == project.root / "datasets" / "default"
    assert (report.path / "data.yaml").read_text(encoding="utf-8") == (
        "train: images/train\nval: images/val\nnames:\n  0: large\n  1: small\n"
    )
    stems = {}
    for side in ("train", "val"):
        images = sorted(p.stem for p in (report.path / "images" / side).glob("*.png"))
        labels = sorted(p.stem for p in (report.path / "labels" / side).glob("*.txt"))
        assert images == labels and images
        assert len(images) == report.tiles[side]
        stems[side] = {parse_tile_name(s)[0] for s in images}
    assert not stems["train"] & stems["val"]
    assert "s4" not in stems["train"] | stems["val"]  # the test slide never reaches training
    assert report.slides == {"train": 2, "val": 1}


def test_labels_are_normalised_and_tiles_have_the_configured_size(make_project):
    _, _, _, report = _build(make_project)
    for label_file in (report.path / "labels").rglob("*.txt"):
        for class_index, points in _labels(label_file):
            assert class_index in (0, 1)
            assert len(points) >= 3
            assert all(0 <= x <= 1 and 0 <= y <= 1 for x, y in points)
    sizes = {Image.open(p).size for p in (report.path / "images").rglob("*.png")}
    assert sizes == {(160, 160)}


def test_every_cell_is_recoverable_exactly_once_from_the_tiles(make_project):
    """Read the dataset back from disk, undo the tiling and count: it must match the truth."""
    _, annotations, _, report = _build(make_project)
    by_slide = {}
    for label_file in (report.path / "labels").rglob("*.txt"):
        slide_id, x0, y0 = parse_tile_name(label_file.stem)
        for class_index, points in _labels(label_file):
            polygon = [(x * 160 + x0, y * 160 + y0) for x, y in points]
            by_slide.setdefault(slide_id, []).append(
                Detection(report.classes[class_index], polygon)
            )
    truth = {a.slide_id: a.counts() for a in annotations}
    assert by_slide
    for slide_id, detections in by_slide.items():
        counts = {}
        for detection in merge_duplicates(detections):
            counts[detection.label] = counts.get(detection.label, 0) + 1
        assert counts == truth[slide_id], slide_id


def test_same_input_gives_the_same_hash_and_a_changed_setting_does_not(make_project):
    project, annotations, _, first = _build(make_project)
    second = build_dataset(project, annotations)
    assert second.dataset_hash == first.dataset_hash
    manifest = load_manifest(first.path)
    assert manifest["dataset_hash"] == first.dataset_hash
    assert manifest["split"]["test"] == ["s4"]
    assert manifest["config"]["tiling"] == {"size": 160, "overlap": 48}
    assert len(manifest["files"]) == 2 * sum(first.tiles.values())

    config = (project.root / "histocel.yaml").read_text(encoding="utf-8")
    (project.root / "histocel.yaml").write_text(
        config.replace("overlap: 48", "overlap: 56"), encoding="utf-8"
    )
    changed = build_dataset(load_project(project.root), annotations)
    assert changed.dataset_hash != first.dataset_hash


def test_rebuild_replaces_the_previous_dataset(make_project):
    project, annotations, _, report = _build(make_project)
    stray = report.path / "images" / "train" / "edited-by-hand.png"
    stray.write_bytes(b"x")
    build_dataset(project, annotations)
    assert not stray.exists()


@pytest.mark.parametrize(("share", "expect_background"), [(0, False), (1, True)])
def test_background_share_controls_empty_tiles(make_project, share, expect_background):
    _, _, _, report = _build(make_project, CONFIG + f"dataset: {{background: {share}}}\n", cells=3)
    total = sum(report.background_tiles.values())
    assert (total > 0) is expect_background
    if expect_background:  # 320x240 with 160/48 tiling gives 3x2 = 6 tiles per slide, all kept
        assert sum(report.tiles.values()) == 6 * 3


def test_grouping_merges_classes(make_project):
    config = CONFIG + "groupings: {single: {cell: [large, small]}}\ndataset: {grouping: single}\n"
    _, annotations, _, report = _build(make_project, config)
    assert report.classes == ["cell"]
    indices = {c for f in (report.path / "labels").rglob("*.txt") for c, _ in _labels(f)}
    assert indices == {0}


def test_build_refuses_a_project_with_errors_or_without_validation(make_project):
    root, _ = make_project(CONFIG.replace("[s4]", "[ghost]"))
    project = load_project(root)
    import_project(project)
    with pytest.raises(BuildError, match="must be fixed first"):
        build_dataset(project, load_annotations(project.canonical_dir))
    (root / "histocel.yaml").write_text(
        CONFIG.replace("val_fraction: 0.3", "val_fraction: 0"), encoding="utf-8"
    )
    with pytest.raises(BuildError, match="validation"):
        build_dataset(load_project(root), load_annotations(project.canonical_dir))


def test_cli_build_after_synth_and_import(tmp_path):
    runner = CliRunner()
    demo = tmp_path / "demo"
    for command in (["synth", str(demo)], ["import", "-p", str(demo)]):
        assert runner.invoke(app, command).exit_code == 0
    built = runner.invoke(app, ["build", "-p", str(demo)])
    assert built.exit_code == 0, built.output
    assert "dataset hash" in built.output
    assert (demo / "datasets" / "default" / "manifest.json").is_file()
    assert runner.invoke(app, ["build", "-p", str(tmp_path)]).exit_code == 1
