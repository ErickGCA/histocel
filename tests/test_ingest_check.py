import json

import pytest
from typer.testing import CliRunner

from histocel.check import ERROR, WARNING, run_checks
from histocel.cli import app
from histocel.core.annotation import METHOD_ASSISTED
from histocel.ingest import IngestError, build_annotations, import_project
from histocel.project import load_project
from histocel.store import load_annotations

BASE = """
classes: [large, small]
sources:
  - {format: coco, path: raw/annotations/all.coco.json, annotator: ann}
"""


def _messages(result, level):
    return [f.message for f in result.findings if f.level == level]


def _checked(root):
    project = load_project(root)
    import_project(project)
    return run_checks(project, load_annotations(project.canonical_dir))


# --- import ---------------------------------------------------------------------------


def test_import_reproduces_the_answer_key_and_adds_provenance(make_project):
    root, truth = make_project(BASE)
    project = load_project(root)
    report = import_project(project)
    imported = load_annotations(project.canonical_dir)

    assert (report.slides, report.cells) == (len(truth), sum(len(a.cells) for a in truth))
    assert [a.slide_id for a in imported] == [a.slide_id for a in truth]
    for got, expected in zip(imported, truth, strict=True):
        assert got.counts() == expected.counts()
        assert (got.width, got.height, got.image) == (
            expected.width,
            expected.height,
            expected.image,
        )
        for cell, original in zip(got.cells, expected.cells, strict=True):
            got_flat = [v for point in cell.polygon for v in point]
            expected_flat = [v for point in original.polygon for v in point]
            assert got_flat == pytest.approx(expected_flat, abs=0.01)
            assert (cell.method, cell.reviewed, cell.annotator) == ("manual", False, "ann")
            assert cell.source == "coco:all.coco.json"


def test_import_is_repeatable_and_removes_stale_files(make_project):
    root, _ = make_project(BASE)
    project = load_project(root)
    import_project(project)
    before = {p.name: p.read_bytes() for p in project.canonical_dir.glob("*.json")}
    (project.canonical_dir / "ghost.json").write_text("{}", encoding="utf-8")
    report = import_project(project)
    assert report.removed == ["ghost.json"]
    assert {p.name: p.read_bytes() for p in project.canonical_dir.glob("*.json")} == before


def test_unknown_class_names_are_listed_together(make_project):
    root, _ = make_project(
        "classes: [large]\nsources: [{format: coco, path: raw/annotations/all.coco.json}]\n"
    )
    with pytest.raises(IngestError) as error:
        build_annotations(load_project(root))
    assert "'small'" in str(error.value) and "alias" in str(error.value)


def test_aliases_groups_and_source_options_are_applied(make_project):
    root, _ = make_project(
        """
classes:
  big: {aliases: [LARGE]}
  tiny: {aliases: [small]}
sources:
  - format: coco
    path: raw/annotations/all.coco.json
    method: assisted
    group: batch-x
    searched: [LARGE]
slides:
  s1: {group: special}
"""
    )
    annotations, _ = build_annotations(load_project(root))
    assert set(annotations[0].counts()) <= {"big", "tiny"}
    assert annotations[0].group == "special" and annotations[1].group == "batch-x"
    assert annotations[0].searched_labels == ("big",)
    assert all(c.method == METHOD_ASSISTED for a in annotations for c in a.cells)


def test_one_slide_in_two_sources_adds_up(make_project):
    root, truth = make_project(
        BASE + "  - {format: coco, path: raw/annotations/all.coco.json, method: assisted}\n"
    )
    annotations, _ = build_annotations(load_project(root))
    assert len(annotations[0].cells) == 2 * len(truth[0].cells)


def test_import_needs_sources(make_project):
    root, _ = make_project("classes: [large, small]\n")
    with pytest.raises(IngestError, match="no sources"):
        build_annotations(load_project(root))


def test_unreadable_source_names_the_file(make_project):
    root, _ = make_project("classes: [a]\nsources: [{format: coco, path: nope.json}]\n", slides=1)
    with pytest.raises(IngestError, match=r"coco:nope\.json.*does not exist"):
        build_annotations(load_project(root))


# --- check ----------------------------------------------------------------------------


def test_a_clean_project_has_no_errors(make_project):
    root, _ = make_project(
        BASE + "split: {test: [s4]}\ntiling: {size: 160, overlap: 48}\n", cells=16
    )
    result = _checked(root)
    assert result.ok, _messages(result, ERROR)
    assert result.split.test == ("s4",)
    assert sum(result.counts["test"].values()) == 16


def test_missing_test_set_and_small_overlap_are_warnings(make_project):
    root, _ = make_project(BASE + "tiling: {size: 160, overlap: 8}\n")
    result = _checked(root)
    warnings = " ".join(_messages(result, WARNING))
    assert result.ok
    assert "no test slides" in warnings
    assert "tiling.overlap is 8 px" in warnings and "at least" in warnings


def test_machine_made_outlines_cannot_grade_a_model(make_project):
    config = """
classes: [large, small]
sources:
  - {format: coco, path: raw/annotations/all.coco.json, method: assisted, reviewed: %s}
split: {test: [s1]}
"""
    root, _ = make_project(config % "false")
    result = _checked(root)
    assert not result.ok
    assert any("never reviewed" in m for m in _messages(result, ERROR))


def test_reviewed_assisted_outlines_are_accepted_in_the_test_set(make_project):
    root, _ = make_project(
        "classes: [large, small]\nsources: [{format: coco, path: raw/annotations/all.coco.json, "
        "method: assisted, reviewed: true}]\nsplit: {test: [s1]}\n"
    )
    assert not any("never reviewed" in m for m in _messages(_checked(root), ERROR))


def test_splitting_a_group_across_test_is_an_error(make_project):
    root, _ = make_project(BASE + "slides: {s1: {group: a}, s2: {group: a}}\nsplit: {test: [s1]}\n")
    result = _checked(root)
    assert any("only partly in the test set" in m for m in _messages(result, ERROR))
    assert any("some slides have a group" in m for m in _messages(result, WARNING))


def test_missing_image_unknown_test_slide_and_partial_search(make_project):
    root, _ = make_project(
        "classes: [large, small]\nsources: [{format: coco, path: raw/annotations/all.coco.json, "
        "searched: [large]}]\nsplit: {test: [ghost]}\n"
    )
    project = load_project(root)
    import_project(project)
    (project.images_dir / "s2.png").unlink()
    result = run_checks(project, load_annotations(project.canonical_dir))
    errors = " ".join(_messages(result, ERROR))
    assert "image s2.png is missing" in errors and "ghost" in errors
    assert any("only some of the classes" in m for m in _messages(result, WARNING))


def test_check_without_import_says_what_to_do(make_project):
    root, _ = make_project(BASE)
    result = run_checks(load_project(root), [])
    assert not result.ok and "histocel import" in result.findings[0].message


# --- command line ---------------------------------------------------------------------


def test_synth_import_check_work_end_to_end(tmp_path):
    runner = CliRunner()
    demo = tmp_path / "demo"
    assert runner.invoke(app, ["synth", str(demo)]).exit_code == 0
    assert runner.invoke(app, ["synth", str(demo)]).exit_code == 1  # refuses to overwrite

    imported = runner.invoke(app, ["import", "-p", str(demo)])
    assert imported.exit_code == 0, imported.output
    assert "240 cells on 6 slides" in imported.output

    checked = runner.invoke(app, ["check", "-p", str(demo)])
    assert checked.exit_code == 0, checked.output
    assert "0 error(s)" in checked.output and "train" in checked.output


def test_cli_reports_errors_with_exit_code_1(tmp_path, make_project):
    runner = CliRunner()
    assert runner.invoke(app, ["check", "-p", str(tmp_path)]).exit_code == 1  # no project

    new = tmp_path / "fresh"
    assert runner.invoke(app, ["init", str(new)]).exit_code == 0
    assert runner.invoke(app, ["init", str(new)]).exit_code == 1
    assert runner.invoke(app, ["import", "-p", str(new)]).exit_code == 1  # no sources yet

    root, _ = make_project(BASE + "split: {test: [ghost]}\n")
    assert runner.invoke(app, ["import", "-p", str(root)]).exit_code == 0
    failed = runner.invoke(app, ["check", "-p", str(root)])
    assert failed.exit_code == 1 and "error:" in failed.output


def test_coco_export_is_valid_json_with_one_category_per_class(make_project):
    root, truth = make_project(BASE)
    data = json.loads((root / "raw/annotations/all.coco.json").read_text(encoding="utf-8"))
    assert sorted(c["name"] for c in data["categories"]) == ["large", "small"]
    assert len(data["annotations"]) == sum(len(a.cells) for a in truth)
