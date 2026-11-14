import pytest
from PIL import Image

from histocel.assist import AssistError, available_assists, get_outliner, register_assist
from histocel.check import ERROR, run_checks
from histocel.core.geometry import bounds, iou
from histocel.ingest import IngestError, build_annotations, import_project
from histocel.project import SourceSpec, load_project
from histocel.sources import SourceError, read_source
from histocel.store import load_annotations

CONFIG = """
classes: [large, small]
sources:
  - format: yolo-bbox
    path: raw/annotations/boxes
    names: [large, small]
    assist: inset
"""


def _inset(model):
    """A stand-in outliner: a diamond inside each box. The third box of a slide fails."""

    def outline(image, boxes):
        assert image.mode == "RGB"
        result = []
        for index, (x0, y0, x1, y1) in enumerate(boxes):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            result.append(None if index == 2 else ((cx, y0), (x1, cy), (cx, y1), (x0, cy)))
        return result

    return outline


@pytest.fixture(autouse=True)
def _assist():
    register_assist("inset", _inset)


@pytest.fixture
def boxes_project(make_project):
    """A project whose only annotations are YOLO boxes drawn around the true cells."""

    def make(config=CONFIG, slides=3):
        root, truth = make_project(config, slides=slides, cells=8)
        folder = root / "raw" / "annotations" / "boxes"
        folder.mkdir()
        names = ["large", "small"]
        for annotation in truth:
            lines = []
            for cell in annotation.cells:
                x0, y0, x1, y1 = bounds(cell.polygon)
                w, h = annotation.width, annotation.height
                lines.append(
                    f"{names.index(cell.label)} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                    f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}"
                )
            (folder / f"{annotation.slide_id}.txt").write_text("\n".join(lines) + "\n")
        return load_project(root), truth

    return make


def test_boxes_are_read_in_pixels_with_the_box_kept(tmp_path, blank_image):
    blank_image(tmp_path / "images" / "s1.png", size=(200, 100))
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "s1.txt").write_text("1 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    spec = SourceSpec("yolo-bbox", labels, names=("a", "b"))
    (slide,) = read_source(spec, tmp_path / "images")
    (cell,) = slide.cells
    assert cell.label == "b"
    assert cell.box == pytest.approx((80, 30, 120, 70))
    assert bounds(cell.polygon) == pytest.approx((80, 30, 120, 70))


@pytest.mark.parametrize(
    ("text", "names", "message"),
    [
        ("0 0.1 0.1 0.5 0.1 0.5 0.5\n", ("a",), "use `format: yolo-seg`"),
        ("2 0.5 0.5 0.1 0.1\n", ("a",), "has no name"),
        ("0 0.5 0.5 0 0.1\n", ("a",), "no area"),
        ("0 0.5 0.5 0.1 0.1\n", None, "only store class numbers"),
    ],
)
def test_box_file_problems_point_to_the_line(tmp_path, blank_image, text, names, message):
    blank_image(tmp_path / "images" / "s1.png")
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "s1.txt").write_text(text, encoding="utf-8")
    with pytest.raises(SourceError, match=message):
        read_source(SourceSpec("yolo-bbox", labels, names=names), tmp_path / "images")


def test_boxes_become_assisted_outlines_that_remember_their_box(boxes_project):
    project, truth = boxes_project()
    annotations, notes = build_annotations(project)
    assert [a.slide_id for a in annotations] == [a.slide_id for a in truth]
    for annotation, original in zip(annotations, truth, strict=True):
        assert len(annotation.cells) == len(original.cells) - 1  # one box per slide fails
        for cell in annotation.cells:
            assert (cell.method, cell.reviewed) == ("assisted", False)
            assert not cell.is_gold
            assert len(cell.polygon) == 4
            assert bounds(cell.polygon) == pytest.approx(cell.box, abs=0.01)
    assert sum("1 box(es) could not be outlined" in note for note in notes) == 3


def test_method_is_assisted_even_if_the_source_claims_manual(boxes_project):
    project, _ = boxes_project(CONFIG + "    method: manual\n")
    annotations, _ = build_annotations(project)
    assert {c.method for a in annotations for c in a.cells} == {"assisted"}


def test_boxes_without_an_assist_or_with_an_unknown_one_are_refused(boxes_project):
    project, _ = boxes_project(CONFIG.replace("    assist: inset\n", ""))
    with pytest.raises(IngestError, match="holds boxes, not outlines"):
        build_annotations(project)
    (project.root / "histocel.yaml").write_text(
        CONFIG.replace("assist: inset", "assist: nope"), encoding="utf-8"
    )
    with pytest.raises(IngestError, match="unknown assist"):
        build_annotations(load_project(project.root))


def test_assisted_slides_cannot_be_test_slides_until_reviewed(boxes_project):
    project, _ = boxes_project(CONFIG + "split: {test: [s1]}\n")
    import_project(project)
    result = run_checks(project, load_annotations(project.canonical_dir))
    assert any("never reviewed" in f.message for f in result.findings if f.level == ERROR)

    (project.root / "histocel.yaml").write_text(
        CONFIG + "    reviewed: true\nsplit: {test: [s1]}\n", encoding="utf-8"
    )
    reviewed = load_project(project.root)
    import_project(reviewed)
    result = run_checks(reviewed, load_annotations(reviewed.canonical_dir))
    assert not any("never reviewed" in f.message for f in result.findings)


def test_assists_are_found_by_name_and_report_how_to_install(monkeypatch):
    assert {"inset", "sam"} <= set(available_assists())
    with pytest.raises(AssistError, match="unknown assist"):
        get_outliner("nope")

    import builtins

    real_import = builtins.__import__

    def no_ultralytics(name, *args, **kwargs):
        if name == "ultralytics":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ultralytics)
    with pytest.raises(AssistError, match=r"pip install \"histocel\[sam\]\""):
        get_outliner("sam")


@pytest.mark.slow
def test_sam_outlines_synthetic_cells_from_their_boxes(boxes_project):
    """The real MobileSAM, on CPU. Skipped unless the optional extra is installed."""
    pytest.importorskip("ultralytics")
    project, truth = boxes_project(slides=1)
    annotation = truth[0]
    outliner = get_outliner("sam")
    with Image.open(project.images_dir / annotation.image) as image:
        outlines = outliner(image.convert("RGB"), [bounds(c.polygon) for c in annotation.cells])
    assert len(outlines) == len(annotation.cells)
    overlaps = [
        iou(outline, cell.polygon)
        for outline, cell in zip(outlines, annotation.cells, strict=True)
        if outline is not None
    ]
    assert len(overlaps) == len(annotation.cells)
    assert sum(overlaps) / len(overlaps) > 0.8
