import json

import pytest

from histocel.project import SourceSpec
from histocel.sources import SourceError, find_image, formats, read_source, slide_id_from


def test_builtin_formats_are_registered():
    assert formats() == ["coco", "qupath", "yolo-bbox", "yolo-seg"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [("338.jpg", "338"), ("338.jpg.jpeg", "338"), ("sub/dir/a b.PNG", "a b"), ("x.v2.tif", "x.v2")],
)
def test_slide_id_from_file_name(name, expected):
    assert slide_id_from(name) == expected


def test_find_image_ignores_extension_and_reports_problems(tmp_path, blank_image):
    blank_image(tmp_path / "s1.jpg")
    assert find_image(tmp_path, "s1").name == "s1.jpg"
    with pytest.raises(SourceError, match="no image"):
        find_image(tmp_path, "s2")
    blank_image(tmp_path / "s1.png")
    with pytest.raises(SourceError, match="several images"):
        find_image(tmp_path, "s1")


def test_unknown_format_and_missing_path(tmp_path):
    with pytest.raises(SourceError, match="unknown format"):
        read_source(SourceSpec("nope", tmp_path), tmp_path)
    with pytest.raises(SourceError, match="does not exist"):
        read_source(SourceSpec("coco", tmp_path / "missing.json"), tmp_path)


# --- COCO -----------------------------------------------------------------------------


def _coco(tmp_path, annotations):
    data = {
        "images": [
            {"id": 1, "file_name": "s1.jpg.jpeg", "width": 100, "height": 80},
            {"id": 2, "file_name": "never-annotated.jpg", "width": 100, "height": 80},
        ],
        "categories": [{"id": 5, "name": "Cell A "}],
        "annotations": annotations,
    }
    path = tmp_path / "a.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return SourceSpec("coco", path)


def test_coco_reads_polygons_with_names_as_written(tmp_path):
    spec = _coco(
        tmp_path,
        [{"image_id": 1, "category_id": 5, "segmentation": [[10, 10, 30, 10, 30, 40, 10, 40]]}],
    )
    (slide,) = read_source(spec, tmp_path)
    assert (slide.slide_id, slide.image, slide.width, slide.height) == (
        "s1",
        "s1.jpg.jpeg",
        100,
        80,
    )
    assert slide.cells[0].label == "Cell A "
    assert slide.cells[0].polygon == ((10, 10), (30, 10), (30, 40), (10, 40))


def test_coco_skips_what_is_not_a_polygon_and_says_so(tmp_path):
    spec = _coco(
        tmp_path,
        [
            {"image_id": 1, "category_id": 5, "segmentation": {"counts": "rle", "size": [80, 100]}},
            {"image_id": 1, "category_id": 5, "segmentation": []},
            {"image_id": 1, "category_id": 5, "segmentation": [[1, 1, 9, 1, 9, 9]]},
        ],
    )
    (slide,) = read_source(spec, tmp_path)
    assert len(slide.cells) == 1
    assert "2 annotation(s) without a polygon" in slide.notes[0]


def test_coco_rejects_other_json(tmp_path):
    path = tmp_path / "x.json"
    path.write_text('{"hello": 1}', encoding="utf-8")
    with pytest.raises(SourceError, match="not a COCO"):
        read_source(SourceSpec("coco", path), tmp_path)


# --- YOLO-seg -------------------------------------------------------------------------


def _yolo(tmp_path, blank_image, text, names=("a", "b")):
    blank_image(tmp_path / "images" / "s1.png", size=(200, 100))
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "s1.txt").write_text(text, encoding="utf-8")
    return SourceSpec("yolo-seg", labels, names=names), tmp_path / "images"


def test_yolo_seg_scales_normalised_coordinates_to_pixels(tmp_path, blank_image):
    spec, images = _yolo(tmp_path, blank_image, "1 0.1 0.1 0.5 0.1 0.5 0.5\n\n0 0 0 1 0 1 1\n")
    (slide,) = read_source(spec, images)
    assert (slide.width, slide.height, slide.image) == (200, 100, "s1.png")
    assert [c.label for c in slide.cells] == ["b", "a"]
    assert slide.cells[0].polygon == ((20, 10), (100, 10), (100, 50))


@pytest.mark.parametrize(
    ("text", "names", "message"),
    [
        ("0 0.1 0.1 0.5 0.5\n", ("a",), "boxes, not outlines"),
        ("3 0 0 1 0 1 1\n", ("a",), "has no name"),
        ("a 0 0 1 0 1 1\n", ("a",), "expected numbers"),
        ("0 0 0 1 0 1 1\n", None, "only store class numbers"),
    ],
)
def test_yolo_seg_problems_point_to_the_line(tmp_path, blank_image, text, names, message):
    spec, images = _yolo(tmp_path, blank_image, text, names)
    with pytest.raises(SourceError, match=message):
        read_source(spec, images)


# --- QuPath GeoJSON -------------------------------------------------------------------


def _feature(ring, classification):
    properties = {"objectType": "annotation"}
    if classification is not None:
        properties["classification"] = classification
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": properties,
    }


def test_qupath_reads_feature_collections_and_drops_the_closing_point(tmp_path, blank_image):
    blank_image(tmp_path / "s1.png")
    ring = [[10, 10], [30, 10], [30, 40], [10, 10]]
    multi = {
        "type": "Feature",
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [
                [[[0, 0], [1, 0], [0, 1], [0, 0]]],
                [[[5, 5], [9, 5], [9, 9], [5, 9], [5, 5]]],
            ],
        },
        "properties": {"classification": "b"},
    }
    data = {
        "type": "FeatureCollection",
        "features": [
            _feature(ring, {"name": "a", "color": [1, 2, 3]}),
            multi,
            _feature(ring, None),
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [1, 1]},
                "properties": {},
            },
        ],
    }
    path = tmp_path / "s1.geojson"
    path.write_text(json.dumps(data), encoding="utf-8")

    for spec_path in (path, tmp_path):  # a single file or a folder of files
        (slide,) = read_source(SourceSpec("qupath", spec_path), tmp_path)
        assert [c.label for c in slide.cells] == ["a", "b"]
        assert slide.cells[0].polygon == ((10, 10), (30, 10), (30, 40))
        assert slide.cells[1].polygon == ((5, 5), (9, 5), (9, 9), (5, 9))  # largest part
        assert len(slide.notes) == 2


def test_qupath_accepts_a_bare_feature_list_and_rejects_other_json(tmp_path, blank_image):
    blank_image(tmp_path / "s1.png")
    path = tmp_path / "s1.geojson"
    path.write_text(json.dumps([_feature([[0, 0], [5, 0], [5, 5], [0, 0]], {"name": "a"})]))
    assert len(read_source(SourceSpec("qupath", path), tmp_path)[0].cells) == 1
    path.write_text('{"type": "nothing"}')
    with pytest.raises(SourceError, match="FeatureCollection"):
        read_source(SourceSpec("qupath", path), tmp_path)
