import pytest

from histocel.core.annotation import (
    METHOD_ASSISTED,
    METHOD_MODEL,
    SCHEMA_VERSION,
    AnnotationError,
    Cell,
    SlideAnnotation,
    from_dict,
    to_dict,
)
from histocel.store import load_annotation, load_annotations, save_annotation

TRIANGLE = ((0.0, 0.0), (10.0, 0.0), (0.0, 10.0))


def _slide(**overrides):
    cells = (
        Cell("a", TRIANGLE, source="coco:x.json", annotator="ann"),
        Cell("b", TRIANGLE, method=METHOD_ASSISTED, box=(0.0, 0.0, 10.0, 10.0)),
    )
    values = dict(slide_id="s1", image="s1.png", width=100, height=80, cells=cells, group="g")
    return SlideAnnotation(**{**values, **overrides})


def test_round_trip_through_dict_is_lossless():
    slide = _slide(searched_labels=("a", "b"))
    assert from_dict(to_dict(slide)) == slide
    assert to_dict(slide)["schema_version"] == SCHEMA_VERSION


def test_gold_means_a_person_drew_or_checked_it():
    assert Cell("a", TRIANGLE).is_gold
    assert not Cell("a", TRIANGLE, method=METHOD_ASSISTED).is_gold
    assert Cell("a", TRIANGLE, method=METHOD_ASSISTED, reviewed=True).is_gold
    assert not Cell("a", TRIANGLE, method=METHOD_MODEL).is_gold
    assert not _slide().is_gold
    assert _slide(cells=(Cell("a", TRIANGLE),)).is_gold


def test_counts_per_label():
    assert _slide().counts() == {"a": 1, "b": 1}


def test_invalid_cells_and_slides_are_refused():
    with pytest.raises(AnnotationError, match="method"):
        Cell("a", TRIANGLE, method="guess")
    with pytest.raises(AnnotationError, match="at least 3 points"):
        Cell("a", TRIANGLE[:2])
    with pytest.raises(AnnotationError, match="dimensions"):
        _slide(width=0)


def test_other_schema_versions_and_missing_fields_are_refused():
    data = to_dict(_slide())
    with pytest.raises(AnnotationError, match="schema_version"):
        from_dict({**data, "schema_version": SCHEMA_VERSION + 1})
    del data["width"]
    with pytest.raises(AnnotationError, match="missing field"):
        from_dict(data)


def test_save_is_byte_stable_and_loads_back(tmp_path):
    slide = _slide()
    path = save_annotation(slide, tmp_path / "canonical")
    first = path.read_bytes()
    save_annotation(slide, tmp_path / "canonical")
    assert path.read_bytes() == first
    assert b"\r\n" not in first
    assert load_annotation(path) == slide
    assert load_annotations(tmp_path / "canonical") == [slide]
