import pytest

from histocel.core import geometry as g

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_area_and_bounds():
    assert g.area(SQUARE) == 100
    assert g.bounds(SQUARE) == (0, 0, 10, 10)


def test_too_few_points_is_an_error():
    with pytest.raises(g.InvalidPolygonError):
        g.area([(0, 0), (1, 1)])


def test_zero_area_is_an_error():
    with pytest.raises(g.InvalidPolygonError):
        g.area([(0, 0), (5, 5), (10, 10)])


def test_self_intersecting_polygon_is_repaired():
    bowtie = [(0, 0), (10, 10), (10, 0), (0, 10)]
    assert g.area(bowtie) == pytest.approx(50)


def test_clip_keeps_the_true_intersection():
    piece, visible = g.clip_to_rect(SQUARE, (5, 0, 20, 20))
    assert visible == pytest.approx(0.5)
    assert g.area(piece) == pytest.approx(50)
    assert g.bounds(piece) == (5, 0, 10, 10)


def test_clip_of_a_polygon_fully_inside_changes_nothing():
    piece, visible = g.clip_to_rect(SQUARE, (-5, -5, 50, 50))
    assert visible == pytest.approx(1.0)
    assert g.area(piece) == pytest.approx(100)


def test_clip_outside_returns_none():
    assert g.clip_to_rect(SQUARE, (20, 20, 30, 30)) is None


def test_clip_touching_only_the_border_returns_none():
    assert g.clip_to_rect(SQUARE, (10, 0, 20, 10)) is None


def test_overlap_over_smaller_sees_a_truncated_copy_as_the_same_object():
    half = [(0, 0), (5, 0), (5, 10), (0, 10)]
    assert g.overlap_over_smaller(SQUARE, half) == pytest.approx(1.0)
    assert g.iou(SQUARE, half) == pytest.approx(0.5)


def test_translate_and_rect_to_polygon():
    assert g.translate([(1, 2)], 10, 20) == [(11, 22)]
    assert g.area(g.rect_to_polygon((0, 0, 4, 5))) == 20
