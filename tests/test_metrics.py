import pytest

from histocel.core.metrics import bootstrap, match_slide, summarize
from histocel.core.tiling import Detection, merge_duplicates


def square(x, y, size=10):
    return [(x, y), (x + size, y), (x + size, y + size), (x, y + size)]


def test_perfect_predictions_score_one_everywhere():
    slides = [
        match_slide(
            "s1",
            [("a", square(0, 0)), ("b", square(50, 0))],
            [("a", square(0, 0)), ("b", square(50, 0))],
        ),
        match_slide("s2", [("a", square(0, 0))], [("a", square(0, 0))]),
    ]
    result = summarize(slides, ["a", "b"])
    for name in ("detection/precision", "detection/recall", "detection/f1", "outline/mean_iou"):
        assert result[name] == pytest.approx(1.0), name
    assert result["classification/accuracy"] == 1.0
    assert result["panoptic_quality"] == pytest.approx(1.0)
    assert result["class/a/f1"] == 1.0 and result["class/b/f1"] == 1.0
    assert result["count/a/mean_absolute_error"] == 0 and result["count/b/mean_absolute_error"] == 0
    assert result["count/b/r2"] == pytest.approx(1.0)  # b: 1 then 0 cells, predicted exactly
    assert result["count/a/r2"] is None  # same count on every slide: R2 is undefined


def test_detection_classification_and_counting_are_told_apart():
    truth = [("a", square(0, 0)), ("a", square(50, 0)), ("b", square(100, 0))]
    predicted = [
        ("a", square(1, 0)),  # found, right class
        ("b", square(50, 0)),  # found, wrong class
        ("a", square(300, 300)),  # nothing there
    ]  # and the cell at x=100 is missed
    score = match_slide("s", truth, predicted)
    assert (len(score.matches), score.missed, score.extra) == (2, ("b",), ("a",))

    result = summarize([score], ["a", "b"])
    assert result["detection/precision"] == pytest.approx(2 / 3)
    assert result["detection/recall"] == pytest.approx(2 / 3)
    assert result["classification/accuracy"] == pytest.approx(0.5)
    assert result["class/a/precision"] == pytest.approx(0.5)
    assert result["class/a/recall"] == pytest.approx(0.5)
    assert result["class/b/f1"] == 0
    # The counts happen to be right although half the cells are wrong: this is exactly
    # why counting alone is not enough, and why it is reported next to detection.
    assert result["count/a/mean_absolute_error"] == 0
    assert result["count/b/mean_absolute_error"] == 0
    assert 0 < result["outline/mean_iou"] < 1


def test_the_iou_threshold_decides_what_counts_as_found():
    truth, predicted = [("a", square(0, 0))], [("a", square(5, 0))]  # IoU = 1/3
    assert match_slide("s", truth, predicted, iou_threshold=0.5).matches == ()
    assert len(match_slide("s", truth, predicted, iou_threshold=0.3).matches) == 1
    with pytest.raises(ValueError):
        match_slide("s", truth, predicted, iou_threshold=0)


def test_matching_is_one_to_one_and_prefers_the_best_overlap():
    truth = [("a", square(0, 0))]
    predicted = [("a", square(2, 0)), ("a", square(0, 0))]
    score = match_slide("s", truth, predicted)
    assert score.matches == (("a", "a", 1.0),)
    assert score.extra == ("a",)


def test_nothing_to_compare_gives_undefined_scores_not_errors():
    result = summarize([match_slide("s", [], [])], ["a"])
    assert result["detection/f1"] is None and result["outline/mean_iou"] is None
    assert result["count/a/mean_absolute_error"] == 0
    assert summarize([], ["a"])["count/a/mean_absolute_error"] is None


def test_counting_error_and_bias():
    slides = [
        match_slide("s1", [("a", square(0, 0)), ("a", square(50, 0))], [("a", square(0, 0))]),
        match_slide("s2", [("a", square(0, 0))], [("a", square(0, 0)), ("a", square(90, 90))]),
        match_slide("s3", [("a", square(0, 0))] * 1, []),
    ]
    result = summarize(slides, ["a"])
    assert result["count/a/mean_absolute_error"] == pytest.approx(1.0)  # |1-2|, |2-1|, |0-1|
    assert result["count/a/mean_error"] == pytest.approx(-1 / 3)  # undercounts on average
    assert result["count/a/r2"] < 0  # worse than always guessing the mean


def test_bootstrap_is_repeatable_and_brackets_the_value():
    slides = [
        match_slide(
            f"s{i}",
            [("a", square(0, 0)), ("a", square(50, 0))],
            [("a", square(0, 0))] * (i % 2 + 1),
        )
        for i in range(8)
    ]
    value = summarize(slides, ["a"])["detection/f1"]
    first = bootstrap(slides, ["a"], samples=200, seed=1)
    assert first == bootstrap(slides, ["a"], samples=200, seed=1)
    low, high = first["detection/f1"]
    assert low <= value <= high and low < high
    assert bootstrap(slides[:1], ["a"]) == {}


def test_merge_prefers_an_uncut_outline_over_a_more_confident_cut_one():
    cut = Detection("a", square(0, 0, 6), score=0.95, whole=False)
    whole = Detection("a", square(0, 0, 10), score=0.60, whole=True)
    assert merge_duplicates([cut, whole]) == [whole]
