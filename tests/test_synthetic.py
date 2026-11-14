import numpy as np
import pytest
from typer.testing import CliRunner

from histocel.cli import app
from histocel.core.geometry import bounds, overlap_over_smaller
from histocel.store import load_annotations
from histocel.synthetic import CellStyle, generate_slide


def test_same_seed_gives_the_same_slide():
    image_a, annotation_a = generate_slide("s", seed=3)
    image_b, annotation_b = generate_slide("s", seed=3)
    assert annotation_a == annotation_b
    assert np.array_equal(np.asarray(image_a), np.asarray(image_b))
    _, other = generate_slide("s", seed=4)
    assert other != annotation_a


def test_answer_key_matches_what_was_asked():
    image, annotation = generate_slide("s", width=400, height=300, n_cells=20, seed=1)
    assert image.size == (400, 300)
    assert (annotation.width, annotation.height) == (400, 300)
    assert len(annotation.cells) == 20
    assert annotation.is_gold
    assert set(annotation.counts()) <= {"large", "small"}


def test_cells_stay_inside_the_slide_and_do_not_overlap():
    _, annotation = generate_slide("s", n_cells=40, seed=2)
    polygons = [list(c.polygon) for c in annotation.cells]
    for polygon in polygons:
        x0, y0, x1, y1 = bounds(polygon)
        assert x0 >= 0 and x1 <= annotation.width and y0 >= 0 and y1 <= annotation.height
    for i, a in enumerate(polygons):
        for b in polygons[i + 1 :]:
            assert overlap_over_smaller(a, b) == 0


def test_cells_are_actually_drawn_where_the_annotation_says():
    styles = {"dot": CellStyle(color=(10, 10, 10), radius=(8.0, 8.0))}
    image, annotation = generate_slide("s", n_cells=5, styles=styles, seed=0)
    pixels = np.asarray(image)
    for cell in annotation.cells:
        x0, y0, x1, y1 = bounds(cell.polygon)
        centre = pixels[int((y0 + y1) / 2), int((x0 + x1) / 2)]
        assert centre.max() < 60  # dark cell on a pale background


def test_impossible_requests_fail_with_a_clear_message():
    with pytest.raises(ValueError, match="could not place"):
        generate_slide("s", width=100, height=100, n_cells=500)


def test_cli_synth_writes_images_and_canonical_annotations(tmp_path):
    result = CliRunner().invoke(app, ["synth", str(tmp_path / "demo"), "--slides", "3"])
    assert result.exit_code == 0, result.output
    images = sorted((tmp_path / "demo" / "raw" / "images").glob("*.png"))
    annotations = load_annotations(tmp_path / "demo" / "canonical")
    assert len(images) == 3 and len(annotations) == 3
    assert [a.image for a in annotations] == [p.name for p in images]


def test_cli_version():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0 and result.output.strip()
