import pytest

from histocel.core.geometry import bounds
from histocel.core.tiling import (
    Detection,
    Tile,
    cells_in_tile,
    merge_duplicates,
    tile_grid,
    to_slide_coords,
)
from histocel.synthetic import generate_slide


def _covered(tiles, width, height):
    seen = [[False] * width for _ in range(height)]
    for t in tiles:
        for y in range(t.y0, t.y1):
            for x in range(t.x0, t.x1):
                seen[y][x] = True
    return all(all(row) for row in seen)


@pytest.mark.parametrize(
    ("width", "height", "size", "overlap"),
    [(960, 1280, 320, 0), (960, 1280, 320, 64), (1000, 700, 320, 64), (333, 517, 128, 32)],
)
def test_grid_covers_every_pixel_and_stays_inside(width, height, size, overlap):
    tiles = tile_grid(width, height, size, overlap)
    assert _covered(tiles, width, height)
    for t in tiles:
        assert 0 <= t.x0 < t.x1 <= width and 0 <= t.y0 < t.y1 <= height
        assert t.width == size and t.height == size


def test_grid_without_overlap_on_exact_multiples_matches_a_plain_grid():
    tiles = tile_grid(960, 1280, 320)
    assert len(tiles) == 3 * 4
    assert len(set(tiles)) == 12


def test_slide_smaller_than_the_tile_gives_one_small_tile():
    assert tile_grid(200, 100, 320, 64) == [Tile(0, 0, 200, 100)]


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (320, 320), (320, -1)])
def test_invalid_grid_parameters(size, overlap):
    with pytest.raises(ValueError):
        tile_grid(960, 1280, size, overlap)


def test_cell_across_a_border_is_cut_not_squashed():
    cell = [(300, 100), (340, 100), (340, 140), (300, 140)]  # straddles x = 320
    left, right = Tile(0, 0, 320, 320), Tile(320, 0, 640, 320)
    in_left = cells_in_tile([cell], left)
    in_right = cells_in_tile([cell], right)
    assert in_left[0].visible == pytest.approx(0.5)
    assert in_right[0].visible == pytest.approx(0.5)
    assert bounds(in_left[0].polygon) == (300, 100, 320, 140)
    assert bounds(in_right[0].polygon) == (0, 100, 20, 140)  # tile coordinates


def test_slivers_are_dropped():
    cell = [(310, 100), (350, 100), (350, 140), (310, 140)]  # 25% left, 75% right
    left = Tile(0, 0, 320, 320)
    assert cells_in_tile([cell], left, min_visible=0.3) == []
    assert len(cells_in_tile([cell], left, min_visible=0.25)) == 1


def test_to_slide_coords_undoes_the_tile_offset():
    tile = Tile(320, 640, 640, 960)
    assert to_slide_coords([(1, 2)], tile) == [(321, 642)]


def test_merge_keeps_the_whole_copy_over_the_truncated_one():
    whole = Detection("a", [(0, 0), (10, 0), (10, 10), (0, 10)], score=0.9)
    half = Detection("a", [(0, 0), (5, 0), (5, 10), (0, 10)], score=0.9)
    far = Detection("b", [(50, 50), (60, 50), (60, 60), (50, 60)], score=0.5)
    merged = merge_duplicates([half, far, whole])
    assert merged == [whole, far]


def test_merge_prefers_the_more_confident_detection():
    a = Detection("a", [(0, 0), (10, 0), (10, 10), (0, 10)], score=0.4)
    b = Detection("b", [(1, 1), (11, 1), (11, 11), (1, 11)], score=0.8)
    assert merge_duplicates([a, b]) == [b]


def test_touching_neighbours_are_not_merged():
    a = Detection("a", [(0, 0), (10, 0), (10, 10), (0, 10)])
    b = Detection("a", [(10, 0), (20, 0), (20, 10), (10, 10)])
    assert len(merge_duplicates([a, b])) == 2


@pytest.mark.parametrize("seed", range(5))
def test_perfect_detector_counts_every_cell_exactly_once(seed):
    """End to end: slice a slide, 'detect' every visible piece, merge, count.

    With an overlap larger than the largest cell, the count must equal the truth.
    This is the property that plain grid tiling without merging does not have.
    """
    _, annotation = generate_slide("s", width=640, height=480, n_cells=45, seed=seed)
    polygons = [list(c.polygon) for c in annotation.cells]
    tiles = tile_grid(annotation.width, annotation.height, size=160, overlap=48)

    detections = []
    for tile in tiles:
        for seen in cells_in_tile(polygons, tile, min_visible=0.05):
            label = annotation.cells[seen.index].label
            detections.append(Detection(label, to_slide_coords(seen.polygon, tile)))

    assert len(detections) > len(polygons)  # borders really do produce duplicates
    merged = merge_duplicates(detections)
    assert len(merged) == len(polygons)

    counts = {}
    for d in merged:
        counts[d.label] = counts.get(d.label, 0) + 1
    assert counts == annotation.counts()
