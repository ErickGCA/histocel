import pytest

from histocel.core.splitting import LeakageError, kfold, split_slides

SLIDES = {
    "s1": "animal-a",
    "s2": "animal-a",
    "s3": "animal-b",
    "s4": "animal-b",
    "s5": "animal-c",
    "s6": "animal-d",
    "s7": None,
    "s8": None,
}


def _group_sides(split, slides):
    sides = {}
    for slide_id, group in slides.items():
        if group is not None:
            sides.setdefault(group, set()).add(split.side_of(slide_id))
    return sides


def test_every_slide_lands_on_exactly_one_side():
    split = split_slides(SLIDES, test=["s5"], val_fraction=0.3, seed=1)
    everything = split.train + split.val + split.test
    assert sorted(everything) == sorted(SLIDES)
    assert len(set(everything)) == len(everything)


@pytest.mark.parametrize("seed", range(10))
def test_groups_never_straddle_sides(seed):
    split = split_slides(SLIDES, test=["s5"], val_fraction=0.3, seed=seed)
    for group, sides in _group_sides(split, SLIDES).items():
        assert len(sides) == 1, f"{group} is on {sides}"


def test_test_set_is_exactly_what_was_asked():
    split = split_slides(SLIDES, test=["s3", "s4", "s7"], seed=3)
    assert split.test == ("s3", "s4", "s7")


def test_same_seed_same_split_and_different_seed_can_differ():
    a = split_slides(SLIDES, test=["s5"], seed=7)
    b = split_slides(SLIDES, test=["s5"], seed=7)
    assert a == b
    assert len({split_slides(SLIDES, seed=s).val for s in range(10)}) > 1


def test_partial_group_in_test_is_refused():
    with pytest.raises(LeakageError, match="only partly in the test set"):
        split_slides(SLIDES, test=["s1"])


def test_unknown_test_slide_is_refused():
    with pytest.raises(KeyError):
        split_slides(SLIDES, test=["ghost"])


def test_validation_gets_something_and_training_is_never_empty():
    split = split_slides(SLIDES, val_fraction=0.2, seed=0)
    assert split.val and split.train
    tiny = split_slides({"only": None, "other": None}, val_fraction=0.9, seed=0)
    assert len(tiny.train) == 1 and len(tiny.val) == 1


def test_zero_val_fraction_puts_everything_in_training():
    split = split_slides(SLIDES, val_fraction=0, seed=0)
    assert split.val == () and len(split.train) == len(SLIDES)


def test_kfold_validates_each_slide_once_and_keeps_groups_together():
    folds = kfold(SLIDES, k=3, seed=0)
    assert len(folds) == 3
    validated = [s for _, val in folds for s in val]
    assert sorted(validated) == sorted(SLIDES)
    for train, val in folds:
        assert not set(train) & set(val)
        for group in {"animal-a", "animal-b"}:
            members = {s for s, g in SLIDES.items() if g == group}
            assert members <= set(train) or members <= set(val)


def test_kfold_rejects_impossible_k():
    with pytest.raises(ValueError):
        kfold(SLIDES, k=1)
    with pytest.raises(ValueError):
        kfold(SLIDES, k=99)
