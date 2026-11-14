import pytest

from histocel.core.taxonomy import Taxonomy, TaxonomyError, UnknownLabelError, normalize


def _taxonomy():
    return Taxonomy(
        classes={
            "hepatocyte": ["hepatócito normal", "Hepatocito"],
            "kupffer": ["células de kupffer"],
            "binucleated": [],
        },
        groupings={"coarse": {"hepatocyte": ["hepatocyte", "binucleated"], "kupffer": ["kupffer"]}},
    )


def test_normalize_folds_case_accents_and_spaces():
    assert normalize("  Hepatócito   NORMAL ") == "hepatocito normal"


def test_resolve_accepts_official_names_and_aliases():
    t = _taxonomy()
    assert t.resolve("hepatocyte") == "hepatocyte"
    assert t.resolve("hepatócito normal ") == "hepatocyte"
    assert t.resolve("CÉLULAS DE KUPFFER") == "kupffer"


def test_unknown_name_is_an_error_that_lists_what_exists():
    with pytest.raises(UnknownLabelError) as error:
        _taxonomy().resolve("Células Kumpfe")
    assert "hepatocyte" in str(error.value)
    assert "Kumpfe" in str(error.value)


def test_names_define_a_stable_order():
    t = _taxonomy()
    assert t.names() == ["hepatocyte", "kupffer", "binucleated"]
    assert t.names("coarse") == ["hepatocyte", "kupffer"]


def test_group_of():
    t = _taxonomy()
    assert t.group_of("binucleated") == "binucleated"
    assert t.group_of("binucleated", "coarse") == "hepatocyte"


def test_alias_claimed_twice_is_rejected():
    with pytest.raises(TaxonomyError, match="claimed by both"):
        Taxonomy(classes={"a": ["x"], "b": ["X"]})


def test_grouping_must_cover_every_class():
    with pytest.raises(TaxonomyError, match="leaves out"):
        Taxonomy(classes={"a": [], "b": []}, groupings={"g": {"only": ["a"]}})


def test_grouping_cannot_use_undeclared_or_repeated_classes():
    with pytest.raises(TaxonomyError, match="not a declared class"):
        Taxonomy(classes={"a": []}, groupings={"g": {"x": ["a", "ghost"]}})
    with pytest.raises(TaxonomyError, match="is in both"):
        Taxonomy(classes={"a": []}, groupings={"g": {"x": ["a"], "y": ["a"]}})


def test_unknown_grouping_and_empty_taxonomy():
    with pytest.raises(TaxonomyError, match="unknown grouping"):
        _taxonomy().names("nope")
    with pytest.raises(TaxonomyError):
        Taxonomy(classes={})
