import pytest

from histocel.project import (
    CONFIG_NAME,
    ProjectError,
    find_project_root,
    init_project,
    load_project,
)


def _write(tmp_path, text):
    (tmp_path / CONFIG_NAME).write_text(text, encoding="utf-8")
    return tmp_path


def test_init_creates_a_loadable_project_and_refuses_to_overwrite(tmp_path):
    root = tmp_path / "new"
    init_project(root)
    project = load_project(root)
    assert project.taxonomy.names() == ["cell-type-a", "cell-type-b"]
    assert project.sources == ()
    assert (root / "raw" / "images").is_dir()
    with pytest.raises(ProjectError, match="already exists"):
        init_project(root)


def test_project_is_found_from_a_subfolder(tmp_path):
    root = _write(tmp_path, "classes: [a]\n")
    deep = root / "raw" / "images"
    deep.mkdir(parents=True)
    assert find_project_root(deep) == root.resolve()


def test_missing_project_explains_what_to_do(tmp_path):
    with pytest.raises(ProjectError, match="histocel init"):
        find_project_root(tmp_path)


def test_full_configuration_is_parsed(tmp_path):
    root = _write(
        tmp_path,
        """
classes:
  hepatocyte: {aliases: ["Hepatócito normal"]}
  kupffer:
  binucleated: []
groupings:
  coarse: {hepatocyte: [hepatocyte, binucleated], kupffer: [kupffer]}
sources:
  - {format: coco, path: raw/annotations/a.json, annotator: ann, group: g1}
  - format: yolo-seg
    path: raw/annotations/b
    names: [hepatocyte, kupffer]
    method: assisted
    searched: [hepatocyte, kupffer]
slides:
  s1: {group: animal-1}
split: {test: [s1], val_fraction: 0.25, seed: 7}
tiling: {size: 256, overlap: 32}
""",
    )
    project = load_project(root)
    assert project.taxonomy.resolve("hepatocito NORMAL") == "hepatocyte"
    assert project.taxonomy.names("coarse") == ["hepatocyte", "kupffer"]
    first, second = project.sources
    assert first.path == (root / "raw/annotations/a.json").resolve()
    assert (first.annotator, first.group, first.method) == ("ann", "g1", "manual")
    assert (second.method, second.names) == ("assisted", ("hepatocyte", "kupffer"))
    assert project.slide_groups == {"s1": "animal-1"}
    assert (project.split.test, project.split.val_fraction, project.split.seed) == (
        ("s1",),
        0.25,
        7,
    )
    assert (project.tiling.size, project.tiling.overlap) == (256, 32)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("classes: [a]\nclases: [b]\n", "unknown key"),
        ("classes: [a]\nsources: [{format: coco}]\n", "missing `path`"),
        ("classes: [a]\nsources: [{format: coco, path: x, mehtod: manual}]\n", "unknown key"),
        ("classes: [a]\nsources: [{format: coco, path: x, method: guess}]\n", "method must be"),
        ("classes: [a]\ntiling: {size: 100, overlap: 100}\n", "overlap"),
        ("classes: {a: {aliases: [x]}, b: {aliases: [X]}}\n", "claimed by both"),
        ("sources: []\n", "at least one class"),
        ("classes: [a\n", "not valid YAML"),
    ],
)
def test_mistakes_in_the_configuration_are_reported(tmp_path, text, message):
    with pytest.raises(ProjectError, match=message):
        load_project(_write(tmp_path, text))
