"""A user's project: the folder that holds ``histocel.yaml`` and their data.

histocel finds the project the way git finds ``.git``: by walking up from the
current directory until it sees the configuration file.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from histocel.core.annotation import METHOD_MANUAL, METHODS
from histocel.core.taxonomy import Taxonomy, TaxonomyError

CONFIG_NAME = "histocel.yaml"
TOP_LEVEL_KEYS = {"classes", "groupings", "sources", "slides", "split", "tiling", "dataset"}
SOURCE_KEYS = {"format", "path", "method", "reviewed", "annotator", "group", "searched", "names"}


class ProjectError(ValueError):
    """Raised when the project folder or its configuration cannot be used."""


@dataclass(frozen=True)
class SourceSpec:
    """One batch of annotations exported from an annotation tool."""

    format: str
    path: Path
    method: str = METHOD_MANUAL
    reviewed: bool = False
    annotator: str | None = None
    group: str | None = None  # default group for the slides of this batch
    searched: tuple[str, ...] | None = None  # classes the annotator looked for
    names: tuple[str, ...] | None = None  # class names by index, for formats that need them


@dataclass(frozen=True)
class SplitConfig:
    test: tuple[str, ...] = ()
    val_fraction: float = 0.2
    seed: int = 0


@dataclass(frozen=True)
class TilingConfig:
    size: int = 320
    overlap: int = 64


@dataclass(frozen=True)
class DatasetConfig:
    grouping: str | None = None  # named grouping of classes to train on
    min_visible: float = 0.25  # a cell enters a tile when this much of it is inside
    background: float = 0.1  # share of the tiles without cells kept as background examples


@dataclass(frozen=True)
class Project:
    root: Path
    taxonomy: Taxonomy
    sources: tuple[SourceSpec, ...] = ()
    slide_groups: Mapping[str, str] = field(default_factory=dict)
    split: SplitConfig = SplitConfig()
    tiling: TilingConfig = TilingConfig()
    dataset: DatasetConfig = DatasetConfig()

    @property
    def images_dir(self) -> Path:
        return self.root / "raw" / "images"

    @property
    def canonical_dir(self) -> Path:
        return self.root / "canonical"


def find_project_root(start: Path) -> Path:
    """Closest folder at or above ``start`` that contains ``histocel.yaml``."""
    start = start.resolve()
    for folder in (start, *start.parents):
        if (folder / CONFIG_NAME).is_file():
            return folder
    raise ProjectError(
        f"no {CONFIG_NAME} found in {start} or any parent folder. "
        "Run `histocel init <folder>` to create a project."
    )


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ProjectError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def _unknown_keys(found: Mapping[str, Any], allowed: set[str], where: str) -> None:
    extra = sorted(set(found) - allowed)
    if extra:
        raise ProjectError(
            f"{where}: unknown key(s) {', '.join(extra)}. Allowed: {', '.join(sorted(allowed))}"
        )


def _classes(raw: Any) -> dict[str, list[str]]:
    if isinstance(raw, list):
        return {str(name): [] for name in raw}
    classes: dict[str, list[str]] = {}
    for name, spec in _mapping(raw, "classes").items():
        aliases = spec.get("aliases", []) if isinstance(spec, Mapping) else (spec or [])
        if not isinstance(aliases, list):
            raise ProjectError(f"classes.{name}: aliases must be a list")
        classes[str(name)] = [str(a) for a in aliases]
    return classes


def _source(raw: Any, index: int, root: Path) -> SourceSpec:
    where = f"sources[{index}]"
    data = _mapping(raw, where)
    _unknown_keys(data, SOURCE_KEYS, where)
    for required in ("format", "path"):
        if required not in data:
            raise ProjectError(f"{where}: missing `{required}`")
    method = data.get("method", METHOD_MANUAL)
    if method not in METHODS:
        raise ProjectError(f"{where}: method must be one of {', '.join(METHODS)}")
    searched, names = data.get("searched"), data.get("names")
    return SourceSpec(
        format=str(data["format"]),
        path=(root / str(data["path"])).resolve(),
        method=method,
        reviewed=bool(data.get("reviewed", False)),
        annotator=data.get("annotator"),
        group=str(data["group"]) if data.get("group") is not None else None,
        searched=tuple(map(str, searched)) if searched is not None else None,
        names=tuple(map(str, names)) if names is not None else None,
    )


def load_project(root: Path) -> Project:
    """Read and validate ``histocel.yaml`` in ``root``."""
    path = root / CONFIG_NAME
    if not path.is_file():
        raise ProjectError(f"{path} does not exist")
    try:
        data = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), CONFIG_NAME)
    except yaml.YAMLError as error:
        raise ProjectError(f"{path} is not valid YAML: {error}") from None
    _unknown_keys(data, TOP_LEVEL_KEYS, CONFIG_NAME)

    try:
        taxonomy = Taxonomy(
            classes=_classes(data.get("classes")),
            groupings={
                str(name): {str(g): [str(m) for m in members] for g, members in groups.items()}
                for name, groups in _mapping(data.get("groupings"), "groupings").items()
            },
        )
    except TaxonomyError as error:
        raise ProjectError(f"{CONFIG_NAME}: {error}") from None

    sources = data.get("sources") or []
    if not isinstance(sources, list):
        raise ProjectError("sources must be a list")

    slide_groups = {}
    for slide_id, spec in _mapping(data.get("slides"), "slides").items():
        group = _mapping(spec, f"slides.{slide_id}").get("group")
        if group is not None:
            slide_groups[str(slide_id)] = str(group)

    split = _mapping(data.get("split"), "split")
    _unknown_keys(split, {"test", "val_fraction", "seed"}, "split")
    tiling = _mapping(data.get("tiling"), "tiling")
    _unknown_keys(tiling, {"size", "overlap"}, "tiling")
    tiling_config = TilingConfig(
        size=int(tiling.get("size", TilingConfig.size)),
        overlap=int(tiling.get("overlap", TilingConfig.overlap)),
    )
    if tiling_config.size <= 0 or not 0 <= tiling_config.overlap < tiling_config.size:
        raise ProjectError("tiling: size must be positive and overlap in [0, size)")

    dataset = _mapping(data.get("dataset"), "dataset")
    _unknown_keys(dataset, {"grouping", "min_visible", "background"}, "dataset")
    dataset_config = DatasetConfig(
        grouping=str(dataset["grouping"]) if dataset.get("grouping") is not None else None,
        min_visible=float(dataset.get("min_visible", DatasetConfig.min_visible)),
        background=float(dataset.get("background", DatasetConfig.background)),
    )
    if not 0 < dataset_config.min_visible <= 1 or not 0 <= dataset_config.background <= 1:
        raise ProjectError("dataset: min_visible must be in (0, 1] and background in [0, 1]")
    if dataset_config.grouping is not None and dataset_config.grouping not in taxonomy.groupings:
        raise ProjectError(
            f"dataset.grouping: {dataset_config.grouping!r} is not declared under `groupings`"
        )

    return Project(
        root=root.resolve(),
        taxonomy=taxonomy,
        sources=tuple(_source(raw, i, root) for i, raw in enumerate(sources)),
        slide_groups=slide_groups,
        split=SplitConfig(
            test=tuple(str(s) for s in split.get("test") or ()),
            val_fraction=float(split.get("val_fraction", SplitConfig.val_fraction)),
            seed=int(split.get("seed", SplitConfig.seed)),
        ),
        tiling=tiling_config,
        dataset=dataset_config,
    )


TEMPLATE = """\
# histocel project configuration.
# Everything histocel needs to know about your data lives in this file.

# Your classes. `aliases` are other spellings found in annotation exports;
# case, accents and extra spaces are ignored when matching. A name that is
# neither a class nor an alias stops the import with an error.
classes:
  cell-type-a:
    aliases: []
  cell-type-b:
    aliases: []

# Optional: merge classes for a dataset without touching the annotations.
# groupings:
#   coarse:
#     merged-name: [cell-type-a, cell-type-b]

# Where your annotations come from. Paths are relative to this file.
# Formats: coco, yolo-seg (needs `names`), qupath (one .geojson per image).
sources: []
#  - format: coco
#    path: raw/annotations/batch1.json
#    annotator: your-name
#    group: animal-1          # default group for the slides in this batch
#  - format: yolo-seg
#    path: raw/annotations/batch2
#    names: [cell-type-a, cell-type-b]
#    method: assisted         # manual (default), assisted or model
#    reviewed: false          # true once a person has checked the outlines

# Optional: the group of each slide (animal, patient, tissue block).
# Slides of one group are never separated between training and test.
# slides:
#   slide-001: {group: animal-1}

split:
  test: []            # slide ids set aside to grade the model; never used to train
  val_fraction: 0.2
  seed: 0

tiling:
  size: 320           # tile side, in pixels
  overlap: 64         # keep this larger than your largest cell

# Optional: how the training dataset is built.
# dataset:
#   grouping: coarse    # train on a named grouping instead of the individual classes
#   min_visible: 0.25   # a cell enters a tile when at least this share of it is inside
#   background: 0.1     # share of the tiles without any cell kept as background examples
"""


def init_project(root: Path) -> Path:
    """Create a new project folder with a commented configuration file."""
    config = root / CONFIG_NAME
    if config.exists():
        raise ProjectError(f"{config} already exists; refusing to overwrite it")
    (root / "raw" / "images").mkdir(parents=True, exist_ok=True)
    (root / "raw" / "annotations").mkdir(parents=True, exist_ok=True)
    config.write_text(TEMPLATE, encoding="utf-8", newline="\n")
    return config
