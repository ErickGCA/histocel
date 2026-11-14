"""Build the training dataset: a derived, disposable product of annotations + configuration."""

from histocel.dataset.build import BuildError, BuildReport, build_dataset, parse_tile_name

__all__ = ["BuildError", "BuildReport", "build_dataset", "parse_tile_name"]
