# Changelog

This project follows [Semantic Versioning](https://semver.org/). Until 1.0, minor versions may break compatibility.

## Unreleased

### Added

- Pure core: polygon geometry, overlapping tile grid with geometric clipping and duplicate merging, class taxonomy with aliases and groupings, group-aware train/val/test splitting and k-fold.
- Canonical annotation format (schema version 1) with per-cell provenance, and deterministic JSON storage.
- Synthetic slide generator with exact answer keys, and the `histocel synth` command.
- Architecture tests that enforce the layering rules and that no data ships with the repository.
- CI on GitHub Actions: lint, format and tests on Python 3.10 and 3.13.
- Project folders configured by `histocel.yaml`, with `histocel init`.
- `histocel import`: COCO, YOLO-seg and QuPath GeoJSON adapters, with class aliases resolved and provenance recorded per cell.
- `histocel check`: reports split leakage, unreviewed outlines in the test set, undeclared or rare classes, missing images and tiling settings smaller than the cells.
- `histocel synth` now creates a complete project (images, COCO export, configuration).
- `histocel build`: group-aware split, overlapping tiles with geometric clipping, background tiles, YOLO segmentation export and a manifest with a content hash. Test slides never enter the dataset.
- `histocel train` and `histocel runs`: training engines behind an interface, Ultralytics YOLO segmentation as the first engine (optional `train` extra), a record per run and a self-describing model folder.
- Colab notebook that installs histocel and runs the same commands on a GPU.
