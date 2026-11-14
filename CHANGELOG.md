# Changelog

This project follows [Semantic Versioning](https://semver.org/). Until 1.0, minor versions may break compatibility.

## Unreleased

### Added

- Pure core: polygon geometry, overlapping tile grid with geometric clipping and duplicate merging, class taxonomy with aliases and groupings, group-aware train/val/test splitting and k-fold.
- Canonical annotation format (schema version 1) with per-cell provenance, and deterministic JSON storage.
- Synthetic slide generator with exact answer keys, and the `histocel synth` command.
- Architecture tests that enforce the layering rules and that no data ships with the repository.
- CI on GitHub Actions: lint, format and tests on Python 3.10 and 3.13.
