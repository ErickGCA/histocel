"""Pure domain logic: geometry, tiling, taxonomy, splitting and the canonical annotation.

Nothing in this package touches the filesystem, the network or a training library.
Functions take plain values and return plain values, so they can be tested in
milliseconds. ``tests/test_architecture.py`` enforces this boundary.
"""
