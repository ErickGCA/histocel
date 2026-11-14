"""Synthetic slides: images drawn by code, with an exact answer key.

They let the test suite and the README example run without any real data.
"""

from histocel.synthetic.generator import CellStyle, default_styles, generate_slide

__all__ = ["CellStyle", "default_styles", "generate_slide"]
