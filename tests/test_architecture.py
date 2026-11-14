"""The layering rules, checked by reading the imports of every module.

- ``histocel.core`` is pure: standard library, shapely and itself. No file access
  helpers, no image library, no training library, nothing else from histocel.
- Heavy optional libraries are confined to the one module that wraps them.
- No real data ships with the repository.
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "histocel"

CORE_ALLOWED_THIRD_PARTY = {"shapely"}
CONFINED = {
    "ultralytics": "engines/ultralytics.py",
    "torch": "engines/ultralytics.py",
    "segment_anything": "assist/sam.py",
}
DATA_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jfif", ".pt", ".pth", ".onnx", ".zip"}


def _imports(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def test_core_is_pure():
    for path in (PACKAGE / "core").rglob("*.py"):
        for module in _imports(path):
            top = module.split(".")[0]
            if top == "histocel":
                assert module.startswith("histocel.core"), f"{path.name} imports {module}"
            else:
                assert top in sys.stdlib_module_names or top in CORE_ALLOWED_THIRD_PARTY, (
                    f"{path.name} imports {module}; the core may only use the standard "
                    f"library and {sorted(CORE_ALLOWED_THIRD_PARTY)}"
                )


def test_core_never_opens_files():
    for path in (PACKAGE / "core").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = {
            n.func.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert "open" not in calls, f"{path.name} calls open()"
        assert not {"pathlib", "os", "io", "json"} & _imports(path), f"{path.name} does I/O"


def test_heavy_libraries_are_confined_to_their_wrapper():
    for path in PACKAGE.rglob("*.py"):
        relative = path.relative_to(PACKAGE).as_posix()
        for module in _imports(path):
            top = module.split(".")[0]
            if top in CONFINED:
                assert relative == CONFINED[top], f"{relative} imports {top}"


def test_repository_ships_no_data():
    ignored = {".git", ".venv", ".pytest_cache", ".ruff_cache", "dist", "build", "scratch"}
    offenders = [
        p.relative_to(ROOT).as_posix()
        for p in ROOT.rglob("*")
        if p.is_file()
        and p.suffix.lower() in DATA_SUFFIXES
        and not ignored & set(p.relative_to(ROOT).parts)
        and not p.relative_to(ROOT).parts[0].startswith(".venv")
        # Figures for the documentation, all generated from synthetic slides.
        and p.relative_to(ROOT).parts[:2] != ("docs", "images")
    ]
    assert offenders == [], f"data files found in the repository: {offenders}"
