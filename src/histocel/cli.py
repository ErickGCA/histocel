"""Command line entry point. A thin shell: all logic lives in the library."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from histocel import __version__
from histocel.check import ERROR, WARNING, run_checks
from histocel.ingest import IngestError, import_project
from histocel.project import (
    CONFIG_NAME,
    Project,
    ProjectError,
    find_project_root,
    init_project,
    load_project,
)
from histocel.sources.coco import to_coco
from histocel.store import load_annotations
from histocel.synthetic import default_styles, generate_slide

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Segment, classify and count cells in microscopy slide images.",
)

PROJECT_OPTION = typer.Option(
    None, "--project", "-p", help="Project folder. Default: found from the current folder."
)


def _fail(message: str) -> typer.Exit:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    return typer.Exit(code=1)


def _project(folder: Path | None) -> Project:
    try:
        return load_project(find_project_root(folder or Path.cwd()))
    except ProjectError as error:
        raise _fail(str(error)) from None


@app.command()
def version() -> None:
    """Print the installed histocel version."""
    typer.echo(__version__)


@app.command()
def init(folder: Path = typer.Argument(..., help="Folder to create the project in.")) -> None:
    """Create a project folder with a commented histocel.yaml."""
    try:
        config = init_project(folder)
    except ProjectError as error:
        raise _fail(str(error)) from None
    typer.echo(f"Project created. Next: put your images in {folder / 'raw' / 'images'},")
    typer.echo(f"declare your classes and sources in {config}, then run `histocel import`.")


@app.command(name="import")
def import_(project: Path | None = PROJECT_OPTION) -> None:
    """Convert the annotation sources of the project to the canonical format."""
    loaded = _project(project)
    try:
        report = import_project(loaded)
    except IngestError as error:
        raise _fail(str(error)) from None
    for note in report.notes:
        typer.secho(f"note: {note}", fg=typer.colors.YELLOW)
    for name in report.removed:
        typer.echo(f"removed stale {name}")
    typer.echo(f"{report.cells} cells on {report.slides} slides written to {loaded.canonical_dir}")


@app.command()
def check(project: Path | None = PROJECT_OPTION) -> None:
    """Report problems in the project before training. Exits with 1 if there are errors."""
    loaded = _project(project)
    result = run_checks(loaded, load_annotations(loaded.canonical_dir))

    if result.counts:
        classes = loaded.taxonomy.names()
        width = max(len(name) for name in (*classes, "slides"))
        typer.echo(f"{'':<{width}}  {'train':>7}{'val':>7}{'test':>7}")
        sides = [getattr(result.split, side) for side in ("train", "val", "test")]
        typer.echo(f"{'slides':<{width}}  " + "".join(f"{len(s):>7}" for s in sides))
        for name in classes:
            row = "".join(f"{result.counts[side][name]:>7}" for side in ("train", "val", "test"))
            typer.echo(f"{name:<{width}}  {row}")
        typer.echo("")

    colors = {ERROR: typer.colors.RED, WARNING: typer.colors.YELLOW}
    for finding in result.findings:
        typer.secho(f"{finding.level}: {finding.message}", fg=colors.get(finding.level))
    errors = sum(1 for f in result.findings if f.level == ERROR)
    warnings = sum(1 for f in result.findings if f.level == WARNING)
    typer.echo(f"{errors} error(s), {warnings} warning(s)")
    if not result.ok:
        raise typer.Exit(code=1)


@app.command()
def synth(
    out: Path = typer.Argument(..., help="Folder to create the synthetic project in."),
    slides: int = typer.Option(6, min=2, help="Number of slides to generate."),
    cells: int = typer.Option(40, min=1, help="Cells per slide."),
    width: int = typer.Option(640, min=64, help="Slide width in pixels."),
    height: int = typer.Option(480, min=64, help="Slide height in pixels."),
    seed: int = typer.Option(0, help="Same seed, same slides."),
) -> None:
    """Create a complete project of synthetic slides, to try histocel without real data."""
    if (out / CONFIG_NAME).exists():
        raise _fail(f"{out / CONFIG_NAME} already exists; choose another folder")
    images = out / "raw" / "images"
    annotations_dir = out / "raw" / "annotations"
    images.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    annotations = []
    for index in range(slides):
        image, annotation = generate_slide(
            f"synthetic-{index + 1:03d}",
            width=width,
            height=height,
            n_cells=cells,
            seed=seed + index,
            group=f"batch-{index // 2 + 1}",
        )
        image.save(images / annotation.image)
        annotations.append(annotation)

    export = annotations_dir / "synthetic.coco.json"
    export.write_text(json.dumps(to_coco(annotations), indent=1), encoding="utf-8", newline="\n")

    last_group = annotations[-1].group
    config = {
        "classes": {name: {"aliases": []} for name in default_styles()},
        "sources": [{"format": "coco", "path": "raw/annotations/synthetic.coco.json"}],
        "slides": {a.slide_id: {"group": a.group} for a in annotations},
        "split": {
            "test": [a.slide_id for a in annotations if a.group == last_group],
            "val_fraction": 0.2,
            "seed": 0,
        },
        "tiling": {"size": 160, "overlap": 48},
    }
    (out / CONFIG_NAME).write_text(
        "# Synthetic project generated by `histocel synth`.\n"
        + yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    typer.echo(f"{slides} synthetic slides written to {out}")
    typer.echo(f"Next: histocel import -p {out}  &&  histocel check -p {out}")


if __name__ == "__main__":
    app()
