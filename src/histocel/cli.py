"""Command line entry point. A thin shell: all logic lives in the library."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import typer
import yaml

from histocel import __version__
from histocel.check import ERROR, WARNING, run_checks
from histocel.dataset import BuildError, build_dataset
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
from histocel.training import TrainingError, list_runs, train_run

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
def build(
    project: Path | None = PROJECT_OPTION,
    name: str = typer.Option("default", help="Name of the dataset folder under datasets/."),
) -> None:
    """Split the slides, cut them into tiles and write the training dataset."""
    loaded = _project(project)
    try:
        report = build_dataset(loaded, load_annotations(loaded.canonical_dir), name=name)
    except BuildError as error:
        raise _fail(str(error)) from None
    typer.echo(f"classes: {', '.join(report.classes)}")
    typer.echo(f"{'':<6}{'slides':>8}{'tiles':>8}{'empty':>8}{'cells':>8}")
    for side in ("train", "val"):
        typer.echo(
            f"{side:<6}{report.slides[side]:>8}{report.tiles[side]:>8}"
            f"{report.background_tiles[side]:>8}{report.cells[side]:>8}"
        )
    typer.echo(f"dataset hash: {report.dataset_hash[:16]}")
    typer.echo(f"written to {report.path}")


@app.command()
def train(
    project: Path | None = PROJECT_OPTION,
    dataset: str = typer.Option("default", help="Dataset to train on, as built by `build`."),
    name: str | None = typer.Option(None, help="Label added to the run folder name."),
    engine: str | None = typer.Option(None, help="Training engine. Default: from histocel.yaml."),
    model: str | None = typer.Option(None, help="Starting model, engine-specific."),
    epochs: int | None = typer.Option(None, min=1),
    image_size: int | None = typer.Option(None, min=32, help="Tiles are resized to this."),
    batch: int | None = typer.Option(None, min=1),
    device: str | None = typer.Option(None, help="For example: cpu, 0. Default: automatic."),
) -> None:
    """Train a model on a built dataset and record the run under runs/."""
    loaded = _project(project)
    overrides = {
        "model": model,
        "epochs": epochs,
        "image_size": image_size,
        "batch": batch,
        "device": device,
    }
    settings = replace(
        loaded.training.settings, **{k: v for k, v in overrides.items() if v is not None}
    )
    try:
        run = train_run(
            loaded, settings, dataset=dataset, engine=engine or loaded.training.engine, name=name
        )
    except TrainingError as error:
        raise _fail(str(error)) from None
    typer.echo(f"run {run.run_id}: {run.data['epochs_run']} epoch(s)")
    for metric, value in run.data["validation_tiles"].items():
        typer.echo(f"  {metric:<16}{value:.3f}  (validation tiles)")
    typer.echo(f"model written to {run.model_dir}")


@app.command()
def runs(project: Path | None = PROJECT_OPTION) -> None:
    """List the recorded training runs of the project."""
    records = list_runs(_project(project))
    if not records:
        typer.echo("no runs yet. Train one with `histocel train`.")
        return
    width = max(len(r.run_id) for r in records)
    typer.echo(f"{'run':<{width}}  {'dataset':<10}{'epochs':>7}{'mask mAP50':>12}")
    for record in records:
        score = record.data.get("validation_tiles", {}).get("mask_map50")
        typer.echo(
            f"{record.run_id:<{width}}  {record.data['dataset']['hash'][:8]:<10}"
            f"{record.data['epochs_run']:>7}{(f'{score:.3f}' if score is not None else '-'):>12}"
        )


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
