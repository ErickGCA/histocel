"""Command line entry point. A thin shell: all logic lives in the library."""

from __future__ import annotations

from pathlib import Path

import typer

from histocel import __version__
from histocel.store import save_annotation
from histocel.synthetic import generate_slide

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Segment, classify and count cells in microscopy slide images.",
)


@app.command()
def version() -> None:
    """Print the installed histocel version."""
    typer.echo(__version__)


@app.command()
def synth(
    out: Path = typer.Argument(..., help="Folder to create the synthetic project data in."),
    slides: int = typer.Option(4, min=1, help="Number of slides to generate."),
    cells: int = typer.Option(40, min=1, help="Cells per slide."),
    width: int = typer.Option(640, min=64, help="Slide width in pixels."),
    height: int = typer.Option(480, min=64, help="Slide height in pixels."),
    seed: int = typer.Option(0, help="Same seed, same slides."),
) -> None:
    """Generate synthetic slides with exact annotations, to try histocel without real data."""
    images = out / "raw" / "images"
    canonical = out / "canonical"
    images.mkdir(parents=True, exist_ok=True)
    for index in range(slides):
        slide_id = f"synthetic-{index + 1:03d}"
        image, annotation = generate_slide(
            slide_id,
            width=width,
            height=height,
            n_cells=cells,
            seed=seed + index,
            group=f"batch-{index // 2 + 1}",
        )
        image.save(images / annotation.image)
        save_annotation(annotation, canonical)
    typer.echo(f"{slides} synthetic slides written to {out}")


if __name__ == "__main__":
    app()
