# histocel

A command line tool to train and use models that **outline, classify and count cells** in microscopy images, with your own data.

histocel ships no images, no classes and no pretrained models. You point it at your slides and annotations, declare your classes, and run.

> **Status: pre-alpha.** The core (geometry, tiling, taxonomy, data splitting) and the synthetic slide generator are done and tested. Import, training, inference and evaluation are under construction. See the [roadmap](#roadmap).

## Why it exists

Computer vision projects in histology tend to trip in the same places. histocel turns each one into a rule of the tool:

- **Split by slide and by group, never by tile.** Slides from the same animal or patient stay on the same side.
- **A frozen test set** that never picks the model and is only read by evaluation.
- **AI-made ground truth does not grade AI.** Every outline records how it was made; the test set only accepts outlines drawn or reviewed by a person.
- **Tile with overlap and merge on the way back**, so a cell on the border between two tiles is counted once.
- **Report detection, classification, outline quality and counting error separately**, each with a confidence interval.
- **Everything is rebuilt by a command.** The training dataset comes from data plus configuration, identically every time.

## Try it without any data

```bash
git clone https://github.com/ErickGCA/histocel
cd histocel
pip install -e .

histocel synth demo --slides 4
```

This creates four synthetic slides in `demo/` (coloured ellipses on a pinkish background) with the exact answer key for every cell in histocel's format. They look nothing like real tissue: they exist to show the tool working and to test the code against a known answer.

## How your data comes in

The histocel repository holds code only. Your data lives in a **project folder** of your own:

```
my-project/
├─ histocel.yaml      # classes, annotation sources, split, tiling
├─ raw/
│  ├─ images/         # your slides, any size
│  └─ annotations/    # exports from your annotation tool
├─ canonical/         # created by histocel: one JSON per slide
├─ datasets/          # created by histocel: disposable
└─ runs/              # created by histocel: models, metrics, records
```

The internal annotation format is described in [docs/canonical-format.md](docs/canonical-format.md).

## Roadmap

| Stage | Delivers | Status |
|---|---|---|
| 1 | Pure core, synthetic slides, tests and CI | done |
| 2 | `init`, `import` (COCO, YOLO-seg) and `check` | to do |
| 3 | `build`: split by slide and group, tiling, manifest | to do |
| 4 | `train`: training engine, run records, Colab notebook | to do |
| 5 | Whole-slide `predict` and `evaluate` with counting error | to do |
| 6 | Box input with assisted outlines (SAM) | to do |
| 7 | User guide and version 0.1 | to do |

Design decisions and their reasons are in [docs/architecture.md](docs/architecture.md).

## Development

```bash
uv sync            # creates the environment with locked versions
uv run pytest      # tests
uv run ruff check . && uv run ruff format --check .
```

## Limits

histocel is a research tool. It is not a medical device and must not be used for diagnosis.

## License

[AGPL-3.0](LICENSE). The planned training engine (YOLO, by Ultralytics) uses this license, and it extends to anyone distributing it alongside. You may use, modify and redistribute freely, keeping the same license.
