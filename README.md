# histocel

A command line tool to train and use models that **outline, classify and count cells** in microscopy images, with your own data.

histocel ships no images, no classes and no pretrained models. You point it at your slides and annotations, declare your classes, and run.

> **Status: pre-alpha.** You can create a project, import annotations (COCO, YOLO-seg, QuPath GeoJSON) and check it for problems, build the training dataset and train a model. Whole-slide inference and evaluation are under construction. See the [roadmap](#roadmap).

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

histocel synth demo          # a complete project of synthetic slides
histocel import -p demo      # annotations -> canonical format
histocel check -p demo       # problems to fix before training
histocel build -p demo       # tiled training dataset + manifest
```

`synth` creates six synthetic slides in `demo/` (coloured ellipses on a pinkish background), a COCO export with the exact outline of every cell, and a `histocel.yaml`. They look nothing like real tissue: they exist to show the tool working and to test the code against a known answer.

## With your own data

```bash
histocel init my-project
```

Put your images in `my-project/raw/images/` and your annotation exports in `my-project/raw/annotations/`, then edit `my-project/histocel.yaml`: list your classes (with the spellings your exports use as aliases) and your annotation sources. Run `histocel import` and `histocel check` from inside the folder.

| Format | `format:` | Notes |
|---|---|---|
| COCO instance segmentation | `coco` | One JSON file. Polygon outlines only; RLE masks are skipped. |
| YOLO segmentation | `yolo-seg` | A folder of `.txt` files. Needs `names: [...]` in the source. |
| QuPath GeoJSON | `qupath` | One `.geojson` per image, named after the image. |

`check` stops you before the mistakes that are expensive to find later: a class name nobody declared, slides of one animal on both sides of the split, a test slide whose outlines no person ever reviewed, an overlap smaller than your largest cell.

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

## The training dataset

`histocel build` writes `datasets/default/` in YOLO segmentation layout: overlapping tiles of the training and validation slides, their labels, a `data.yaml` and a `manifest.json`.

- **Test slides are left out entirely.** They are only read later, as whole slides, by evaluation.
- **Cells on a tile border are cut by real geometry**, and with an overlap larger than your largest cell every cell is whole in at least one tile.
- **A share of the empty tiles is kept** so the model also sees plain background.
- **The manifest holds a hash of the content.** Same annotations and settings, same hash, on any machine. Treat the folder as disposable: never edit it, rebuild it.

## Training

Training needs the optional engine, which pulls in PyTorch:

```bash
pip install "histocel[train] @ git+https://github.com/ErickGCA/histocel"

histocel train --name first-try      # settings come from the `training` section of histocel.yaml
histocel runs                        # list recorded runs
```

Without a GPU, use [notebooks/train_on_colab.ipynb](notebooks/train_on_colab.ipynb): it installs histocel and runs the same commands on a free Colab GPU, with your project folder on Google Drive.

Each run gets a folder under `runs/`:

- `run.json` records the dataset hash, the split, the settings and the histocel and engine versions.
- `model/` holds the weights and a `model.json` with the classes and tile size the model expects, so it cannot be used with the wrong ones.

The scores printed by `train` are measured on validation tiles and only steer training. The final score comes from whole test slides, in the evaluation stage.

## Roadmap

| Stage | Delivers | Status |
|---|---|---|
| 1 | Pure core, synthetic slides, tests and CI | done |
| 2 | `init`, `import` (COCO, YOLO-seg, QuPath GeoJSON) and `check` | done |
| 3 | `build`: split by slide and group, tiling, manifest | done |
| 4 | `train`: training engine, run records, Colab notebook | done |
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
