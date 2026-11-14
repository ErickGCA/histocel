# User guide

This guide takes you from a folder of slide images and annotations to a trained model with a measured counting error. Every command also works on the synthetic project created by `histocel synth`, so you can follow along before touching your own data.

- [Install](#install)
- [Ideas you need](#ideas-you-need)
- [Step by step](#step-by-step)
- [histocel.yaml reference](#histocelyaml-reference)
- [Annotating faster with boxes](#annotating-faster-with-boxes)
- [Growing the dataset with the model's help](#growing-the-dataset-with-the-models-help)
- [Reading the evaluation](#reading-the-evaluation)
- [When something goes wrong](#when-something-goes-wrong)
- [Limits](#limits)

## Install

histocel needs Python 3.10 or newer.

```bash
pip install "histocel @ git+https://github.com/ErickGCA/histocel"
```

That is enough to create a project, import annotations, check them and build the dataset. Training, prediction and evaluation need the training engine, which pulls in PyTorch:

```bash
pip install "histocel[train] @ git+https://github.com/ErickGCA/histocel"
```

On a computer without a GPU, training is slow. Use [the Colab notebook](../notebooks/train_on_colab.ipynb): it installs histocel on a free GPU and runs the same commands on a project folder kept in your Google Drive.

## Ideas you need

**Project.** A folder of yours with a `histocel.yaml` file. Your images, annotations, datasets and models live there. histocel finds the project from the folder you are in, the way git finds a repository. Nothing of yours ever goes into the histocel repository.

**Canonical annotations.** Whatever tool you annotate with, `histocel import` converts its export into one JSON file per slide under `canonical/`. Each cell records its class, its outline in pixels and how the outline was made. See [canonical-format.md](canonical-format.md).

**Gold.** An outline is gold when a person drew it, or a person reviewed it after a tool or a model made it. Only gold outlines may grade a model.

**Group.** What makes slides look alike: the same animal, patient or tissue block. Slides of one group are kept on the same side of the split. If you ignore this, the score looks better than the model really is.

**Test slides.** Slides you set aside by hand under `split.test`. They are never used for training or for choosing the model, and they are the only slides `evaluate` looks at. Choose them once and leave them alone.

## Step by step

### 1. Create the project

```bash
histocel init my-project
cd my-project
```

Copy your slide images into `raw/images/` and the exports of your annotation tool into `raw/annotations/`. The name of an image file, without its extension, is the slide's id.

### 2. Describe your data

Edit `histocel.yaml`. At minimum, list your classes and where the annotations are:

```yaml
classes:
  hepatocyte:
    aliases: ["Hepatocyte ", "hepatocito normal"]
  kupffer:
    aliases: ["kupffer cell"]

sources:
  - format: coco
    path: raw/annotations/batch1.json
    annotator: ana
    group: animal-17

split:
  test: [slide-031, slide-032]
```

`aliases` are the spellings your exports actually contain. Case, accents and extra spaces are ignored, so `"Hepatócito Normal "` matches `hepatocito normal`.

### 3. Import and check

```bash
histocel import
histocel check
```

`import` stops if it finds a class name you did not declare, and lists all of them at once. Add each one as a class or as an alias and run it again.

`check` prints how many slides and cells fall on each side of the split, then errors and warnings. Errors must be fixed. Warnings tell you what will weaken the result, such as a class with very few examples or a tile overlap smaller than your largest cell.

### 4. Build the dataset

```bash
histocel build
```

This cuts the training and validation slides into overlapping tiles under `datasets/default/`. Treat the folder as disposable: never edit it, rebuild it. `manifest.json` holds a hash that identifies exactly what the dataset contains.

### 5. Train

```bash
histocel train --name first-try
histocel runs
```

Each run gets a folder under `runs/`, with a record of what was trained on which data and a `model/` folder that knows its own classes and tile size. The scores printed here are measured on validation tiles and only steer training.

### 6. Evaluate

```bash
histocel evaluate
```

This is the honest score: the model analyses the test slides whole, and the result is compared with their annotations. See [Reading the evaluation](#reading-the-evaluation).

### 7. Analyse new slides

```bash
histocel predict new-slides/ --out results
```

`results/` receives one marked image per slide, `counts.csv`, the predictions in canonical format and `predictions.coco.json`, which most annotation tools can open.

## histocel.yaml reference

```yaml
classes:                      # required
  <name>:
    aliases: [<spelling>, ...]

groupings:                    # optional: named ways of merging classes
  <grouping>:
    <merged-name>: [<class>, ...]

sources:                      # one entry per annotation export
  - format: coco | yolo-seg | qupath | yolo-bbox
    path: <file or folder, relative to this file>
    names: [<class>, ...]     # yolo-seg and yolo-bbox: class names by number
    annotator: <who>
    group: <default group for the slides in this export>
    searched: [<class>, ...]  # classes the annotator looked for; default: all
    method: manual | assisted | model     # default: manual
    reviewed: true | false                # default: false
    assist: sam               # yolo-bbox only: turn boxes into outlines
    assist_model: mobile_sam.pt

slides:                       # optional: group of individual slides
  <slide-id>: {group: <group>}

split:
  test: [<slide-id>, ...]     # set aside by hand; never trained on
  val_fraction: 0.2
  seed: 0

tiling:
  size: 320                   # tile side in pixels
  overlap: 64                 # keep larger than your largest cell

dataset:
  grouping: <grouping>        # train on merged classes
  min_visible: 0.25           # share of a cell that must be inside a tile
  background: 0.1             # share of empty tiles kept as background

training:
  engine: ultralytics
  model: yolo11n-seg.pt
  epochs: 100
  image_size: 640             # tiles are resized to this
  batch: 16
  patience: 30
  seed: 0
```

A key that histocel does not know is an error, so a typo never silently becomes a default.

### Annotation formats

| `format` | What it is | Notes |
|---|---|---|
| `coco` | COCO instance segmentation JSON | Polygon outlines only. RLE masks are skipped and reported. |
| `yolo-seg` | Folder of YOLO segmentation `.txt` files | Needs `names`. Images must be in `raw/images/`. |
| `qupath` | QuPath GeoJSON, one file per image | The class is read from each object's classification. Unclassified objects are skipped and reported. |
| `yolo-bbox` | Folder of YOLO detection `.txt` files | Boxes, not outlines. Needs `names` and `assist`. |

## Annotating faster with boxes

Drawing a box around a cell is several times faster than drawing its outline. If your annotations are boxes, histocel can turn each one into an outline with a Segment Anything model:

```bash
pip install "histocel[sam] @ git+https://github.com/ErickGCA/histocel"
```

```yaml
sources:
  - format: yolo-bbox
    path: raw/annotations/boxes
    names: [hepatocyte, kupffer]
    assist: sam
```

The outlines are stored as `assisted` and not reviewed, with the original box kept beside them. They can train a model. They cannot grade one: `check`, `build` and `evaluate` refuse a test slide that contains them. Once a person has looked at the outlines of a batch and corrected what was wrong, set `reviewed: true` on that source.

The class of each cell still comes from the person who drew the box. Only the shape is made by the tool.

## Growing the dataset with the model's help

Once you have a first model, let it do the first draft of new annotations:

1. Run `histocel predict` on slides nobody annotated.
2. Open `predictions.coco.json` in your annotation tool. Delete what is wrong, fix what is crooked, add what is missing.
3. Export the corrected annotations and add them as a new source with `method: model` and `reviewed: true`.
4. Run `import`, `build` and `train` again.

Correcting a draft takes much less time than starting from nothing, and the gap grows as the model improves. Keep the test slides out of this loop, and compare runs on the same test slides: that is what makes the comparison fair.

## Reading the evaluation

`evaluate` answers four separate questions, because one combined score would hide which part failed.

| Question | Metrics | What a bad value means |
|---|---|---|
| Was each cell found, whatever its class? | `detection/precision`, `recall`, `f1` | Low recall: cells are being missed. Low precision: the model sees cells that are not there. |
| Among the cells found, was the class right? | `classification/accuracy`, `class/<name>/...` | The model finds cells but confuses classes. More examples of the confused classes help. |
| How well do the outlines overlap? | `outline/mean_iou` | Outlines are loose or tight. This matters if you measure cell size, less if you only count. |
| How far is the count from the real one? | `count/<name>/mean_absolute_error`, `mean_error`, `r2` | `mean_error` is the bias: negative means undercounting. `mean_absolute_error` is the typical miss per slide. |

Two things to keep in mind:

- **Counts can be right for the wrong reasons.** A model that misses ten cells and invents ten others has a perfect count and poor detection. Read counting and detection together.
- **The interval matters as much as the number.** Each metric comes with a 95% confidence interval from resampling whole slides. With a handful of test slides the interval is wide, and a difference between two models that falls inside it is not a difference you can trust.

Two outlines match when their overlap (IoU) is at least `--iou`, 0.5 by default. For very small cells a few pixels of boundary error can push a correct detection below 0.5; if detection looks worse than the marked images suggest, try `--iou 0.3` and report the threshold you used. Scores obtained with different thresholds do not compare.

## When something goes wrong

| Message | What to do |
|---|---|
| `these class names are not declared` | Add each name to `classes`, as a new class or as an alias. |
| `is only partly in the test set` | Slides of one group must stay together. Put the whole group in `split.test` or none of it. |
| `outline(s) made by a tool or model and never reviewed` | Review that batch and set `reviewed: true`, or take the slide out of `split.test`. |
| `tiling.overlap is ... but the largest cell spans ...` | Raise `tiling.overlap` to the suggested value. |
| `class ... has no examples in the training slides` | Move slides between test and training, or annotate more of that class. |
| `no slides left for validation` | Add slides or groups, or raise `split.val_fraction`. |
| `the ultralytics engine is not installed` | Install the `train` extra. |
| `this source holds boxes, not outlines` | Add `assist: sam` to that source. |
| `the model predicts ..., which matches neither the classes of this project` | The model was trained with other classes. Train a new one, or restore the grouping it was trained with. |

## Limits

- histocel is a research tool. It is not a medical device and must not be used for diagnosis.
- It has been exercised end to end on synthetic slides. On real tissue, expect to tune tile size, overlap and training settings, and expect to find rough edges.
- Images are read whole into memory. Gigapixel whole-slide images are not supported yet.
- An interrupted training run cannot be resumed yet.
- Mask mAP is not computed; Panoptic Quality is reported as a secondary number.
