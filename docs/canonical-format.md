# Canonical annotation format

histocel stores annotations as one JSON file per slide, in the project's `canonical/` folder. Every annotation tool is converted to this format on the way in, and every training format is produced from it.

Schema version: **1**.

## Example

```json
{
  "schema_version": 1,
  "slide_id": "slide-042",
  "image": "slide-042.png",
  "width": 960,
  "height": 1280,
  "group": "animal-17",
  "searched_labels": ["hepatocyte", "kupffer"],
  "cells": [
    {
      "label": "hepatocyte",
      "polygon": [[412.0, 300.5], [431.2, 298.0], [440.1, 318.7], [420.3, 330.0]],
      "method": "manual",
      "reviewed": false,
      "source": "coco:batch1.json",
      "annotator": "annotator-1",
      "box": null
    }
  ]
}
```

## Slide fields

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | integer | Version of this format. histocel refuses versions it does not know. |
| `slide_id` | string | Unique identifier of the slide in the project. It is also the file name. |
| `image` | string | Image file name, relative to `raw/images/`. |
| `width`, `height` | integers | Size of the original image, in pixels. |
| `group` | string or null | What makes slides look alike: animal, patient, tissue block. Slides of one group are never split between training and test. |
| `searched_labels` | list or null | Classes the annotator looked for on this slide. Null means every class of the project. Without it, a missing class could mean "absent" or "nobody looked". |
| `cells` | list | The annotated cells. |

## Cell fields

| Field | Type | Meaning |
|---|---|---|
| `label` | string | Official class name, the finest one the annotator marked. Grouping classes is a dataset decision. |
| `polygon` | list of `[x, y]` | Outline in pixels of the original image, origin at the top-left corner. Implicitly closed; at least 3 points. |
| `method` | string | How the outline was made: `manual` (a person drew it), `assisted` (a tool produced it from a human hint, such as a box) or `model` (a trained model predicted it). |
| `reviewed` | boolean | Whether a person checked the outline after it was made. |
| `source` | string or null | Where it came from, for example `coco:batch1.json`. |
| `annotator` | string or null | Who annotated it. |
| `box` | `[x0, y0, x1, y1]` or null | The original hint, when the outline is assisted. |

## Gold

A cell is **gold** when `method` is `manual` or `reviewed` is true. Only slides where every cell is gold may enter the test set: an outline made by a machine and never checked can train a model, but cannot grade one.

## Stability

histocel writes these files deterministically (sorted keys, `\n` line endings, coordinates with two decimals). Saving the same annotation twice produces byte-identical files, so they behave well under version control.
