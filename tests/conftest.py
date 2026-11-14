import json

import pytest
from PIL import Image

from histocel.project import CONFIG_NAME
from histocel.sources.coco import to_coco
from histocel.synthetic import generate_slide


@pytest.fixture
def make_project(tmp_path):
    """Build a small project on disk from synthetic slides and a config text.

    Returns ``(root, annotations)``. The annotations are the exact answer key; a COCO
    export of them is written to ``raw/annotations/all.coco.json``.
    """

    def build(config: str, slides: int = 4, cells: int = 12):
        root = tmp_path / "project"
        images = root / "raw" / "images"
        annotations_dir = root / "raw" / "annotations"
        images.mkdir(parents=True)
        annotations_dir.mkdir(parents=True)
        annotations = []
        for index in range(slides):
            image, annotation = generate_slide(
                f"s{index + 1}", width=320, height=240, n_cells=cells, seed=index
            )
            image.save(images / annotation.image)
            annotations.append(annotation)
        (annotations_dir / "all.coco.json").write_text(
            json.dumps(to_coco(annotations)), encoding="utf-8"
        )
        (root / CONFIG_NAME).write_text(config, encoding="utf-8")
        return root, annotations

    return build


@pytest.fixture
def blank_image():
    def make(path, size=(100, 80)):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (240, 200, 220)).save(path)
        return path

    return make
