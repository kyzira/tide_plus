"""Tests for the model-output adapters (self-contained; no external data needed)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from tidecv import adapters
from tidecv.data import Data


def _make_gt():
    gt = Data("gt")
    gt.add_class(0, "A")
    gt.add_class(1, "B")
    gt.add_image(7, "000042.jpg")  # image id 7, file 000042.jpg
    return gt


def test_yolo_detect_line(tmp_path):
    """A YOLO detection line denormalizes to the right [x,y,w,h] box and image id."""
    from PIL import Image

    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    Image.new("RGB", (200, 100)).save(img_dir / "000042.jpg")
    # class 1, centre (0.5,0.5), size (0.2,0.4) -> box [80,30,40,40], conf 0.8
    (lbl_dir / "000042.txt").write_text("1 0.5 0.5 0.2 0.4 0.8\n")

    gt = _make_gt()
    preds = adapters.yolo.from_label_dir(str(lbl_dir), gt, str(img_dir), task="detect")

    assert len(preds.annotations) == 1
    det = preds.annotations[0]
    assert det["image"] == 7
    assert det["class"] == 1
    assert det["score"] == pytest.approx(0.8)
    assert det["bbox"] == pytest.approx([80.0, 30.0, 40.0, 40.0])


def test_yolo_seg_line_to_rle(tmp_path):
    """A YOLO polygon line becomes a COCO RLE plus a tight bounding box."""
    from PIL import Image

    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    Image.new("RGB", (100, 100)).save(img_dir / "000042.jpg")
    # square polygon from (10,10) to (30,30) normalized on a 100x100 image
    (lbl_dir / "000042.txt").write_text("0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n")

    gt = _make_gt()
    preds = adapters.yolo.from_label_dir(str(lbl_dir), gt, str(img_dir), task="seg")

    assert preds.has_masks() is True
    det = preds.annotations[0]
    assert det["class"] == 0
    assert isinstance(det["mask"], dict) and det["mask"].get("counts")
    assert det["bbox"] == pytest.approx([10.0, 10.0, 20.0, 20.0])


def test_yolo_class_map(tmp_path):
    """class_map remaps YOLO indices onto GT category ids."""
    from PIL import Image

    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    Image.new("RGB", (100, 100)).save(img_dir / "000042.jpg")
    (lbl_dir / "000042.txt").write_text("5 0.5 0.5 0.2 0.2\n")  # yolo class 5 -> gt 1

    gt = _make_gt()
    preds = adapters.yolo.from_label_dir(str(lbl_dir), gt, str(img_dir),
                                         task="detect", class_map={5: 1})
    assert preds.annotations[0]["class"] == 1


def test_yolo_auto_task_distinguishes(tmp_path):
    """auto mode: 4-5 numbers -> box, longer -> polygon."""
    from PIL import Image

    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    Image.new("RGB", (100, 100)).save(img_dir / "000042.jpg")
    (lbl_dir / "000042.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n"                       # box
        "1 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n"       # polygon
    )
    gt = _make_gt()
    preds = adapters.yolo.from_label_dir(str(lbl_dir), gt, str(img_dir), task="auto")
    masks = [a["mask"] for a in preds.annotations]
    assert masks[0] is None                 # box line -> no mask
    assert isinstance(masks[1], dict)       # polygon line -> RLE


def test_rfdetr_from_detections():
    """supervision.Detections-like object maps to [x,y,w,h] boxes."""
    class Det:
        xyxy = [[10.0, 20.0, 40.0, 60.0]]
        confidence = [0.7]
        class_id = [1]

    gt = _make_gt()
    data = adapters.rfdetr.from_detections(Det(), image_id=7, gt=gt)
    det = data.annotations[0]
    assert det["image"] == 7
    assert det["class"] == 1
    assert det["score"] == pytest.approx(0.7)
    assert det["bbox"] == pytest.approx([10.0, 20.0, 30.0, 40.0])
