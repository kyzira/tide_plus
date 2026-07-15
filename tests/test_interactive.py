"""Tests for the interactive click-efficiency benchmark (self-contained, no SAM/GPU)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from tidecv.data import Data
from tidecv.interactive import (
    Click, first_click, next_click, iou, decode_instance, instances_by_image,
    InteractiveEvaluator, DiskModel,
)


# --------------------------------------------------------------------------- #
# clicker
# --------------------------------------------------------------------------- #
def test_first_click_is_inside_gt():
    gt = np.zeros((100, 100), bool)
    gt[30:70, 40:60] = True
    c = first_click(gt)
    assert c.positive is True
    assert gt[c.y, c.x]  # click lands inside the mask


def test_next_click_positive_on_false_negative():
    gt = np.zeros((100, 100), bool)
    gt[20:80, 20:80] = True
    pred = np.zeros((100, 100), bool)  # predicts nothing -> whole GT is a false negative
    c = next_click(pred, gt)
    assert c.positive is True and gt[c.y, c.x]


def test_next_click_negative_on_false_positive():
    gt = np.zeros((100, 100), bool)          # nothing is GT
    pred = np.zeros((100, 100), bool)
    pred[20:80, 20:80] = True                # model over-segments -> false positive
    c = next_click(pred, gt)
    assert c.positive is False and pred[c.y, c.x]


def test_next_click_none_when_perfect():
    gt = np.zeros((50, 50), bool)
    gt[10:20, 10:20] = True
    assert next_click(gt.copy(), gt) is None


def test_iou_basic():
    a = np.zeros((10, 10), bool); a[:5, :] = True
    b = np.zeros((10, 10), bool); b[:5, :] = True
    assert iou(a, b) == 1.0
    b[:] = False
    assert iou(a, b) == 0.0


# --------------------------------------------------------------------------- #
# gt_masks
# --------------------------------------------------------------------------- #
def test_decode_polygon_instance():
    ann = {"mask": [[10, 10, 30, 10, 30, 30, 10, 30]], "ignore": False}
    m = decode_instance(ann, height=50, width=50)
    assert m.dtype == bool and m.shape == (50, 50)
    assert 350 <= m.sum() <= 450  # ~20x20 square


def test_instances_by_image_groups_and_skips_ignore():
    gt = Data("gt")
    gt.add_class(0, "A")
    gt.add_image(1, "a.jpg")
    gt.add_ground_truth(1, 0, box=[10, 10, 20, 20], mask=[[10, 10, 30, 10, 30, 30, 10, 30]])
    gt.add_ignore_region(1, 0, box=[0, 0, 5, 5], mask=[[0, 0, 5, 0, 5, 5, 0, 5]])
    grouped = instances_by_image(gt, image_dims={1: (50, 50)})
    assert set(grouped.keys()) == {1}
    assert len(grouped[1]) == 1  # ignore region excluded


# --------------------------------------------------------------------------- #
# converging model -> exercises the "reached target" NoC path
# --------------------------------------------------------------------------- #
class NearestClickModel:
    """Each pixel takes the polarity of its nearest click. Converges to any target as
    corrective clicks accumulate. Cheap enough for small test masks."""
    def set_image(self, image):
        arr = np.asarray(image)
        self.h, self.w = arr.shape[:2]

    def predict(self, clicks):
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        best_d = np.full((self.h, self.w), np.inf)
        best_pos = np.zeros((self.h, self.w), bool)
        for c in clicks:
            d = (xx - c.x) ** 2 + (yy - c.y) ** 2
            closer = d < best_d
            best_pos[closer] = c.positive
            best_d[closer] = d[closer]
        return best_pos, 1.0


def _evaluator():
    # gt/image_dir unused because we call _run_instance directly.
    gt = Data("gt"); gt.add_class(0, "A")
    return InteractiveEvaluator(gt, image_dir=".", target_iou=0.85, max_clicks=20)


def test_run_instance_converges_and_iou_increases():
    gt = np.zeros((80, 80), bool)
    gt[20:60, 20:60] = True  # solid square target

    model = NearestClickModel()
    model.set_image(np.zeros((80, 80, 3), np.uint8))

    ev = _evaluator()
    ious = ev._run_instance(model, gt)

    # The corrective clicks must drive IoU up substantially from the first click...
    assert ious[-1] > ious[0] + 0.2
    assert ious[-1] > 0.6
    # ...and the "reached target" NoC path fires for an achievable target.
    k = ev._noc(ious, 0.5)
    assert k is not None and 1 <= k <= ev.max_clicks


def test_noc_and_aggregate():
    ev = _evaluator()
    # instance 1 reaches 0.9 on click 3; instance 2 never passes 0.6
    per_instance = [
        {"class": 0, "ious": [0.4, 0.7, 0.95]},
        {"class": 0, "ious": [0.3, 0.5, 0.6]},
    ]
    ev.iou_targets = (0.85, 0.90)
    ev.primary_target = 0.90
    agg = ev._aggregate(per_instance)

    assert agg["num_instances"] == 2
    # inst1 NoC@90 = 3, inst2 fails -> counted as max_clicks (20). mean = (3+20)/2 = 11.5
    assert agg["NoC"]["90"] == pytest.approx(11.5)
    assert agg["NoF"]["90"] == 1
    assert len(agg["iou_curve"]) == ev.max_clicks
    assert "A" in agg["per_class"]
    assert agg["per_class"]["A"]["n"] == 2


def test_diskmodel_plumbing():
    """DiskModel + evaluator internals run without error and produce a valid report shape."""
    gt = np.zeros((120, 120), bool)
    gt[40:80, 40:80] = True
    model = DiskModel(radius=25)
    model.set_image(np.zeros((120, 120, 3), np.uint8))
    ev = _evaluator()
    ious = ev._run_instance(model, gt)
    assert 0.0 <= ious[-1] <= 1.0 and len(ious) >= 1
