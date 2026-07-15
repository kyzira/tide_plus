"""Regression tests for the core-engine bug fixes.

Run with:  pytest tests/ -q
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tidecv import TIDE
from tidecv.data import Data
from tidecv.ap import ClassedAPDataObject, APDataObject


# --------------------------------------------------------------------------- #
# Bug 1: get_mAP must not divide by zero when every class is empty.
# --------------------------------------------------------------------------- #
def test_get_map_empty_no_zerodivision():
    obj = ClassedAPDataObject()
    assert obj.get_mAP() == 0.0  # no classes at all

    # A class that was registered but never received data / positives.
    obj.objs[0] = APDataObject()
    assert obj.get_mAP() == 0.0
    assert obj.get_precision() == 0
    assert obj.get_recall() == 0


# --------------------------------------------------------------------------- #
# Bug 2: has_masks must detect polygon-list masks, not only RLE dicts.
# --------------------------------------------------------------------------- #
def test_has_masks_detects_polygons():
    d = Data("poly")
    d.add_ground_truth(1, 0, box=[0, 0, 10, 10], mask=[[0, 0, 10, 0, 10, 10, 0, 10]])
    assert d.has_masks() is True


def test_has_masks_detects_rle():
    d = Data("rle")
    d.add_ground_truth(1, 0, box=[0, 0, 10, 10],
                       mask={"counts": b"abc", "size": [20, 20]})
    assert d.has_masks() is True


def test_has_masks_false_for_boxes_only():
    d = Data("box")
    d.add_ground_truth(1, 0, box=[0, 0, 10, 10], mask=None)
    assert d.has_masks() is False


def test_has_masks_false_for_empty_masks():
    d = Data("empty")
    d.add_ground_truth(1, 0, box=[0, 0, 10, 10], mask=[])          # empty polygon
    d.add_ground_truth(1, 0, box=[0, 0, 10, 10], mask={})          # empty dict
    assert d.has_masks() is False


# --------------------------------------------------------------------------- #
# Bug 3: confusion matrix must match one-to-one and count missed GTs as FN.
# --------------------------------------------------------------------------- #
def _box(x, y, w=20, h=20):
    return [x, y, w, h]


def _run(gt, preds):
    tide = TIDE(mode=TIDE.BOX)
    tide.evaluate(gt, preds, name="m")
    return tide.runs["m"]


def test_confusion_matrix_counts_false_negatives():
    """Image has both a pred and GTs; one GT is left unmatched -> must be an FN row."""
    gt = Data("gt")
    gt.add_class(0, "A")
    gt.add_class(1, "B")
    gt.add_ground_truth(1, 0, box=_box(0, 0))      # will be matched
    gt.add_ground_truth(1, 1, box=_box(500, 500))  # far away -> missed (FN)

    preds = Data("pred")
    preds.add_detection(1, 0, 0.9, box=_box(0, 0))  # matches first GT only

    run = _run(gt, preds)
    labels = run.class_labels
    cm_counts = run.confusion_matrix  # normalized="true" -> row-normalized
    bg = labels.index("background")
    b_idx = labels.index("B")

    # Class B's entire GT row must point at background (missed), not vanish.
    assert cm_counts[b_idx, bg] == 1.0
    # Class A matched correctly.
    a_idx = labels.index("A")
    assert cm_counts[a_idx, a_idx] == 1.0


def test_confusion_matrix_one_to_one():
    """Two preds on one GT: only one may claim it; the other is a false positive."""
    gt = Data("gt")
    gt.add_class(0, "A")
    gt.add_ground_truth(1, 0, box=_box(0, 0))

    preds = Data("pred")
    preds.add_detection(1, 0, 0.9, box=_box(0, 0))    # best -> matches
    preds.add_detection(1, 0, 0.5, box=_box(2, 2))    # duplicate -> FP (background)

    run = _run(gt, preds)
    labels = run.class_labels
    a_idx = labels.index("A")
    bg = labels.index("background")

    # One true match (A->A) and one FP (background->A). Background row must be all FP.
    assert run.confusion_matrix[a_idx, a_idx] == 1.0
    assert run.confusion_matrix[bg, a_idx] == 1.0


# --------------------------------------------------------------------------- #
# Bug 5: average_out_summary averages each key by how many runs reported it.
# --------------------------------------------------------------------------- #
def test_average_out_summary_per_key_counts():
    tide = TIDE()
    # Two runs with a shared key and run-specific per-class keys.
    tide.summary = {
        "run1": {"mAP 50:95": 40.0, "Per-Class AP": {"A": 50.0, "B": 30.0}},
        "run2": {"mAP 50:95": 60.0, "Per-Class AP": {"A": 70.0}},  # no "B"
    }
    tide.runs = {}  # skip confusion-matrix averaging
    tide.average_out_summary()
    avg = tide.summary["Combined Average"]

    assert avg["mAP 50:95"] == 50.0                 # (40+60)/2
    assert avg["Per-Class AP"]["A"] == 60.0         # (50+70)/2
    assert avg["Per-Class AP"]["B"] == 30.0         # only run1 had B -> /1, not /2


def test_average_out_summary_skips_previous_average():
    tide = TIDE()
    tide.summary = {
        "run1": {"mAP 50:95": 40.0},
        "run2": {"mAP 50:95": 60.0},
        "Combined Average": {"mAP 50:95": 999.0},  # stale; must be ignored
    }
    tide.runs = {}
    tide.average_out_summary()
    assert tide.summary["Combined Average"]["mAP 50:95"] == 50.0
