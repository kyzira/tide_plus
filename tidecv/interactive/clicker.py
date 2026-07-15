"""Deterministic robot-user click simulator (RITM / SimpleClick / SAM-eval protocol).

Given the target GT mask and the model's current prediction, decide the next click:

* **First click** — positive, at the most-interior point of the GT mask (the pixel
  farthest from the mask boundary, i.e. the argmax of its distance transform).
* **Each later click** — look at the error between prediction and GT, split into the
  false-negative region (GT the model missed) and the false-positive region (mask that
  spilled outside GT). Take the *larger* error region and click its most-interior point:
  **positive** if it's a false negative, **negative** if it's a false positive.

Deterministic (no randomness) so model-vs-model comparisons are fair and reproducible.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import cv2

# A single simulated click. positive=True means "this pixel belongs to the object".
Click = namedtuple("Click", ["x", "y", "positive"])


def _most_interior_point(region: np.ndarray):
    """Return (x, y) of the pixel farthest inside a boolean region, or None if empty."""
    if not region.any():
        return None
    # distanceTransform needs a uint8 image; foreground = region.
    dt = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
    idx = int(np.argmax(dt))
    y, x = np.unravel_index(idx, dt.shape)
    return int(x), int(y)


def first_click(gt_mask: np.ndarray) -> Click:
    """The initial positive click at the GT mask's most-interior point."""
    pt = _most_interior_point(gt_mask.astype(bool))
    if pt is None:
        raise ValueError("cannot place a first click on an empty GT mask")
    return Click(pt[0], pt[1], True)


def next_click(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Click | None:
    """The next corrective click from the current error, or None if masks already match."""
    gt = gt_mask.astype(bool)
    pred = pred_mask.astype(bool)

    false_neg = np.logical_and(gt, np.logical_not(pred))   # missed GT -> positive click
    false_pos = np.logical_and(pred, np.logical_not(gt))   # spilled mask -> negative click

    fn_area = int(false_neg.sum())
    fp_area = int(false_pos.sum())
    if fn_area == 0 and fp_area == 0:
        return None

    if fn_area >= fp_area:
        pt = _most_interior_point(false_neg)
        return Click(pt[0], pt[1], True) if pt else None
    pt = _most_interior_point(false_pos)
    return Click(pt[0], pt[1], False) if pt else None
