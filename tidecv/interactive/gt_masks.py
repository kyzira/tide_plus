"""Decode ground-truth instances into per-instance binary masks.

The interactive benchmark measures, per GT *instance*, how many clicks a model needs to
reproduce that instance's mask. This module turns a ``tidecv.Data`` ground truth (whose
annotations carry COCO RLE or polygon segmentations) into ``(annotation, bool H×W mask)``
pairs, grouped by image so a model can encode each image once and reuse it.
"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils

from ..data import Data
from .. import functions as f


def _ann_to_rle(ann: dict, height: int | None = None, width: int | None = None):
    """Return a COCO RLE for an annotation's mask (RLE dict or polygon list)."""
    m = ann.get("mask")
    if m is None or (hasattr(m, "__len__") and len(m) == 0):
        return None
    if isinstance(m, dict):
        # Already an RLE. 'counts' may be a str (needs bytes for pycocotools).
        rle = dict(m)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
        return rle
    # Polygon list -> RLE (needs image dimensions).
    if height is None or width is None:
        raise ValueError("polygon mask needs image height/width to rasterize")
    return f.toRLE(m, width, height)


def decode_instance(ann: dict, height: int | None = None, width: int | None = None):
    """Decode one annotation's mask to a boolean H×W array (None if it has no mask)."""
    rle = _ann_to_rle(ann, height, width)
    if rle is None:
        return None
    return mask_utils.decode(rle).astype(bool)


def instances_by_image(gt: Data, image_dims: dict | None = None) -> dict:
    """Group decoded GT instance masks by image id.

    Args:
        gt: ground-truth ``Data``.
        image_dims: optional ``{image_id: (width, height)}`` needed only when masks are
            stored as polygons (RLE masks already carry their size).

    Returns:
        ``{image_id: [(annotation, bool_mask), ...]}`` for non-ignored instances that
        have a decodable mask.
    """
    image_dims = image_dims or {}
    out: dict = {}
    for ann in gt.annotations:
        if ann.get("ignore"):
            continue
        w = h = None
        if ann["image"] in image_dims:
            w, h = image_dims[ann["image"]]
        mask = decode_instance(ann, height=h, width=w)
        if mask is None:
            continue
        out.setdefault(ann["image"], []).append((ann, mask))
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boolean masks."""
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0
