"""YOLO (Ultralytics) output adapter.

Two ways in:

* :func:`from_label_dir` — read a directory of YOLO ``.txt`` files (one per image), the
  format written by ``yolo predict ... save_txt=True`` and by most YOLO training sets.
  Supports both detection (``cls cx cy w h [conf]``) and segmentation
  (``cls x1 y1 x2 y2 … [conf]``) lines, all coordinates normalized to [0, 1].
* :func:`from_model` — run a live ``ultralytics.YOLO`` model over images (optional; only
  imported when called, so ultralytics is not a hard dependency).

Image dimensions are needed to denormalize YOLO coordinates. They are read from the actual
image files in ``image_dir``; polygons are converted to COCO RLE (so both box- and
mask-mode evaluation work) and a tight bounding box is derived from the polygon.
"""
from __future__ import annotations

import glob
import os

from ..data import Data
from .. import functions as f
from .base import build_image_index, resolve_image_id, map_class


def _read_image_size(path: str):
    """Return (width, height) for an image, trying PIL then OpenCV."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.width, im.height
    except Exception:
        pass
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"could not read image for size: {path}")
    h, w = img.shape[:2]
    return w, h


def _find_image(image_dir: str, stem: str):
    """Find an image file in image_dir whose stem matches (any common extension)."""
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
                ".JPG", ".JPEG", ".PNG"):
        cand = os.path.join(image_dir, stem + ext)
        if os.path.isfile(cand):
            return cand
    hits = glob.glob(os.path.join(image_dir, stem + ".*"))
    return hits[0] if hits else None


def _parse_line(tokens: list, task: str):
    """Parse one YOLO label line -> (cls, kind, coords, score).

    kind is 'box' (coords = [cx,cy,w,h] normalized) or 'poly' (coords = flat xy normalized).
    score is None when the line carries no confidence column.
    """
    cls = int(float(tokens[0]))
    rest = [float(t) for t in tokens[1:]]

    resolved = task
    if task == "auto":
        # 4 (box) or 5 (box+conf) numbers -> detection; anything longer -> polygon.
        resolved = "detect" if len(rest) in (4, 5) else "seg"

    if resolved == "detect":
        if len(rest) == 5:
            return cls, "box", rest[:4], rest[4]
        if len(rest) == 4:
            return cls, "box", rest[:4], None
        raise ValueError(f"detection line needs 4 or 5 values, got {len(rest)}")

    # segmentation: an odd count means a trailing confidence value.
    score = None
    coords = rest
    if len(rest) % 2 == 1:
        score = rest[-1]
        coords = rest[:-1]
    if len(coords) < 6:
        raise ValueError(f"polygon needs >=3 points (6 values), got {len(coords)}")
    return cls, "poly", coords, score


def from_label_dir(label_dir: str, gt: Data, image_dir: str,
                   task: str = "auto", class_map: dict | None = None,
                   default_score: float = 1.0, name: str | None = None) -> Data:
    """Build a predictions ``Data`` from a directory of YOLO ``.txt`` files.

    Args:
        label_dir: directory of ``<stem>.txt`` YOLO label files.
        gt: the ground-truth ``Data`` (used to resolve image ids by file name).
        image_dir: directory holding the images (needed to denormalize coordinates).
        task: ``"auto"`` | ``"detect"`` | ``"seg"``.
        class_map: optional ``{yolo_class -> gt_category_id}`` remap (default: identity).
        default_score: score to assign when a line has no confidence column.
        name: name for the resulting predictions set.
    """
    data = Data(name or os.path.basename(os.path.normpath(label_dir)))
    index = build_image_index(gt)

    txt_files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
    if not txt_files:
        raise FileNotFoundError(f"no .txt label files found in {label_dir!r}")

    n_skipped = 0
    for txt in txt_files:
        stem = os.path.splitext(os.path.basename(txt))[0]
        image_id = resolve_image_id(index, stem)
        if image_id is None:
            n_skipped += 1
            continue

        img_path = _find_image(image_dir, stem)
        if img_path is None:
            raise FileNotFoundError(f"no image found for label {txt!r} in {image_dir!r}")
        W, H = _read_image_size(img_path)

        with open(txt, "r") as fh:
            for line in fh:
                tokens = line.split()
                if len(tokens) < 5:
                    continue
                cls, kind, coords, score = _parse_line(tokens, task)
                gt_cls = map_class(class_map, cls)
                score = default_score if score is None else score

                if kind == "box":
                    cx, cy, bw, bh = coords
                    box = [(cx - bw / 2) * W, (cy - bh / 2) * H, bw * W, bh * H]
                    data.add_detection(image_id, gt_cls, score, box=box, mask=None)
                else:
                    poly = [c * (W if i % 2 == 0 else H) for i, c in enumerate(coords)]
                    polys = [poly]  # COCO polygon format: list of polygons
                    box = f.polyToBox(polys)
                    rle = f.toRLE(polys, W, H)
                    data.add_detection(image_id, gt_cls, score, box=box, mask=rle)

    if n_skipped:
        print(f"[yolo adapter] {n_skipped} label file(s) had no matching GT image; skipped.")
    return data


def from_model(model, images, gt: Data, class_map: dict | None = None,
               name: str | None = None, **predict_kwargs) -> Data:
    """Run a live Ultralytics model over ``images`` and collect its output into ``Data``.

    ``model`` may be an ``ultralytics.YOLO`` instance or a path/str to weights.
    ``images`` is anything ``model.predict`` accepts (a dir, a glob, or a list of paths).
    """
    from ultralytics import YOLO  # optional dependency, imported on demand
    if not hasattr(model, "predict"):
        model = YOLO(model)

    data = Data(name or "yolo")
    index = build_image_index(gt)
    results = model.predict(images, verbose=False, **predict_kwargs)

    for res in results:
        image_id = resolve_image_id(index, os.path.basename(res.path))
        if image_id is None:
            continue
        boxes = res.boxes
        masks = getattr(res, "masks", None)
        n = 0 if boxes is None else len(boxes)
        for i in range(n):
            cls = int(boxes.cls[i].item())
            score = float(boxes.conf[i].item())
            gt_cls = map_class(class_map, cls)
            x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i].tolist())
            box = [x1, y1, x2 - x1, y2 - y1]

            rle = None
            if masks is not None and masks.xy is not None and i < len(masks.xy):
                poly = masks.xy[i].reshape(-1).tolist()  # absolute pixel coords
                if len(poly) >= 6:
                    H, W = res.orig_shape
                    rle = f.toRLE([poly], W, H)
                    box = f.polyToBox([poly])
            data.add_detection(image_id, gt_cls, score, box=box, mask=rle)
    return data
