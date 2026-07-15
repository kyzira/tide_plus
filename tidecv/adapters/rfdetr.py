"""RF-DETR output adapter.

RF-DETR is a bounding-box detector. Its predictions usually arrive either as a
``supervision.Detections`` object (``.xyxy``, ``.confidence``, ``.class_id``) or as a
COCO-style results JSON. Both are supported; the live-model path imports ``rfdetr`` only
when called so it stays an optional dependency.
"""
from __future__ import annotations

import os

from ..data import Data
from .base import build_image_index, resolve_image_id, map_class
from . import coco as _coco


def from_json(path: str, name: str | None = None) -> Data:
    """RF-DETR exported to a COCO-style results JSON — same as any COCO detector output."""
    return _coco.from_json(path, name)


def from_detections(detections, image_id, gt: Data | None = None,
                    class_map: dict | None = None, data: Data | None = None,
                    name: str | None = None) -> Data:
    """Add one image's ``supervision.Detections`` to a ``Data`` (created if not given).

    ``image_id`` is used directly as the GT image id (pass the id you registered in the GT).
    Boxes come in as xyxy (absolute pixels) and are stored as ``[x, y, w, h]``.
    """
    if data is None:
        data = Data(name or "rfdetr")

    xyxy = detections.xyxy
    conf = getattr(detections, "confidence", None)
    class_id = getattr(detections, "class_id", None)

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        score = 1.0 if conf is None else float(conf[i])
        cls = 0 if class_id is None else int(class_id[i])
        gt_cls = map_class(class_map, cls)
        data.add_detection(image_id, gt_cls, score, box=[x1, y1, x2 - x1, y2 - y1], mask=None)
    return data


def from_model(model, images, gt: Data, class_map: dict | None = None,
               threshold: float = 0.5, name: str | None = None) -> Data:
    """Run a live RF-DETR model over ``images`` (paths) and collect boxes into ``Data``.

    ``model`` may be an ``rfdetr`` model instance or a weights path (loaded as RFDETRBase).
    """
    if isinstance(model, str):
        from rfdetr import RFDETRBase  # optional dependency, imported on demand
        model = RFDETRBase(pretrain_weights=model)

    from PIL import Image
    data = Data(name or "rfdetr")
    index = build_image_index(gt)

    if isinstance(images, (str, os.PathLike)):
        images = [images]

    for img_path in images:
        image_id = resolve_image_id(index, os.path.basename(str(img_path)))
        if image_id is None:
            continue
        detections = model.predict(Image.open(img_path).convert("RGB"), threshold=threshold)
        from_detections(detections, image_id, gt=gt, class_map=class_map, data=data)
    return data
