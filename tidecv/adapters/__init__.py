"""Model-output adapters: normalize any supported model's output into ``tidecv.Data``.

Each submodule understands one output format and speaks the common ``Data`` interface, so
that heterogeneous models can be compared on the same ground truth::

    from tidecv import TIDE, datasets, adapters

    gt   = datasets.COCO(r"...\\GT.json")
    yolo = adapters.yolo.from_label_dir(r"...\\preds\\labels", gt, image_dir=r"...\\images")
    rfd  = adapters.coco.from_json(r"...\\rfdetr_results.json")

    tide = TIDE()
    tide.evaluate_multiple_models_on_one_gt(gt, [yolo, rfd], names=["YOLO", "RF-DETR"])

Available submodules:
    - :mod:`coco`   — COCO-style results JSON (also RF-DETR / any COCO exporter)
    - :mod:`yolo`   — Ultralytics label dirs or a live YOLO model
    - :mod:`rfdetr` — RF-DETR (supervision.Detections / live model / COCO JSON)
"""
from . import base
from . import coco
from . import yolo
from . import rfdetr

__all__ = ["base", "coco", "yolo", "rfdetr"]
