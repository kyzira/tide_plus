"""COCO-results adapter — the canonical prediction format.

Thin wrapper over :func:`tidecv.datasets.COCOResult` so that COCO-style detection JSON
(the format written by pycocotools / most detector eval scripts, and by RF-DETR export)
goes through the same ``adapters`` entry point as everything else.
"""
from __future__ import annotations

from ..data import Data
from .. import datasets


def from_json(path: str, name: str | None = None) -> Data:
    """Load a COCO-style results JSON (list of ``{image_id, category_id, score, ...}``)."""
    return datasets.COCOResult(path, name)
