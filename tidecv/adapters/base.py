"""Shared machinery for turning a model's native output into a ``tidecv.Data`` object.

Every adapter ultimately produces a :class:`tidecv.data.Data` of *detections* that can be
fed straight into ``TIDE.evaluate(gt, preds)``. Two cross-cutting concerns live here:

* **Image alignment** — a model's output references images by file name (or a bare stem),
  but ``Data`` / TIDE reference them by the ground-truth image *id*. ``build_image_index``
  maps ``file stem -> image_id`` from the GT so any adapter can resolve its predictions.
* **Class alignment** — a model's class indices rarely equal the GT category ids. A
  ``class_map`` ({model_class -> gt_category_id}) handles the remap; the default is identity.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

from ..data import Data


def build_image_index(gt: Data) -> dict:
    """Map a file-name *stem* (``000000``) and full name (``000000.jpg``) to the GT image id."""
    index = {}
    for image_id, info in gt.images.items():
        name = info.get("name")
        if name is None:
            continue
        base = os.path.basename(str(name))
        index[base] = image_id
        index[os.path.splitext(base)[0]] = image_id
    return index


def resolve_image_id(index: dict, file_name: str):
    """Look a prediction's image up in the index by full name then by stem. None if unknown."""
    base = os.path.basename(str(file_name))
    if base in index:
        return index[base]
    stem = os.path.splitext(base)[0]
    return index.get(stem)


def map_class(class_map, model_class: int) -> int:
    """Remap a model class index to a GT category id (identity when no map is given)."""
    if class_map is None:
        return model_class
    if model_class not in class_map:
        raise KeyError(
            f"model class {model_class!r} is not in class_map (keys: {sorted(class_map)})"
        )
    return class_map[model_class]


class PredictionAdapter(ABC):
    """Base class: subclasses implement :meth:`to_data` returning a predictions ``Data``."""

    def __init__(self, name: str | None = None):
        self._name = name

    @abstractmethod
    def to_data(self) -> Data:  # pragma: no cover - interface
        ...

    def __call__(self) -> Data:
        return self.to_data()
