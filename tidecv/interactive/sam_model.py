"""Pluggable segmentation models for the interactive benchmark.

The evaluator only needs a small interface:

    model.set_image(image)                 # image: PIL.Image or H×W×3 ndarray
    mask, score = model.predict(clicks)    # clicks: list[Click]; mask: bool H×W

Two implementations ship here:

* :class:`Sam3LabelingModel` — real SAM 2 / SAM 3 inference, reusing the family-neutral
  ``SamBackend`` from the ``sam3_labeling`` project (auto-detects the family from the
  snapshot). This is what you point at each finetuned checkpoint on a GPU box.
* :class:`DiskModel` — a dependency-free deterministic stand-in that grows a mask from
  positive clicks and carves it back with negative ones. It is not a real segmenter; it
  exists so the harness (clicker loop + metrics) can be tested without heavy inference.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from .clicker import Click


def _image_hw(image):
    """(height, width) for a PIL image or an ndarray."""
    if hasattr(image, "size") and not hasattr(image, "shape"):  # PIL.Image
        w, h = image.size
        return h, w
    arr = np.asarray(image)
    return arr.shape[0], arr.shape[1]


class SamModel:
    """Interface every benchmark model implements (duck-typed; subclassing optional)."""

    name: str = "sam"

    def set_image(self, image) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, clicks):  # pragma: no cover - interface
        """Return (mask: bool H×W ndarray, score: float) for the given click history."""
        raise NotImplementedError


# Default location of the reusable backend (override via arg or SAM3_LABELING_ROOT env).
_DEFAULT_SAM3_ROOT = os.environ.get("SAM3_LABELING_ROOT", r"C:\Code Python\sam3_labeling")


class Sam3LabelingModel(SamModel):
    """Real SAM 2 / SAM 3 model backed by ``sam3_labeling``'s ``SamBackend``.

    Args:
        model_dir: path to (or HF id of) the SAM snapshot / finetune.
        name: display name for this model in the report.
        sam3_labeling_root: repo root of the ``sam3_labeling`` project (so ``import src…``
            resolves). Defaults to ``SAM3_LABELING_ROOT`` env or the standard checkout path.
        device / dtype / family / crop_mode: passed straight to ``SamBackend``.
    """

    def __init__(self, model_dir: str, name: str | None = None,
                 sam3_labeling_root: str | None = None, device: str = "cuda",
                 dtype: str = "bfloat16", family: str = "auto", crop_mode: str = "full",
                 **backend_kwargs):
        self.model_dir = model_dir
        self.name = name or os.path.basename(os.path.normpath(model_dir))
        self._root = sam3_labeling_root or _DEFAULT_SAM3_ROOT
        self._device = device
        self._dtype = dtype
        self._family = family
        self._crop_mode = crop_mode
        self._backend_kwargs = backend_kwargs
        self._backend = None
        self._Point = None
        self._hw = None

    def _ensure_loaded(self):
        if self._backend is not None:
            return
        if self._root and self._root not in sys.path:
            sys.path.insert(0, self._root)
        from src.model.sam_backend import SamBackend
        from src.core.annotation import Point
        self._Point = Point
        self._backend = SamBackend(
            model_dir=self.model_dir, device=self._device, dtype=self._dtype,
            crop_mode=self._crop_mode, family=self._family, **self._backend_kwargs)
        self._backend.load_model()

    def set_image(self, image) -> None:
        self._ensure_loaded()
        from PIL import Image
        if not hasattr(image, "convert"):  # ndarray -> PIL
            image = Image.fromarray(np.asarray(image))
        self._hw = _image_hw(image)
        self._backend.set_image(image)

    def predict(self, clicks):
        self._ensure_loaded()
        pts = [self._Point(x=int(c.x), y=int(c.y), positive=bool(c.positive)) for c in clicks]
        result = self._backend.predict(pts)
        if result is None:  # no positive click yet -> empty mask
            h, w = self._hw
            return np.zeros((h, w), dtype=bool), 0.0
        return result.mask.astype(bool), float(result.score)


class DiskModel(SamModel):
    """Deterministic test stand-in: union of disks at positive clicks minus disks at
    negative clicks. Converges toward any target as the clicker adds corrective clicks."""

    name = "disk"

    def __init__(self, radius: int = 30, name: str | None = None):
        self.radius = radius
        if name:
            self.name = name
        self._hw = None

    def set_image(self, image) -> None:
        self._hw = _image_hw(image)

    def predict(self, clicks):
        h, w = self._hw
        yy, xx = np.ogrid[:h, :w]
        mask = np.zeros((h, w), dtype=bool)
        r2 = self.radius * self.radius
        for c in clicks:
            if c.positive:
                mask |= ((xx - c.x) ** 2 + (yy - c.y) ** 2) <= r2
        for c in clicks:
            if not c.positive:
                mask &= ~(((xx - c.x) ** 2 + (yy - c.y) ** 2) <= r2)
        return mask, 1.0
