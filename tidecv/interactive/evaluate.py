"""Interactive click-efficiency benchmark.

For every GT instance and every registered model, simulate the robot-user click loop
(see :mod:`clicker`) and record how the mask IoU improves click-by-click. Aggregate into
the standard interactive-segmentation metrics:

* **NoC@t**  — mean Number-of-Clicks to reach IoU ≥ t (e.g. 85 %, 90 %).
* **NoF@t**  — Number-of-Failures: instances that never reach t within ``max_clicks``.
* **IoU curve** — mean IoU after 1, 2, … ``max_clicks`` clicks.
* **per-class** NoC / NoF — does one model handle a defect class better than another?
"""
from __future__ import annotations

import glob
import os

import numpy as np

from ..data import Data
from .clicker import first_click, next_click
from .gt_masks import instances_by_image, iou as mask_iou


class InteractiveEvaluator:
    def __init__(self, gt: Data, image_dir: str, target_iou: float = 0.90,
                 max_clicks: int = 20, iou_targets=(0.85, 0.90),
                 image_dims: dict | None = None, max_instances: int | None = None):
        self.gt = gt
        self.image_dir = image_dir
        self.max_clicks = int(max_clicks)
        # Targets to report NoC/NoF for; the primary (for NoF headline) is the largest.
        self.iou_targets = tuple(sorted(set(iou_targets) | {target_iou}))
        self.primary_target = max(self.iou_targets)
        self.image_dims = image_dims
        self.max_instances = max_instances

        self.models: dict = {}
        self.report: dict | None = None

    # ------------------------------------------------------------------ setup
    def add_model(self, name: str, model) -> None:
        model.name = name
        self.models[name] = model

    def _image_path(self, image_id) -> str:
        name = self.gt.images[image_id]["name"]
        cand = os.path.join(self.image_dir, os.path.basename(str(name)))
        if os.path.isfile(cand):
            return cand
        stem = os.path.splitext(os.path.basename(str(name)))[0]
        hits = glob.glob(os.path.join(self.image_dir, stem + ".*"))
        if hits:
            return hits[0]
        raise FileNotFoundError(f"image for id {image_id!r} ({name!r}) not found in {self.image_dir!r}")

    # ------------------------------------------------------------------ core loop
    def _run_instance(self, model, gt_mask) -> list:
        """Return the IoU after each click: ious[k] = IoU after k+1 clicks."""
        clicks = [first_click(gt_mask)]
        mask, _ = model.predict(clicks)
        ious = [mask_iou(mask, gt_mask)]

        while len(clicks) < self.max_clicks and ious[-1] < self.primary_target:
            nc = next_click(mask, gt_mask)
            if nc is None:  # prediction already equals GT everywhere the clicker can act
                break
            clicks.append(nc)
            mask, _ = model.predict(clicks)
            ious.append(mask_iou(mask, gt_mask))
        return ious

    def _noc(self, ious: list, target: float):
        """1-based click count to first reach target; None if never within max_clicks."""
        for k, v in enumerate(ious, start=1):
            if v >= target:
                return k
        return None

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        instances = instances_by_image(self.gt, self.image_dims)
        # Flatten with a stable order; optionally cap for quick runs.
        flat = [(img_id, ann, m) for img_id in instances for (ann, m) in instances[img_id]]
        if self.max_instances is not None:
            flat = flat[: self.max_instances]
        # Re-group capped set by image so we still encode each image once.
        by_image: dict = {}
        for img_id, ann, m in flat:
            by_image.setdefault(img_id, []).append((ann, m))

        report = {}
        for name, model in self.models.items():
            per_instance = []
            for img_id, insts in by_image.items():
                from PIL import Image
                image = Image.open(self._image_path(img_id)).convert("RGB")
                model.set_image(image)
                for ann, gt_mask in insts:
                    ious = self._run_instance(model, gt_mask)
                    per_instance.append({"class": ann["class"], "ious": ious})
            report[name] = self._aggregate(per_instance)
        self.report = report
        return report

    # ------------------------------------------------------------------ aggregate
    def _aggregate(self, per_instance: list) -> dict:
        n = len(per_instance)
        maxc = self.max_clicks
        out = {
            "num_instances": n,
            "max_clicks": maxc,
            "NoC": {},
            "NoF": {},
            "iou_curve": [0.0] * maxc,
            "mean_iou_final": 0.0,
            "per_class": {},
        }
        if n == 0:
            return out

        # NoC / NoF per target (failures counted as max_clicks in the NoC mean).
        for t in self.iou_targets:
            key = str(int(round(t * 100)))
            nocs, fails = [], 0
            for inst in per_instance:
                k = self._noc(inst["ious"], t)
                if k is None:
                    fails += 1
                    nocs.append(maxc)
                else:
                    nocs.append(k)
            out["NoC"][key] = round(float(np.mean(nocs)), 2)
            out["NoF"][key] = fails

        # IoU curve: pad each instance's IoU list to max_clicks with its last value
        # (once the loop stops, the mask/IoU is held constant).
        padded = []
        finals = []
        for inst in per_instance:
            ious = inst["ious"]
            last = ious[-1] if ious else 0.0
            padded.append(ious + [last] * (maxc - len(ious)))
            finals.append(last)
        arr = np.array(padded)  # [n, maxc]
        out["iou_curve"] = [round(float(v), 4) for v in arr.mean(axis=0)]
        out["mean_iou_final"] = round(float(np.mean(finals)), 4)

        # Per-class NoC/NoF at the primary target.
        pkey = str(int(round(self.primary_target * 100)))
        by_class: dict = {}
        for inst in per_instance:
            by_class.setdefault(inst["class"], []).append(inst)
        for cls_id, insts in by_class.items():
            cls_name = self.gt.classes.get(cls_id, f"class {cls_id}")
            nocs, fails = [], 0
            for inst in insts:
                k = self._noc(inst["ious"], self.primary_target)
                if k is None:
                    fails += 1
                    nocs.append(maxc)
                else:
                    nocs.append(k)
            out["per_class"][cls_name] = {
                f"NoC@{pkey}": round(float(np.mean(nocs)), 2),
                f"NoF@{pkey}": fails,
                "n": len(insts),
            }
        return out

    # ------------------------------------------------------------------ output
    def save_report(self, path: str) -> None:
        from .. import functions as f
        if self.report is None:
            raise RuntimeError("call run() before save_report()")
        f.save_json(self.report, path)

    def plot(self, out_dir: str) -> None:
        from .report import plot_report
        if self.report is None:
            raise RuntimeError("call run() before plot()")
        plot_report(self.report, out_dir, iou_targets=self.iou_targets,
                    primary_target=self.primary_target)

    def print_report(self) -> None:
        if self.report is None:
            raise RuntimeError("call run() before print_report()")
        for name, r in self.report.items():
            print(f"\n-- {name} --  ({r['num_instances']} instances, max {r['max_clicks']} clicks)")
            for t, v in r["NoC"].items():
                print(f"  NoC@{t}: {v:.2f}   NoF@{t}: {r['NoF'][t]}")
            print(f"  mean final IoU: {r['mean_iou_final']:.3f}")
