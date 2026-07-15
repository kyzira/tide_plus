"""Interactive click-efficiency benchmark for promptable segmentation models.

Answers "which (finetuned) SAM reaches a good mask in the fewest clicks" by simulating a
deterministic robot user and reporting the standard NoC@IoU / NoF metrics.

    from tidecv import datasets
    from tidecv.interactive import InteractiveEvaluator, Sam3LabelingModel

    gt = datasets.COCO(r"...\\GT.json")
    ev = InteractiveEvaluator(gt, image_dir=r"...\\images", target_iou=0.90, max_clicks=20)
    ev.add_model("SAM3",    Sam3LabelingModel(r"...\\models\\sam3"))
    ev.add_model("SAM2-v6", Sam3LabelingModel(r"...\\models\\sam2-v6"))
    ev.run(); ev.print_report(); ev.save_report("click_report.json"); ev.plot("ClickBench")

``DiskModel`` is a dependency-free stand-in for testing the harness without a GPU.
"""
from .clicker import Click, first_click, next_click
from .gt_masks import instances_by_image, decode_instance, iou
from .sam_model import SamModel, Sam3LabelingModel, DiskModel
from .evaluate import InteractiveEvaluator

__all__ = [
    "Click", "first_click", "next_click",
    "instances_by_image", "decode_instance", "iou",
    "SamModel", "Sam3LabelingModel", "DiskModel",
    "InteractiveEvaluator",
]
